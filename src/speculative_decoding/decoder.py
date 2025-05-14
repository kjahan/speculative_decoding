import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class SpeculativeDecoder:
    def __init__(self, draft_model_name="facebook/opt-125m", target_model_name="facebook/opt-350m"):
        # Initialize models and tokenizers
        self.draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
        
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        
        # Move models to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.draft_model.to(self.device)
        self.target_model.to(self.device)

    def run_draft_model(self, prompt, k=5):
        """Run the draft model to generate k speculative tokens."""
        draft_next_token_ids = []
        draft_tokens_probs = []

        for pos in range(k):
            # Tokenize the prompt
            encoded_prompt = self.draft_tokenizer(prompt, return_tensors='pt')
            encoded_prompt = {key: value.to(self.device) for key, value in encoded_prompt.items()}

            # Run the model and get the logits
            with torch.no_grad():
                outputs = self.draft_model(**encoded_prompt)
                logits = outputs.logits

            # Get next token probabilities
            next_token_logits = logits[0, -1, :]
            probabilities = F.softmax(next_token_logits, dim=-1)

            # Get top token
            top_k = 5
            top_token_probs, top_token_ids = torch.topk(probabilities, k=top_k)
            top_token = self.draft_tokenizer.decode([top_token_ids[0].item()])
            
            # Add predicted token to prompt
            prompt = prompt + top_token
            draft_next_token_ids.append(top_token_ids[0].item())
            draft_tokens_probs.append(probabilities.cpu().numpy())

        return {'draft_token_ids': draft_next_token_ids, 'probs': draft_tokens_probs}

    def get_target_all_token_probabilities(self, prompt):
        """Get probability distributions for each token position from target model."""
        inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.target_model(**inputs).logits

        logits = logits.squeeze(0)
        input_ids = inputs["input_ids"].squeeze(0)
        probabilities = F.softmax(logits, dim=-1)

        position_probabilities = []
        for i in range(len(input_ids)):
            position_probs = probabilities[i].cpu().numpy()
            position_probabilities.append({
                'position': i,
                'probs': position_probs
            })

        return position_probabilities

    def adjust_and_sample(self, p, q):
        """Adjust probability distribution p using q and sample from result."""
        adjusted_distribution = p - q
        adjusted_distribution = np.maximum(0, adjusted_distribution)
        adjusted_distribution /= np.sum(adjusted_distribution)
        sampled_token = np.random.choice(len(p), p=adjusted_distribution)
        return sampled_token

    def get_draft_tokens(self, draft_next_token_ids):
        """Convert token IDs to tokens."""
        return [self.draft_tokenizer.decode([token_id]) for token_id in draft_next_token_ids]

    def run_speculative_decoding(self, prompt, draft_next_token_ids, draft_tokens_probs, k=5):
        """Run the speculative decoding algorithm."""
        new_prompt = prompt + ''.join(self.get_draft_tokens(draft_next_token_ids))
        position_probabilities = self.get_target_all_token_probabilities(new_prompt)
        inputs = self.target_tokenizer(new_prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"].squeeze(0)

        accepted_tokens = []
        for pos, token_id in enumerate(draft_next_token_ids):
            # Get probabilities from both models
            q = round(draft_tokens_probs[pos][token_id].item(), 4)
            p_inx = len(input_ids) - k + pos - 1
            p = round(position_probabilities[p_inx]['probs'][token_id].item(), 4)
            token = self.target_tokenizer.decode([token_id])

            # Apply acceptance/rejection logic
            if p >= q:
                accepted_tokens.append(token)
            else:
                prob = p/q
                if random.random() <= prob:
                    accepted_tokens.append(token)
                else:
                    # Sample from adjusted distribution
                    ps = position_probabilities[p_inx]['probs']
                    qs = draft_tokens_probs[pos]
                    sampled_token_id = self.adjust_and_sample(ps, qs)
                    token = self.draft_tokenizer.decode([sampled_token_id])
                    accepted_tokens.append(token)
                    break

        return accepted_tokens 