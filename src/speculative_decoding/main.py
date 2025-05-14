from decoder import SpeculativeDecoder

def main():
    # Initialize the decoder
    decoder = SpeculativeDecoder()
    
    # Example prompt
    prompt = "Paris is the capital of"
    k = 5
    
    # Run draft model to get speculative tokens
    results = decoder.run_draft_model(prompt, k)
    draft_next_token_ids = results['draft_token_ids']
    draft_tokens_probs = results['probs']
    
    # Run speculative decoding
    accepted_tokens = decoder.run_speculative_decoding(
        prompt, 
        draft_next_token_ids, 
        draft_tokens_probs,
        k
    )
    
    # Print results
    print(f"Original prompt: {prompt}")
    print(f"Accepted tokens: {accepted_tokens}")
    print(f"Final output: {prompt + ''.join(accepted_tokens)}")

if __name__ == "__main__":
    main() 