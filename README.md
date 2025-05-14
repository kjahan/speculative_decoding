# Speculative Decoding

This project implements speculative decoding for language models, which aims to speed up generative model inference time. It uses a smaller draft model (OPT-125M) to generate speculative tokens and a larger target model (OPT-350M) to verify and accept/reject these tokens.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/speculative_decoding.git
cd speculative_decoding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the example script:
```bash
python src/speculative_decoding/main.py
```

## How it Works

The implementation follows the speculative decoding algorithm where:

1. A draft model (smaller) generates k speculative tokens
2. The target model (larger) evaluates these tokens
3. Tokens are accepted if:
   - Target model probability >= Draft model probability
   - Or randomly with probability p(x)/q(x) if target probability < draft probability
4. If a token is rejected, we sample from the adjusted distribution max(0, p(x)-q(x))

## Models Used

- Draft Model: facebook/opt-125m
- Target Model: facebook/opt-350m

Here our goal is to speed up generative model inference time. This has a few use cases such as fast edit suggestions for coding.

We use a small decoder (e.g. `opt-125m`) as the Draft model and a larger decoder (e.g. `opt-350m`) as the main model. We prompt the draft model to generate `k speculative tokens` along with their probabilities.

Next, we feed those `k tokens` along with the original prompt to the main model to get their likelihoods at once from the attention mask layer. Then we use the probabilities for speculative tokens from the draft model and main model to accept or reject the suggested tokens.

See this [video](https://www.youtube.com/watch?v=S-8yr_RibJ4) for more details.

The key insight is that there are many trivial tokens like "of" that a smaller model can easily predict. Therefore, we can use the smaller model to complete the prompt faster and then use the large model for verification. See below for some more references:

[A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

[Lecture 22: Hacker's Guide to Speculative Decoding in VLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)

[A Hacker's Guide to Speculative Decoding in vLLM](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)
