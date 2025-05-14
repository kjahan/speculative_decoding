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

The key insight is that there are many simple tokens like "of" that a small LLM can predict. Therefore, we can use the small model to complete the prompt faster and use the large model for verification. See below for some some references:

[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)

[Speculative Decoding: When Two LLMs are Faster than One](https://www.youtube.com/watch?v=S-8yr_RibJ4)

[A Hitchhiker's Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

[Lecture 22: Hacker's Guide to Speculative Decoding in VLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)

[A Hacker's Guide to Speculative Decoding in vLLM](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)
