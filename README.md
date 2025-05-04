# Speculative Decoding
Speculative Sampling

Here our goal is to speed up generative model inference time. This has a few use cases such as fast edit suggestions for coding.

We use a small decoder (e.g. `opt-125m`) as the Draft model and a larger decoder (e.g. `opt-350m`) as the main model. We prompt the draft model to generate `k speculative tokens` along with their probabilities.

Next, we feed those `k tokens` along with the original prompt to the main model to get their likelihoods at once from the attention mask layer. Then we use the probabilities for speculative tokens from the draft model and main model to accept or reject the suggested tokens.

See this [video](https://www.youtube.com/watch?v=S-8yr_RibJ4)) for more details.

The key insight is that there are many trivial tokens like "of" that a smaller model can easily predict. Therefore, we can use the smaller model to complete the prompt faster and then use the large model for verification. See below for some more references:

[A Hitchhiker’s Guide to Speculative Decoding](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/)

[Lecture 22: Hacker's Guide to Speculative Decoding in VLLM](https://www.youtube.com/watch?v=9wNAgpX6z_4)

[A Hacker’s Guide to Speculative Decoding in vLLM](https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p)
