# Speculative Decoding
Speculative Sampling

Here our goal is to speed up generative model inference time. This has many use cases for edit suggestion for writing or coding.

We will use a draft/basic decoder like GPT-2 and a bigger size model as the core model. We will prompt the draft model and generate k speculative tokens along with their probabilities.

Next we feed those k tokens along with the original prompt to the main model to get their liklihhods at once from the attention mask layer. Then we use the probabilities for speculative tokens from the draft model and main model to accept or reject speculated tokens.

See this video for more explanations:

https://www.youtube.com/watch?v=S-8yr_RibJ4

The key insight is that there are many simple tokens like "of" that even smaller model can easily predict them so we can use the smaller model to generate them faster and then use the bigger size model for facts and harder tokens!

https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/

https://www.youtube.com/watch?v=9wNAgpX6z_4

https://docs.google.com/presentation/d/1p1xE-EbSAnXpTSiSI0gmy_wdwxN5XaULO3AnCWWoRe4/edit#slide=id.p