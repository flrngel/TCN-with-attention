# TCN with attention

Temporal Convolutional Network with attention layer

Concept of model is mostly like [Simple Neural Attentive Meta-Learner](https://github.com/sagelywizard/snail).
But in this model, attention layer is on every top of convolutions layers.

## Results

Dataset: Agnews
- with attention: 0.82
- without attention: 0.81

### My thoughts on results

Most of convolutional products on agnews shows 0.81 accuracy. (See my [A Structured Self-Attentive Sentence Embedding](https://github.com/flrngel/Self-Attentive-tensorflow), [TagSpace](https://github.com/flrngel/TagSpace-tensorflow)) 0.82 acc seems worthiness.
