# TCN with attention

Temporal Convolutional Network with attention layer

Concept of model is mostly like [Simple Neural Attentive Meta-Learner](https://github.com/sagelywizard/snail).
But in this model, attention layer is on every top of convolutions layers. And attention size is differ from SNAIL

## Results

Dataset: Agnews
- with attention: 0.82
- without attention: 0.81

### My thoughts on results

Most of simple models on agnews shows 0.81 accuracy. (Which tested on [A Structured Self-Attentive Sentence Embedding](https://github.com/flrngel/Self-Attentive-tensorflow), [TagSpace](https://github.com/flrngel/TagSpace-tensorflow) and it uses word based embedding)

So 0.82 accuracy with **character based model** seems worthiness.
