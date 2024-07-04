# [antilect](http://www.antilect.com)

Reproducing, breaking, and iterating [Graph Metanetworks for Proessing Diverse Neural Architectures](https://arxiv.org/pdf/2312.04501)

## what are we doing
creating graph representations of neural networks, called parameter graphs. we then train a graph neural network to make predictions about neural architectures and their weight spaces based on their parameter graph.

## why are we doing it
we would like to scale graph metanetworks to larger and deeper models to perform tasks like assisting in architecture search. we would also like to apply similar metanetworks to make predictions not only over weight spaces, but also in activation space. we would like to apply metanets to [dictionary learning](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) and compare its performance against [sparse autoencoders](https://github.com/AntonP999/Sparse_autoencoder) for the same task

