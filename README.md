# [antilect](http://www.antilect.com)

Reproducing, breaking, and iterating [Graph Metanetworks for Proessing Diverse Neural Architectures](https://arxiv.org/pdf/2312.04501)

## What are we doing
Creating graph representations of neural networks, called parameter graphs. We then train a graph neural network to make predictions about neural architectures and their weight spaces based on their parameter graph.

## Why are we doing it
- To scale graph metanetworks to larger and deeper models, facilitating tasks such as architecture search.
- To apply similar metanetworks to make predictions over activation spaces, not just weight spaces.
- To explore the application of metanets to [dictionary learning](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) and compare their performance against [sparse autoencoders](https://github.com/AntonP999/Sparse_autoencoder).

## Where are we at
Currently, we can successfully represent linear, convolutional, normalization, and activation layers in parameter graphs. We are in the process of generating the training data and developing the graph neural network to perform meta predictions.

## Try it out!
Simply clone the open graph_playground.ipynb to generate and visiualize parameter graphs from a neural network. 
## Interested? Join our [research updates list](https://forms.gle/mbPynMm5EMcZ5ZxWA)!
Stay updated on our progress and get a behind-the-scenes look at our ambitious projects.
