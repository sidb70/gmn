import React from 'react';
import PlotlyGraph from './components/PlotlyGraph';
import './App.css';
//import 'dotenv/config';



function App() {
  const SERVER_IP = '164.92.74.215'

  return (

    <div className="App">
      <div className="container">
        <h1>antilect</h1>
        <h2>understanding the weight spaces of neural networks</h2>
        <h3>
          <a target="_blank" rel="noopener noreferrer" href="https://forms.gle/Zsr67eJ6RzrgyKMD9">Research Updates Email List</a>
        </h3>
        <p>The graphs shown below are neural networks represented as parameter graphs from 
        <a href="https://arxiv.org/pdf/2312.04501" className='inline-link'>Graph Metanetworks for Processing Diverse Neural Architectures</a>.
          These graphs efficiently create permutation equivariant representations of neural networks. By feeding these graph representations into a graph neural network (GNN), we can compare and analyze different network architectures.
        </p>
        <p>
            Currently, we are reproducing the paper linked above and have implemented parameter graph representations for linear, convolutional, normalization, and nonlinearity layers. We are in the process of creating a dataset of parameter graphs for various neural network architectures and tasks. These parameter graphs can serve as training data for a message-passing GNN to perform tasks such as architecture search, comparison, and analysis.
        </p>
        <p>
            We are developing a graph preprocessing and training pipeline for a message-passing GNN that can handle parameter graphs as input and predict their accuracy for given tasks. Our long-term vision is to create metanetworks that generalize to larger and deeper models.
        </p>
        <p>
          To reach this goal, we need to create even more efficient parameter graph representations, as the current representations will be difficult to scale to the size of modern LLMs (with billions of parameters).
        </p>

        <p>
          But this is just the beginning. We're in the lab, working hard to make this a reality.
        </p>
        <p>
          <a href="https://www.linkedin.com/in/sid-bhat/" className="author-link">Siddhartha Bhattacharya,</a>
          <a href="https://www.linkedin.com/in/uzair-m/" className="author-link">Uzair Mohammed,</a> 
          <a href="https://www.linkedin.com/in/daniel-helo-puccini-a3779b242/" className="author-link">Daniel Helo,</a> 
        </p>
        <PlotlyGraph url={ `http://${SERVER_IP}:8000/plot/1` } title="Multi-Layer Perceptron" />
        <PlotlyGraph url={`http://${SERVER_IP}:8000/plot/2`} title="Convolutional Neural Network" />
      </div>
    </div>
  );
}

export default App;