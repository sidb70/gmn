import React from 'react';
import PlotlyGraph from './components/PlotlyGraph';
import './App.css';
//import 'dotenv/config';



function App() {
  const SERVER_IP = '127.0.0.1'

  return (

    <div className="App">
      <div className="container">
        <h1>acumen</h1>
        <h2>understanding the internal representations of LLMs</h2>
        <h3>
          <a href="https://www.linkedin.com/in/sid-bhat/" className="author-link">Siddhartha Bhattacharya,</a>
          <a href="https://www.linkedin.com/in/uzair-m/" className="author-link">Uzair Mohammed,</a> 
          <a href="https://www.linkedin.com/in/daniel-helo-puccini-a3779b242/" className="author-link">Daniel Helo,</a> 
        </h3>
        
        <p>The graphs shown below are neural networks represented as parameter graphs from 
          <a href="https://arxiv.org/pdf/2312.04501" className='inline-link'>Graph Metanetworks for Processing Diverse Neural Architectures</a>.
          These graphs are able to efficiently create equivarient representations of neural networks, which can be used to compare and analyze different neural network architectures.
          So far, we have implemented the parameter graph representations for linear, convolutional, normalization, and nonlinearity layers. 
        </p>
        <p>
          We can generate arbitrary neural network architectures and represent them as parameter graphs. We can use the parameter graphs of these neural networks
          as training data for a graph neural network to perform a variety of tasks such as architecture search, architecture comparison, and architecture analysis. 
          Our goal for this project is to train a <a href="https://www.anthropic.com/research/mapping-mind-language-model">dictionary learner for large language models</a> (LLMs)
          in order to interpret the learned representations of concepts within the LLM. With such a mapping, we can better understand the inner workings of LLMs and fine tune them for 
          specific tasks in a single shot.
        </p>
        <p>
          Before we get there, however, we need to create even more efficient parameter graph representations, as the current representations will be difficult to scale to the size of modern LLMs (with billions of parameters).
          We also need to create a training pipeline for the graph neural network that can handle the large size of the parameter graphs.
        </p>

        <p>
          But this is just the beginning. We're in the lab, working hard to make this a reality.
        </p>
        <PlotlyGraph url={ `http://${SERVER_IP}:8000/plot/1` } title="Linear Only Neural Network" />
        <PlotlyGraph url={`http://${SERVER_IP}:8000/plot/2`} title="Convolutional Neural Network" />
      </div>
    </div>
  );
}

export default App;