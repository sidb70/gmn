import React, { useEffect, useRef } from 'react';
import CytoscapeGraph from './CytoscapeGraph';

const App: React.FC = () => {
  const cyRef = useRef<HTMLDivElement>(null);
  const cyGraph = useRef<CytoscapeGraph | null>(null);

  useEffect(() => {
    if (cyRef.current) {
      cyGraph.current = new CytoscapeGraph(cyRef.current);

      cyGraph.current.onReady(() => {
        

        // open test.json and load nodes and edges
        fetch('test.json')
          .then(response => response.json())
          .then(data => {
            console.log(data);
            if (cyGraph.current) {
              data.nodes.forEach((node: any) => {
                if (cyGraph.current){
                  cyGraph.current.addNode(node.id);
                }

              });

              data.links.forEach((edge: any) => {
                if (cyGraph.current){
                  cyGraph.current.addEdge(edge.source, edge.target);
                }
              });

              cyGraph.current.runLayout();
            }});
              

        // Add nodes
        // cyGraph.current.addNode('A');
        // cyGraph.current.addNode('B');
        // cyGraph.current.addNode('C');

        // // Add edges
        // cyGraph.current.addEdge('A', 'B');
        // cyGraph.current.addEdge('B', 'C');
        // cyGraph.current.addEdge('C', 'A');
        // cyGraph.current.addEdge('A', 'B'); // Adding a multi-edge

        // // Customize colors
        // cyGraph.current.setNodeColor('A', '#FF0000');
        // cyGraph.current.setEdgeColor('A', 'B', '#FF0000');

        // // Run layout
        // cyGraph.current.runLayout();
            
          
          
      });
    }}, []);
    
      return (
    <div className="App">
      <div ref={cyRef} style={{ width: '100%', height: '600px' }}></div>
    </div>
  );
};

export default App;
