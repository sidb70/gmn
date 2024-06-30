// import React, { useEffect, useRef } from 'react';
// import CytoscapeGraph from './CytoscapeGraph';

// const App: React.FC = () => {
//   const cyRef = useRef<HTMLDivElement>(null);
//   const cyGraph = useRef<CytoscapeGraph | null>(null);

//   useEffect(() => {
//     if (cyRef.current) {
//       cyGraph.current = new CytoscapeGraph(cyRef.current);

//       cyGraph.current.onReady(() => {
//         // open test.json and load nodes and edges
//         fetch('test.json')
//           .then(response => response.json())
//           .then(data => {
//             console.log(data);
//             if (cyGraph.current) {
//               cyGraph.current.batchAdd(data.nodes, data.links);
//               // data.nodes.forEach((node: any) => {
                
//               //   if (cyGraph.current){
//               //     cyGraph.current.addNode(node.id);
//               //   }

//               // });

//               // data.links.forEach((edge: any) => {
//               //   if (cyGraph.current){
//               //     cyGraph.current.addEdge(edge.source, edge.target);
//               //   }
//               // });

//               cyGraph.current.runLayout();
//             }});
          
//       });
//     }}, []);
    
//       return (
//     <div className="App">
//       <div ref={cyRef} style={{ width: '100%', height: '600px' }}></div>
//     </div>
//   );
// };

// export default App;


/*---------------------------- PLOTLY GRAPH ----------------------------*/
// App.tsx
import React from 'react';
import PlotlyGraph from './components/PlotlyGraph';

function App() {
  return (
    <div className="App">
      <PlotlyGraph url='http://localhost:8000/plot/1'/>
    </div>
  );
}

export default App;
