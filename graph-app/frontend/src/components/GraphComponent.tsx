import React, { useRef, useEffect, useState } from 'react';
import ForceGraph3D from '3d-force-graph';

interface NodeFeatures{
    layer_num: number;
    rel_index: number;
    node_type: number;
}
interface EdgeFeatures{
    weight: number;
    layer_num: number;
    edge_type: number;
    pos_encoding_x: number;
    pos_encoding_y: number;
    pos_encoding_depth: number;
}
interface Node {
    id: number;
    name?: string;
    features: NodeFeatures;
  }
  
interface Edge {
    id: string; // Add this line
    source: number;
    target: number;
    features: EdgeFeatures;
}

interface GraphData {
    nodes: Node[];
    links: Edge[];
}


const GraphComponent: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const graphRef = useRef<any>(null);

  useEffect(() => {
    fetch('test.json')
      .then(response => response.json())
      .then((data: any) => {
        console.log('Fetched data:', data);
        if (Array.isArray(data.nodes) && Array.isArray(data.links)) {
          setGraphData(data as GraphData);
        } else {
          console.error('Invalid data structure:', data);
          setGraphData({ nodes: [], links: [] });
        }
      })
      .catch(error => {
        console.error('Error loading graph data:', error);
        setGraphData({ nodes: [], links: [] });
      });
  }, []);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize the graph
    graphRef.current = ForceGraph3D()(containerRef.current)
      .nodeLabel('id')
      .backgroundColor('#101020')
      .forceEngine('d3')
      .nodeColor(() => '#ffffff')
      .linkColor(() => '#ffffff');

    const handleResize = () => {
      graphRef.current.width(window.innerWidth);
      graphRef.current.height(window.innerHeight);
    };

    window.addEventListener('resize', handleResize);
    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
      graphRef.current._destructor();
    };
  }, []);

  useEffect(() => {
    if (graphRef.current && graphData) {
      console.log("Updating graph data:", graphData);
      graphRef.current.graphData(graphData);
    }
  }, [graphData]);

  if (!graphData) return <div>Loading...</div>;

  return <div ref={containerRef} style={{ width: '100vw', height: '100vh' }} />;
};

export default GraphComponent;