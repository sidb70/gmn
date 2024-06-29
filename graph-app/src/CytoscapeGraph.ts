import cytoscape, { Core, NodeSingular, EdgeSingular } from 'cytoscape';


// Define the type for nodes and edges to ensure the 'group' property matches the expected type
type NodeElement = { group: 'nodes'; data: { id: string; }; };
type EdgeElement = { group: 'edges'; data: { source: string; target: string; }; };


// This array now correctly matches the expected type 'ElementDefinition[]'

class CytoscapeGraph {
  private cy: Core;
  private isReady: boolean = false;
  private readyCallback: (() => void) | null = null;

  constructor(container: HTMLElement) {
    this.cy = cytoscape({
      container: container,
      elements: [],
      style: [
        {
          selector: 'node',
          style: {
            'background-color': '#FFD700',
            'label': 'data(id)',
            'width': 10,
            'height': 10,
          },
        },
        {
          selector: 'edge',
          style: {
            'width': 1,
            'line-color': '#ccc',
            'opacity': 0.5,
            'curve-style': 'bezier',
          },
        },
      ],
      layout: {
        name: 'cose',
        randomize: true,
        nodeRepulsion: 10000 as any,
        idealEdgeLength: 100 as any,
        nodeOverlap: 20 as any,
        gravity: 80 as any,
        numIter: 1000 as any,
        initialTemp: 200    as any,
        coolingFactor: 0.95 as any,
        minTemp: 1.0      as any,
      },
    });

    this.cy.ready(() => {
      this.isReady = true;
      if (this.readyCallback) {
        this.readyCallback();
      }
    });
  }

  onReady(callback: () => void): void {
    if (this.isReady) {
      callback();
    } else {
      this.readyCallback = callback;
    }
  }

  addNode(id: string): NodeSingular {
    return this.cy.add({
      group: 'nodes',
      data: { id },
    });
  }

  addEdge(source: string, target: string): EdgeSingular {
    return this.cy.add({
      group: 'edges',
      data: { source, target },
    });
  }

  setNodeColor(id: string, color: string): void {
    const node = this.cy.getElementById(id);
    if (node.isNode()) {
      node.style('background-color', color);
    }
  }

  setEdgeColor(sourceId: string, targetId: string, color: string): void {
    const edge = this.cy.edges(`[source = "${sourceId}"][target = "${targetId}"]`);
    edge.style('line-color', color);
  }

  batchAdd(nodes: {id: string}[], edges: {source: string, target: string}[]): void {
    const elementsToAdd: (NodeElement | EdgeElement)[] = [
      ...nodes.map(node => ({ group: 'nodes', data:  {id: node.id} }) as NodeElement),
      ...edges.map(edge => ({ group: 'edges', data: { source: edge.source, target: edge.target } }) as EdgeElement),
    ];
    this.cy.batch(() => {
      this.cy.add(elementsToAdd);
    });
  }

  runLayout(): void {
    this.cy.layout({ name: 'cose' }).run();
  }
}

export default CytoscapeGraph;
