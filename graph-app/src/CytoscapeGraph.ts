import cytoscape, { Core, NodeSingular, EdgeSingular } from 'cytoscape';

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

  runLayout(): void {
    this.cy.layout({ name: 'cose' }).run();
  }
}

export default CytoscapeGraph;
