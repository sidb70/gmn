import { PlotlyHTMLElement } from 'plotly.js';

declare global {
  interface Window {
    Plotly: {
      newPlot: (
        divId: string | HTMLElement,
        data: any[],
        layout?: Partial<Plotly.Layout>,
        config?: Partial<Plotly.Config>
      ) => Promise<PlotlyHTMLElement>;
      react: (
        divId: string | HTMLElement,
        data: any[],
        layout?: Partial<Plotly.Layout>,
        config?: Partial<Plotly.Config>
      ) => Promise<PlotlyHTMLElement>;
    };
  }
}