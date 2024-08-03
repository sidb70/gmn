// PlotlyGraph.jsx
import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

function PlotlyGraph({ url, title }) {
  const [plotHTML, setPlotHTML] = useState('');

  useEffect(() => {
    if (url) {
      console.log('Fetching plot:', url)
      fetch(url)
        .then(response => response.text())
        .then(data => {
          console.log('Plot fetched')
          setTimeout(() => executeScripts(data), 100);
          setPlotHTML(data);
          executeScripts(data);
        })
        .catch(error => console.error('Error fetching plot:', error));
    }
  }, [url]);

  const executeScripts = (html) => {
    const scriptRegex = /<script\b[^>]*>([\s\S]*?)<\/script>/gm;
    console.log('Executing scripts:');
    let scripts;
    while ((scripts = scriptRegex.exec(html))) {
      const scriptContent = scripts[1];
      console.log('found script')
      const script = document.createElement('script');
      script.text = scriptContent;
      document.body.appendChild(script).parentNode.removeChild(script);
    }
  };

  return (
    <div className="PlotlyGraph">
      <h2>{title}</h2>
      <div style={{minHeight: '400px'}} dangerouslySetInnerHTML={{ __html: plotHTML }} />
    </div>
  );
}

PlotlyGraph.propTypes = {
  url: PropTypes.string.isRequired,
  title: PropTypes.string,
};

export default PlotlyGraph;