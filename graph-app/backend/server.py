from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/plot/{graph_id}", response_class=HTMLResponse)
def get_plot(graph_id: int):
    if not os.path.exists(f'./backend/static/graph{graph_id}.html'):
        return 'Graph not found'
    with open(f'./backend/static/graph{graph_id}.html', 'r') as file:
        html_content = file.read()
    return html_content


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)