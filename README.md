# AIDocumentAnalyser
Ways of using LLMs to Analyse different kinds of documents - Tabular, Text, Images, etc

## Jarvis Lite web app

This repo now includes a very simple chat web app called Jarvis Lite.

How to run locally:
- Install dependencies: `pip install -r requirements.txt` (or `python3 -m pip install -r requirements.txt`)
- Start the server: `python app.py` (or `python3 app.py`)
- Open http://localhost:8000 in your browser

The backend exposes a single streaming endpoint at `/stream` that returns a demo response as Server-Sent Events.
