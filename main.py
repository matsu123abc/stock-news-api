from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from bs4 import BeautifulSoup

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return "<h2>stock-news-api 起動OK</h2>"

@app.post("/extract_news")
def extract_news(url: str):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.select("p")
        text = "\n".join([p.get_text().strip() for p in paragraphs])
        text = "\n".join([line for line in text.split("\n") if line.strip()])

        return JSONResponse({"text": text})

    except Exception as e:
        return JSONResponse({"error": str(e)})
