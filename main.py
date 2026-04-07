from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
import yfinance as yf
from pydantic import BaseModel

app = FastAPI()

# ============================
# Azure OpenAI クライアント
# ============================
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# ============================
# 起動確認
# ============================
@app.get("/", response_class=HTMLResponse)
def index():
    return "<h2>stock-news-api 起動OK（ステップ3）</h2>"

# ============================
# ニュース本文抽出
# ============================
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

# ============================
# GPT：企業名抽出
# ============================
@app.post("/extract_company")
def extract_company(news: str):
    prompt = f"""
以下のニュース本文から、主要な企業名を1つだけ抽出してください。
余計な説明は不要で、企業名のみ返してください。

{news}
"""

    try:
        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

        company = res.choices[0].message.content.strip()
        return JSONResponse({"company": company})

    except Exception as e:
        return JSONResponse({"error": str(e)})

# ============================
# yf：企業情報取得
# ============================

class TickerRequest(BaseModel):
    ticker: str

@app.post("/stock_info")
def stock_info(req: TickerRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        info = t.info

        name = info.get("shortName") or info.get("longName") or ticker
        price = info.get("regularMarketPrice")
        sector = info.get("sector")
        currency = info.get("currency")

        return {
            "ticker": ticker,
            "name": name,
            "price": price,
            "sector": sector,
            "currency": currency,
        }

    except Exception as e:
        return {"error": str(e)}
