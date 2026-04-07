from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import requests
from bs4 import BeautifulSoup
from openai import AzureOpenAI
import os
import yfinance as yf
from pydantic import BaseModel
import json
import re

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

class TrendRequest(BaseModel):
    ticker: str

@app.post("/stock_trend")
def stock_trend(req: TrendRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="3mo")

        if hist.empty:
            return {"error": "株価データが取得できませんでした。"}

        price_now = float(hist["Close"].iloc[-1])
        price_3m_ago = float(hist["Close"].iloc[0])
        ret_3m = (price_now / price_3m_ago - 1) * 100

        # 1ヶ月分（21営業日）
        hist_1m = hist.iloc[-21:] if len(hist) >= 21 else hist
        price_1m_ago = float(hist_1m["Close"].iloc[0])
        ret_1m = (price_now / price_1m_ago - 1) * 100

        return {
            "ticker": ticker,
            "price_now": price_now,
            "ret_1m": ret_1m,
            "ret_3m": ret_3m
        }

    except Exception as e:
        return {"error": str(e)}

class SummaryRequest(BaseModel):
    ticker: str

@app.post("/company_summary")
def company_summary(req: SummaryRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        info = t.info

        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "不明")
        summary = info.get("longBusinessSummary", "")

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "summary": summary
        }

    except Exception as e:
        return {"error": str(e)}

class SummaryJaRequest(BaseModel):
    ticker: str

@app.post("/company_summary_ja")
def company_summary_ja(req: SummaryJaRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        info = t.info

        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "不明")
        summary_en = info.get("longBusinessSummary", "")

        if not summary_en:
            return {"error": "企業概要データがありません。"}

        # GPT に日本語要約させる
        prompt = f"""
以下は企業の英語説明です。これを日本語で5〜7行に要約してください。

【企業説明（英語）】
{summary_en}
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        summary_ja = res.choices[0].message.content.strip()

        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "summary_ja": summary_ja
        }

    except Exception as e:
        return {"error": str(e)}

class FinancialRequest(BaseModel):
    ticker: str

@app.post("/financials")
def financials(req: FinancialRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        fin = t.financials

        if fin is None or fin.empty:
            return {"error": "業績データが取得できませんでした。"}

        # 最新4期分だけ抽出
        fin = fin.T[["Total Revenue", "Net Income", "Diluted EPS"]].tail(4)
        fin.columns = ["売上高", "純利益", "EPS"]

        # JSON 形式で返す
        data = fin.reset_index().rename(columns={"index": "期"})  
        records = data.to_dict(orient="records")

        return {
            "ticker": ticker,
            "financials": records
        }

    except Exception as e:
        return {"error": str(e)}

class FinancialSummaryRequest(BaseModel):
    ticker: str

@app.post("/financials_summary")
def financials_summary(req: FinancialSummaryRequest):
    ticker = req.ticker.strip()
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    try:
        t = yf.Ticker(ticker)
        fin = t.financials

        if fin is None or fin.empty:
            return {"error": "業績データが取得できませんでした。"}

        # 最新4期分だけ抽出
        fin = fin.T[["Total Revenue", "Net Income", "Diluted EPS"]].tail(4)
        fin.columns = ["売上高", "純利益", "EPS"]

        fin_text = fin.to_string()

        # GPT に要約させる
        prompt = f"""
以下の業績データをもとに、企業の財務状況を5〜7行で日本語で要約してください。

【業績データ】
{fin_text}

【出力形式】
- 売上の傾向
- 利益の傾向
- EPSの傾向
- 財務の健全性
- 投資家が注目すべきポイント
"""

        res = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        summary = res.choices[0].message.content.strip()

        return {
            "ticker": ticker,
            "financials_summary": summary
        }

    except Exception as e:
        return {"error": str(e)}

class NewsAnalyzeRequest(BaseModel):
    news: str
    ticker: str

@app.post("/analyze_news_simple")
def analyze_news_simple(req: NewsAnalyzeRequest):
    news = req.news.strip()
    ticker = req.ticker.strip()

    if not news:
        return {"error": "ニュース本文が空です。"}
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    # --- 企業概要（日本語） ---
    try:
        t = yf.Ticker(ticker)
        info = t.info
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector", "不明")
        summary_en = info.get("longBusinessSummary", "")
    except Exception:
        name = ticker
        sector = "不明"
        summary_en = ""

    # GPTで日本語要約
    if summary_en:
        prompt_summary = f"""
以下の企業説明を日本語で5〜7行に要約してください。

【企業説明】
{summary_en}
"""
        res_sum = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt_summary}],
            temperature=0.2,
        )
        summary_ja = res_sum.choices[0].message.content.strip()
    else:
        summary_ja = "企業概要データなし"

    # --- 株価トレンド ---
    try:
        hist = t.history(period="3mo")
        price_now = float(hist["Close"].iloc[-1])
        price_3m_ago = float(hist["Close"].iloc[0])
        ret_3m = (price_now / price_3m_ago - 1) * 100

        hist_1m = hist.iloc[-21:] if len(hist) >= 21 else hist
        price_1m_ago = float(hist_1m["Close"].iloc[0])
        ret_1m = (price_now / price_1m_ago - 1) * 100
    except Exception:
        price_now = None
        ret_1m = None
        ret_3m = None

    # --- GPT ニュース分析（軽量版） ---
    prompt = f"""
あなたはプロの株式アナリストです。
以下の情報をもとに、このニュースが企業にとってどれほど重要かを分析してください。

【企業名】{name}
【セクター】{sector}

【企業概要（要約）】
{summary_ja}

【株価情報】
現在株価: {price_now}
1ヶ月リターン: {ret_1m}
3ヶ月リターン: {ret_3m}

【ニュース本文】
{news}

【出力形式】
以下の JSON だけを返してください：

{{
  "sentiment_score": "数値（-1.0〜+1.0）",
  "impact": "強い上昇 / 上昇 / 中立 / 下落 / 強い下落",
  "reason": "ニュースが企業に与える影響の理由（200文字以内）"
}}
"""

    res = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw = res.choices[0].message.content.strip()

    # --- JSON抽出（完全安定版） ---
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("JSON が見つかりません")

        js = raw[start:end+1]

        # "+0.7" → "0.7"
        js = js.replace("+", "")

        data = json.loads(js)

    except Exception as e:
        return {"error": "JSON解析エラー", "raw": raw, "detail": str(e)}

    # --- ここから先は JSON が正しい前提で処理 ---
    return {
        "ticker": ticker,
        "company": name,
        "analysis": data
    }

class SimilarNewsRequest(BaseModel):
    news: str
    ticker: str


@app.post("/analyze_similar_news")
def analyze_similar_news(req: SimilarNewsRequest):
    news_text = req.news.strip()
    ticker = req.ticker.strip()

    if not news_text:
        return {"error": "ニュース本文が空です。"}
    if not ticker:
        return {"error": "ティッカーコードを入力してください。"}

    # --- 企業名取得 ---
    try:
        t = yf.Ticker(ticker)
        company_name = t.info.get("shortName", ticker)
    except Exception:
        company_name = ticker

    # --- 1. GPT キーワード抽出 ---
    prompt_kw = f"""
以下のニュース本文から、Web検索に使える重要キーワードを5〜10個抽出してください。
・名詞のみ
・1行に1つ
・余計な説明は書かない

【ニュース本文】
{news_text}
"""

    res_kw = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_kw}],
        temperature=0.2,
    )

    lines = [l.strip() for l in res_kw.choices[0].message.content.split("\n") if l.strip()]
    keywords = [re.sub(r"^[\-・\d\.]+\s*", "", l) for l in lines]

    if not keywords:
        return {"error": "キーワード抽出に失敗しました。"}

    query = " ".join(keywords)

    # --- 2. FastAPI /tools/news に問い合わせ ---
    try:
        url = "http://localhost:8000/tools/news"
        r = requests.get(url, params={"keyword": query}, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as e:
        return {"error": "類似ニュース検索エラー", "detail": str(e)}

    if not articles:
        return {"html": "<p>類似ニュースが見つかりませんでした。</p>"}

    # --- 3. 類似ニュース一覧 HTML テーブル ---
    rows = []
    for a in articles:
        title = a.get("title", "")
        link = a.get("link", "")
        source = a.get("source", "")
        date = a.get("date", "")
        snippet = a.get("snippet", "")

        rows.append(
            f"<tr>"
            f"<td><a href='{link}' target='_blank'>{title}</a></td>"
            f"<td>{source}</td>"
            f"<td>{date}</td>"
            f"<td>{snippet}</td>"
            f"</tr>"
        )

    list_html = """
<h3>類似ニュース一覧</h3>
<table border="1" style="border-collapse: collapse; width:100%;">
<tr><th>タイトル</th><th>メディア</th><th>日付</th><th>概要</th></tr>
""" + "\n".join(rows) + "</table>"

    # --- 4. GPT によるまとめ分析 ---
    news_block = ""
    for i, a in enumerate(articles, 1):
        news_block += f"{i}. {a.get('title','')}\n{a.get('snippet','')}\n\n"

    prompt_summary = f"""
あなたはプロの株式アナリストです。
以下の「類似ニュース一覧」をもとに、このテーマが投資家にとってどのような意味を持つかを整理してください。

【対象銘柄】
ティッカー: {ticker}
企業名: {company_name}

【類似ニュース一覧】
{news_block}

【タスク】
1. 類似ニュースに共通する「業界テーマ」を1〜3個に整理
2. このテーマが、上記銘柄にとってどのような中期的な意味を持つかを説明
3. セクター全体のトレンド（政策・規制・技術・需要）を整理
4. 投資家が今後フォローすべきポイントを3〜5個、箇条書きで提示

【出力形式】
見出し＋箇条書き中心で、日本語で簡潔に。
"""

    res_sum = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_summary}],
        temperature=0.2,
    )

    summary_html = "<h3>類似ニュースまとめ分析</h3><p>" + res_sum.choices[0].message.content.replace("\n", "<br>") + "</p>"

    # --- 最終 HTML を返す ---
    return {
        "html": list_html + summary_html
    }

