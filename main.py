from fastapi import FastAPI
import requests
import os
import json
import yfinance as yf
from pydantic import BaseModel
from openai import AzureOpenAI
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup

# ============================
# FastAPI 初期化
# ============================
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
# 1. SERP API ニュース検索
# ============================
class NewsSearchResponse(BaseModel):
    articles: list

@app.get("/tools/news")
def search_news(keyword: str):
    try:
        url = "https://serpapi.com/search"
        params = {
            "engine": "google_news",
            "q": keyword,
            "api_key": os.getenv("SERPAPI_KEY")
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        articles = []
        for item in data.get("news_results", []):
            articles.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "source": item.get("source"),
                "date": item.get("date"),
                "snippet": item.get("snippet")
            })

        return {"articles": articles}

    except Exception as e:
        return {"error": str(e)}

@app.post("/extract_news")
def extract_news(url: str):
    """
    ニュースURLから本文を抽出するAPI
    """
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        # <p>タグをすべて抽出
        paragraphs = soup.select("p")
        text = "\n".join([p.get_text().strip() for p in paragraphs])

        # 空行を除去
        text = "\n".join([line for line in text.split("\n") if line.strip()])

        return JSONResponse({"text": text})

    except Exception as e:
        return JSONResponse({"error": f"本文抽出エラー: {str(e)}"})

# ============================
# 2. 類似ニュース分析 API
# ============================
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

    # --- GPT：検索クエリ生成 ---
    prompt_kw = f"""
以下のニュース本文から、Google News 検索でヒットしやすい検索クエリを1つ作成してください。

【条件】
・文章形式（名詞の羅列は禁止）
・20〜40文字程度
・固有名詞を含める
・検索意図が明確な自然な文章
・出力はクエリ1行のみ

【ニュース本文】
{news_text}
"""

    res_kw = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_kw}],
        temperature=0.2,
    )
    query = res_kw.choices[0].message.content.strip()

    # --- SERP API 呼び出し ---
    try:
        url = "http://localhost:8000/tools/news"
        r = requests.get(url, params={"keyword": query}, timeout=10)
        r.raise_for_status()
        articles = r.json().get("articles", [])
    except Exception as e:
        return {"error": "類似ニュース検索エラー", "detail": str(e)}

    if not articles:
        return {"html": "<p>類似ニュースが見つかりませんでした。</p>"}

    # --- 類似ニュース一覧 HTML ---
    rows = []
    for a in articles:
        rows.append(
            f"<tr>"
            f"<td><a href='{a.get('link','')}' target='_blank'>{a.get('title','')}</a></td>"
            f"<td>{a.get('source','')}</td>"
            f"<td>{a.get('date','')}</td>"
            f"<td>{a.get('snippet','')}</td>"
            f"</tr>"
        )

    list_html = """
<h3>類似ニュース一覧</h3>
<table border="1" style="border-collapse: collapse; width:100%;">
<tr><th>タイトル</th><th>メディア</th><th>日付</th><th>概要</th></tr>
""" + "\n".join(rows) + "</table>"

    # --- GPT：まとめ分析 ---
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
3. セクター全体のトレンドを整理
4. 投資家が今後フォローすべきポイントを3〜5個提示

【出力形式】
見出し＋箇条書き中心で、日本語で簡潔に。
"""

    res_sum = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_summary}],
        temperature=0.2,
    )

    summary_html = "<h3>類似ニュースまとめ分析</h3><p>" + res_sum.choices[0].message.content.replace("\n", "<br>") + "</p>"

    return {"html": list_html + summary_html}

# ============================
# 3. 総合ニュース分析 API（メイン）
# ============================
class NewsWithTickerRequest(BaseModel):
    news: str
    ticker: str

@app.post("/analyze_news_with_ticker")
def analyze_news_with_ticker(req: NewsWithTickerRequest):
    news_text = req.news.strip()
    ticker = req.ticker.strip()

    if not news_text:
        return {"error": "ニュース本文が空です。"}
    if not ticker:
        return {"error": "ティッカーを入力してください。"}

    # --- 企業情報 ---
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

    # --- 企業概要（日本語要約） ---
    if summary_en:
        prompt_sum = f"""
以下の企業説明を日本語で5〜7行に要約してください。

【企業説明】
{summary_en}
"""
        res_sum = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt_sum}],
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

    # --- 業績データ ---
    try:
        fin = t.financials
        revenue = fin.loc["Total Revenue"].iloc[0]
        net_income = fin.loc["Net Income"].iloc[0]
        eps = fin.loc["Diluted EPS"].iloc[0]
    except Exception:
        revenue = None
        net_income = None
        eps = None

    # --- 業績要約 ---
    prompt_fin = f"""
以下の業績データをもとに、企業の財務状況を日本語で4〜6行に要約してください。

売上高: {revenue}
純利益: {net_income}
EPS: {eps}
"""
    res_fin = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_fin}],
        temperature=0.2,
    )
    fin_summary = res_fin.choices[0].message.content.strip()

    # --- ニュース分析 ---
    prompt_news = f"""
あなたはプロの株式アナリストです。
以下のニュースが企業に与える影響を分析し、JSON で返してください。

【企業名】{name}
【セクター】{sector}

【企業概要】
{summary_ja}

【株価情報】
現在株価: {price_now}
1ヶ月リターン: {ret_1m}
3ヶ月リターン: {ret_3m}

【ニュース本文】
{news_text}

【出力形式】
{{
  "sentiment_score": "数値（-1.0〜+1.0）",
  "impact": "強い上昇 / 上昇 / 中立 / 下落 / 強い下落",
  "reason": "ニュースが企業に与える影響の理由（200文字以内）"
}}
"""
    res_news = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt_news}],
        temperature=0.2,
    )
    raw = res_news.choices[0].message.content.strip()

    start = raw.find("{")
    end = raw.rfind("}")
    js = raw[start:end+1].replace("+", "")
    analysis = json.loads(js)

    # --- 類似ニュース ---
    try:
        url = "http://localhost:8000/analyze_similar_news"
        r = requests.post(url, json={"news": news_text, "ticker": ticker}, timeout=20)
        similar_html = r.json().get("html", "")
    except:
        similar_html = "<p>類似ニュース取得エラー</p>"

    # --- HTML 結合 ---
    summary_html = summary_ja.replace("\n", "<br>")
    fin_html = fin_summary.replace("\n", "<br>")
    chart_url = f"https://stocks.finance.yahoo.co.jp/stocks/chart/?code={ticker}"

    html = f"""
<h2>{name}（{ticker}）ニュース総合分析</h2>

<h3>企業概要</h3>
<p>{summary_html}</p>

<h3>株価トレンド</h3>
<p>現在株価: {price_now}<br>
1ヶ月リターン: {ret_1m}%<br>
3ヶ月リターン: {ret_3m}%</p>

<h3>業績サマリー</h3>
<p>{fin_html}</p>

<h3>ニュース分析</h3>
<p><b>スコア:</b> {analysis["sentiment_score"]}<br>
<b>影響度:</b> {analysis["impact"]}<br>
<b>理由:</b> {analysis["reason"]}</p>

{similar_html}

<h3>チャート</h3>
<p><a href="{chart_url}" target="_blank">Yahoo! JAPAN チャートを見る</a></p>
"""

    return {"html": html}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>ニュース総合分析ツール</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { font-family: sans-serif; padding: 20px; }
input, textarea { width: 100%; padding: 10px; margin-top: 8px; font-size: 16px; }
button { width: 100%; padding: 12px; margin-top: 15px; font-size: 18px; background: #0078D4; color: white; border: none; border-radius: 6px; }
#result { margin-top: 30px; }
</style>
</head>
<body>

<h2>ニュース総合分析ツール</h2>

<label>ニュースURL</label>
<input id="urlInput" placeholder="https://example.com/news/123">

<button onclick="extractNews()">ニュース本文を取得</button>

<label>抽出されたニュース本文</label>
<textarea id="newsInput" rows="6" readonly placeholder="ニュース本文がここに表示されます"></textarea>

<label>ティッカーコード</label>
<input id="tickerInput" placeholder="7203.T">

<button onclick="analyze()">分析する</button>

<div id="result"></div>

<script>
async function extractNews() {
    const url = document.getElementById("urlInput").value.trim();
    if (!url) {
        alert("ニュースURLを入力してください");
        return;
    }

    const r = await fetch("/extract_news", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(url)
    });

    const data = await r.json();
    document.getElementById("newsInput").value = data.text || "本文抽出に失敗しました";
}

async function analyze() {
    const newsText = document.getElementById("newsInput").value.trim();
    const ticker = document.getElementById("tickerInput").value.trim();

    if (!newsText) {
        alert("ニュース本文が空です（URLから抽出してください）");
        return;
    }
    if (!ticker) {
        alert("ティッカーコードを入力してください");
        return;
    }

    const res = await fetch("/analyze_news_with_ticker", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({news: newsText, ticker: ticker})
    });

    const data = await res.json();
    document.getElementById("result").innerHTML = data.html || "<p>エラーが発生しました</p>";
}
</script>

</body>
</html>
"""
