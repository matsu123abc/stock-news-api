from fastapi import FastAPI
import requests
import os
import re
import json
import yfinance as yf
from pydantic import BaseModel
from openai import AzureOpenAI
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from bs4 import BeautifulSoup
from fastapi import Query

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

# ============================
# /tools/news（Gradio版の移植）
# ============================
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

@app.get("/tools/news")
def get_news(keyword: str):
    url = "https://serpapi.com/search"
    params = {
        "engine": "google",
        "q": keyword + " ニュース",
        "api_key": SERPER_API_KEY,
        "num": 5
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []

    def safe(value):
        return value if value is not None else ""

    # ① top_stories
    if "top_stories" in data:
        for item in data["top_stories"]:
            articles.append({
                "title": safe(item.get("title")),
                "snippet": safe(item.get("snippet")),
                "link": safe(item.get("link")),
                "source": safe(item.get("source")),
                "date": safe(item.get("date"))
            })

    # ② organic_results
    if "organic_results" in data:
        for item in data["organic_results"]:
            articles.append({
                "title": safe(item.get("title")),
                "snippet": safe(item.get("snippet")),
                "link": safe(item.get("link")),
                "source": safe(item.get("source"))
            })

    # ③ news_results（あれば）
    if "news_results" in data:
        for item in data["news_results"]:
            articles.append({
                "title": safe(item.get("title")),
                "snippet": safe(item.get("snippet")),
                "link": safe(item.get("link")),
                "source": safe(item.get("source"))
            })

    return {
        "keyword": keyword,
        "count": len(articles),
        "articles": articles
    }

class NewsUrl(BaseModel):
    url: str

@app.post("/extract_news")
def extract_news(req: NewsUrl):
    """
    ニュースURLから本文を抽出するAPI
    """
    try:
        r = requests.get(req.url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        paragraphs = soup.select("p")
        text = "\n".join([p.get_text().strip() for p in paragraphs])
        text = "\n".join([line for line in text.split("\n") if line.strip()])

        return JSONResponse({"text": text})

    except Exception as e:
        return JSONResponse({"error": f"本文抽出エラー: {str(e)}"})
    
def summarize_similar_news(articles, ticker: str, company_name: str):
    if not articles:
        return "<p>類似ニュースが見つかりませんでした。</p>"

    lines = []
    for i, a in enumerate(articles, 1):
        title = a.get("title", "")
        link = a.get("link", "")
        source = a.get("source", "")
        date = a.get("date", "")
        snippet = a.get("snippet", "")

        lines.append(f"{i}. タイトル: {title}")
        lines.append(f"   URL: {link}")
        if source:
            lines.append(f"   メディア: {source}")
        if date:
            lines.append(f"   日付: {date}")
        if snippet:
            lines.append(f"   概要: {snippet}")
        lines.append("")

    news_block = "\n".join(lines)

    prompt = f"""
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
    res = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = res.choices[0].message.content.strip()

    # ★ f-string の外で整形する（これが重要）
    safe_text = (
        text.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )

    return f"""
<h3>類似ニュースまとめ分析</h3>
<div id="similarNewsSummary">
{safe_text}
</div>
"""

def search_similar_news(keyword: str):
    url = "https://stock-news-api-b3bzg9dzbtgmdxbz.japanwest-01.azurewebsites.net/tools/news"
    r = requests.get(url, params={"keyword": keyword}, timeout=30)
    r.raise_for_status()
    return r.json().get("articles", [])

def extract_keywords(news_text: str):
    prompt = f"""
以下のニュース本文から、Web検索に使える重要キーワードを5〜10個抽出してください。
・名詞のみ
・1行に1つ
・余計な説明は書かない

【ニュース本文】
{news_text}
"""
    res = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    lines = [l.strip() for l in res.choices[0].message.content.split("\n") if l.strip()]
    keywords = [re.sub(r"^[\-・\d\.]+\s*", "", l) for l in lines]
    return [k for k in keywords if k]

def analyze_similar_news(news_text: str, ticker: str):
    if not news_text.strip():
        return "ニュース本文が空です。"
    if not ticker.strip():
        return "ティッカーコードを入力してください。"

    # 企業名取得
    try:
        t = yf.Ticker(ticker)
        company_name = t.info.get("shortName", ticker)
    except Exception:
        company_name = ticker

    # 1. キーワード抽出
    keywords = extract_keywords(news_text)
    if not keywords:
        return "<p>キーワード抽出に失敗しました。</p>"

    query = " ".join(keywords)

    # 2. FastAPI 経由で類似ニュース検索
    try:
        articles = search_similar_news(query)

    except Exception as e:
        return f"<h3>類似ニュース検索エラー</h3><pre>{e}</pre>"

    if not articles:
        return "<p>類似ニュースが見つかりませんでした。</p>"

    # 3. 類似ニュース一覧テーブル
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
<table border="1" style="border-collapse: collapse;">
<tr><th>タイトル</th><th>メディア</th><th>日付</th><th>概要</th></tr>
""" + "\n".join(rows) + "</table>"

    # 4. GPT によるまとめ分析
    summary_html = summarize_similar_news(articles, ticker, company_name)

    return list_html + summary_html

# ============================
# 2. 類似ニュース分析 API
# ============================
class SimilarNewsRequest(BaseModel):
    news: str
    ticker: str

@app.post("/analyze_similar_news")
def analyze_similar_news_api(payload: SimilarNewsRequest):
    news_text = payload.news
    ticker = payload.ticker
    return analyze_similar_news(news_text, ticker)

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
    try:
        res_sum = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            messages=[{"role": "user", "content": prompt_sum}],
            temperature=0.2,
        )
        summary_ja = res_sum.choices[0].message.content.strip()
    except Exception:
        summary_ja = summary_en

    company_summary = summary_ja

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

    # --- 株価トレンド文章化（★追加） ---
    if ret_3m is not None:
        if ret_3m > 5:
            trend_text = "直近3ヶ月は上昇トレンド"
        elif ret_3m < -5:
            trend_text = "直近3ヶ月は下落トレンド"
        else:
            trend_text = "直近3ヶ月は横ばい"
    else:
        trend_text = "トレンドデータなし"

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

    # --- 再取得（company_name / sector_name）---
    info = yf.Ticker(ticker).info
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector_name = info.get("sector") or "不明"

    # --- ニュース分析 ---
    prompt = f"""
あなたはプロの株式アナリストです。
以下のニュースが「この銘柄にとってどれほど重要か」を深く分析してください。

【銘柄情報】
ティッカー: {ticker}
企業名: {company_name}
セクター: {sector_name}

【企業概要（要約）】
{company_summary}

【株価情報】
現在株価: {price_now}
1ヶ月リターン: {ret_1m} %
3ヶ月リターン: {ret_3m} %
株価トレンド: {trend_text}

【ニュース本文】
{news_text}

【タスク】
1. このニュースが「この銘柄にとって」どれほど重要かを -1.0〜+1.0 で数値化
2. 株価への影響を「強い上昇」「上昇」「中立」「下落」「強い下落」で評価
3. 企業の事業内容・収益構造との関係を深掘りして説明
4. セクター全体への影響と、競合他社との相対的な影響を説明
5. 短期（〜1ヶ月）・中期（〜6ヶ月）のリスクとチャンスを整理
6. 投資家が次に確認すべきポイント（2〜3個）を挙げる

【出力形式】
以下の3行だけを返してください：

スコア: （数値）
影響度: （強い上昇 / 上昇 / 中立 / 下落 / 強い下落）
理由: （文章）
"""

    res_news = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # GPT の出力をそのまま使用（JSON パース不要）
    analysis_text = res_news.choices[0].message.content.strip()

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
<pre style="font-size: 15px; white-space: pre-wrap;">{analysis_text}</pre>

{similar_html}

<h3>チャート</h3>
<p><a href="{chart_url}" target="_blank">Yahoo! JAPAN チャートを見る</a></p>
"""

    return {"html": html}

class RecommendRequest(BaseModel):
    ticker: str
    news: str
    similar_news_summary: str | None = None

@app.post("/recommend_stocks")
async def recommend_stocks(payload: dict):
    ticker = payload.get("ticker", "")
    news = payload.get("news", "")
    similar_news_summary = payload.get("similar_news_summary", "")

    prompt = f"""
あなたはプロの株式アナリストです。
以下のニュース内容と類似ニュースを踏まえ、
「今回のニューステーマから最も恩恵を受ける可能性が高い銘柄」を
セクターに限定せず、3〜5社選定してください。

【対象銘柄】
ティッカー: {ticker}

【ニュース本文】
{news}

【類似ニュース（要約またはキーワード）】
{similar_news_summary}

【分析条件】
- セクターに限定せず、テーマ横断で選定する
- 類似ニュースで過去に株価が動いた銘柄を優先する
- ニュースが示す成長領域・構造変化・投資テーマを抽出する
- そのテーマと事業ポートフォリオの相性が良い企業を選ぶ
- 時価総額、収益構造、海外比率、技術優位性も考慮する
- 「有名どころ」ではなく「恩恵の大きさ」で順位付けする

【出力形式】
1位: 銘柄名（ティッカー） - 理由（1〜2文）
2位: 〜
3位: 〜
"""

    res = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    text = res.choices[0].message.content.strip()

    html = f"""
<pre style="font-size: 15px; white-space: pre-wrap;">
{text}
</pre>
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
        #result, #similarResult, #recommendArea { margin-top: 30px; }
    </style>
</head>
<body>

    <h2>ニュース総合分析ツール</h2>

    <label>ニュースURL（任意）</label>
    <input id="urlInput" placeholder="https://example.com/news/123">

    <button onclick="extractNews()">ニュース本文を取得</button>

    <label>ニュース本文（自動抽出 or 手動貼り付け）</label>
    <textarea id="newsInput" rows="10" placeholder="ニュース本文を貼り付け、またはURLから抽出"></textarea>

    <button onclick="clearNews()">本文クリア</button>

    <label>ティッカーコード</label>
    <input id="tickerInput" placeholder="7203.T">

    <button onclick="analyze()">分析する</button>
    <button onclick="analyzeSimilar()">類似ニュースを検索</button>
    <button onclick="recommendStocks()">推奨銘柄を表示</button>

    <button onclick="saveAnalysisAsHtml()">HTML保存</button>

    <div id="result"></div>
    <div id="similarResult"></div>
    <div id="recommendArea"></div>

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
        body: JSON.stringify({ url: url })
    });

    const data = await r.json();
    document.getElementById("newsInput").value =
        data.text || "本文抽出に失敗しました。手動で貼り付けてください。";
}

async function analyze() {
    const newsText = document.getElementById("newsInput").value.trim();
    const ticker = document.getElementById("tickerInput").value.trim();

    if (!newsText) {
        alert("ニュース本文が空です");
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
    document.getElementById("result").innerHTML =
        data.html || "<p>エラーが発生しました</p>";
}

function clearNews() {
    document.getElementById("newsInput").value = "";
}

async function analyzeSimilar() {
    const newsText = document.getElementById("newsInput").value.trim();
    const ticker = document.getElementById("tickerInput").value.trim();

    if (!newsText) {
        alert("ニュース本文が空です");
        return;
    }
    if (!ticker) {
        alert("ティッカーコードを入力してください");
        return;
    }

    const res = await fetch("/analyze_similar_news", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({news: newsText, ticker: ticker})
    });

    const html = await res.text();
    document.getElementById("similarResult").innerHTML = html;
}

async function recommendStocks() {
    const ticker = document.getElementById("tickerInput").value.trim();
    const newsText = document.getElementById("newsInput").value.trim();
    const similarSummary = document.getElementById("similarNewsSummary")
        ? document.getElementById("similarNewsSummary").innerText.trim()
        : "";

    const res = await fetch("/recommend_stocks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            ticker: ticker,
            news: newsText,
            similar_news_summary: similarSummary
        })
    });

    const data = await res.json();
    document.getElementById("recommendArea").innerHTML =
        data.html || "<p>推奨銘柄取得エラー</p>";
}

/* HTML保存（正規表現なし・安全版） */
function saveAnalysisAsHtml() {
    const newsText = document.getElementById("newsInput").value.trim();
    const result = document.getElementById("result").innerHTML;
    const similar = document.getElementById("similarResult").innerHTML;
    const recommend = document.getElementById("recommendArea").innerHTML;

    const now = new Date();
    const timestamp =
        now.getFullYear().toString() +
        ("0" + (now.getMonth() + 1)).slice(-2) +
        ("0" + now.getDate()).slice(-2) + "_" +
        ("0" + now.getHours()).slice(-2) +
        ("0" + now.getMinutes()).slice(-2) +
        ("0" + now.getSeconds()).slice(-2);

    var safeNewsHtml = newsText
        ? newsText
            .split("&").join("&amp;")
            .split("<").join("&lt;")
            .split(">").join("&gt;")
            .split("\\n").join("<br>")
        : "（ニュース本文なし）";

    var fullHtml = ""
        + "<!DOCTYPE html><html lang='ja'><head><meta charset='UTF-8'>"
        + "<title>ニュース総合分析 保存版</title></head><body>"
        + "<h2>ニュース総合分析 保存版</h2>"
        + "<p>保存日時: " + timestamp + "</p>"
        + "<h3>ニュース本文</h3><div>" + safeNewsHtml + "</div>"
        + "<h3>ニュース分析</h3>" + result
        + "<h3>類似ニュースまとめ</h3>" + similar
        + "<h3>推奨銘柄</h3>" + recommend
        + "</body></html>";

    const blob = new Blob([fullHtml], { type: "text/html" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "news_analysis_" + timestamp + ".html";
    a.click();

    URL.revokeObjectURL(url);
}
</script>

</body>
</html>
"""
