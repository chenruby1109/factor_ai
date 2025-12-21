import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 (針對現貨交易調整) ---
RF = 0.015  # 無風險利率
MRP = 0.055 # 市場風險溢酬
G_GROWTH = 0.02 # 股利長期成長率

# --- 核心功能函數 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    """下載大盤指數 (TWII)"""
    try:
        market = yf.download("^TWII", period="1y", interval="1d", progress=False)
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = market.columns.get_level_values(0)
        market['Return'] = market['Close'].pct_change()
        return market['Return'].dropna()
    except:
        return pd.Series()

@st.cache_data(ttl=3600) 
def get_all_tw_tickers():
    tickers = []
    name_map = {}
    try:
        # 抓取 twstock 內建清單
        for code, info in twstock.codes.items():
            if info.type == '股票':
                suffix = ".TW" if info.market == '上市' else ".TWO"
                full_ticker = code + suffix
                tickers.append(full_ticker)
                name_map[full_ticker] = info.name
        return tickers, name_map
    except Exception as e:
        return [], {}

def get_realtime_price_robust(stock_code):
    """【V8.3 價格修復版】(History + Realtime 雙重驗證)"""
    price = None
    try:
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    if price is None:
        try:
            code = stock_code.split('.')[0]
            realtime = twstock.realtime.get(code)
            if realtime['success']:
                rt_price = realtime['realtime']['latest_trade_price']
                if rt_price and rt_price != '-' and float(rt_price) > 0:
                    price = float(rt_price)
                else:
                    best_bid = realtime['realtime']['best_bid_price'][0]
                    if best_bid and best_bid != '-' and float(best_bid) > 0:
                        price = float(best_bid)
        except: pass
    return price

def calculate_ai_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.1 AI 綜合評估核心】
    針對「現貨買入」優化：移除融資建議，加入買點計算與 AI 評分
    """
    try:
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 抓取數據 (拉長至 1 年以計算年線與波動)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None # 排除雞蛋水餃股

        # 基礎計算
        stock_returns = data['Close'].pct_change().dropna()
        ticker_info = yf.Ticker(ticker_symbol).info
        
        # --- 1. 技術指標與買點計算 ---
        ma20 = data['Close'].rolling(20).mean().iloc[-1]  # 月線 (支撐/買點)
        ma60 = data['Close'].rolling(60).mean().iloc[-1]  # 季線 (趨勢)
        ma100 = data['Close'].rolling(100).mean().iloc[-1] # 用於 CGO 成本
        
        # 建議買點邏輯：
        # 如果是強勢股(在月線之上)，建議掛在月線(MA20)附近接，不要追高。
        # 如果股價已經修正到月線下，則建議以現價觀察。
        suggested_buy_price = ma20 if current_price > ma20 else current_price

        # --- 2. CGO 與 波動率 ---
        cgo_val = (current_price - ma100) / ma100
        volatility = stock_returns.std() * (252**0.5)

        # --- 3. Gordon 合理價 ---
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        # 計算 Ke (資金成本) 僅用於折現，不給融資建議
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        if len(aligned) > 60:
            cov = aligned.cov().iloc[0, 1]
            mkt_var = aligned['Market'].var()
            beta = cov / mkt_var if mkt_var != 0 else 1.0
            ke = RF + beta * MRP
            k_minus_g = max(ke - G_GROWTH, 0.015) 
            if div_rate and div_rate > 0:
                fair_value = round(div_rate / k_minus_g, 2)

        # --- 4. AI 綜合評估 (0-100分) ---
        # 這是專為「現貨波段」設計的權重
        ai_score = 0
        highlights = []

        # A. 趨勢面 (Trend) - 佔 30分
        if current_price > ma20 and ma20 > ma60:
            ai_score += 20
            highlights.append("📈多頭排列")
        if current_price > ma60:
            ai_score += 10

        # B. 籌碼/情緒面 (CGO + Vol) - 佔 30分
        if cgo_val > 0.05: # 大部分人賺錢，惜售
            ai_score += 15
            highlights.append("🔥籌碼鎖定(CGO高)")
        if volatility < 0.35: # 波動穩定
            ai_score += 15
            highlights.append("🛡️波動穩定")

        # C. 基本面 (Value/Growth) - 佔 25分
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            ai_score += 15
            highlights.append("👑高ROE")
        
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            ai_score += 10
            highlights.append("🚀營收高成長")

        # D. 估值面 (Valuation) - 佔 15分
        pb = ticker_info.get('priceToBook', 0)
        if pb > 0 and pb < 2.0:
            ai_score += 15
            highlights.append("💎股價低估")

        # 額外加分：安全邊際
        if not np.isnan(fair_value) and fair_value > current_price * 1.1:
            ai_score += 5
            highlights.append("💰低於合理價")

        # 篩選門檻：分數太低的不顯示
        if ai_score >= 60:
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""), # 簡化代號顯示
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI綜合評分": ai_score,
                "建議買點": round(suggested_buy_price, 2), # 新增建議買點
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "CGO指標": round(cgo_val * 100, 1),
                "波動率": round(volatility, 2),
                "亮點": " | ".join(highlights)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 曜鼎豐 - AI 智能選股 V9.1", layout="wide")

st.title("🦄 Miniko & 曜鼎豐 - AI 智能選股戰情室 V9.1")
st.markdown("""
**專屬設定：** 現貨交易模式 (No Leverage) | AI 綜合評分 Top 100 | 智能買點計算
""")

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 按下按鈕後，AI 將掃描全台股，並依照綜合分數選出前 100 檔最強現貨標的。")
    
    if st.button("🚀 啟動 AI 全面掃描", type="primary"):
        with st.spinner("Step 1: 讀取市場數據與參數..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定目標：{len(tickers)} 檔股票，開始 AI 運算...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 平行運算加速
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(calculate_ai_factors, t, name_map, market_returns): t for t in tickers}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed += 1
                if completed % 20 == 0: # 減少更新頻率以提升效能
                    progress_bar.progress(completed / len(tickers))
                    status_text.text(f"AI 分析中: {completed}/{len(tickers)}")
                if data:
                    st.session_state['results'].append(data)

        status_text.text("✅ AI 運算完成！")

with col2:
    if 'results' not in st.session_state or not st.session_state['results']:
        st.warning("👈 請點擊左側按鈕開始分析。")
        st.markdown("### 📊 AI 評分邏輯說明")
        st.markdown("""
        * **趨勢 (30%)**：股價是否站在月線/季線之上 (多頭排列)。
        * **籌碼 (30%)**：CGO 指標 (惜售程度) 與 低波動率 (籌碼穩定)。
        * **基本 (25%)**：高 ROE 與 營收成長率。
        * **估值 (15%)**：低股價淨值比 (PB) 與 合理價位。
        """)
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # --- 關鍵邏輯：只取 AI 評分最高的前 100 檔 ---
        df = df.sort_values(by=['AI綜合評分', 'CGO指標'], ascending=[False, False])
        df_top100 = df.head(100) # 取前 100
        
        st.subheader(f"🏆 AI 精選推薦：前 100 檔優質現貨 ({len(df_top100)}/{len(df)})")
        
        st.dataframe(
            df_top100,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "AI綜合評分", "現價", "建議買點", "合理價", "CGO指標", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(help="股票代號"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="依據月線(20MA)計算之支撐價位，若現價過高建議等待回調"),
                "AI綜合評分": st.column_config.ProgressColumn(
                    format="%d 分", 
                    min_value=0, 
                    max_value=100,
                    help="Miniko AI 綜合多因子評分，越高越好"
                ),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model 估算"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值越大代表籌碼越穩"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
