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

# --- 全局參數 ---
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
        # 示範抓取 twstock 內建清單 (建議分批或使用完整清單)
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
    """【V8.3 價格修復版】"""
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

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.1 現貨實戰版核心】
    """
    try:
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 抓取 1 年數據
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None 

        # --- 1. CAPM (計算資金成本供評價用，不給舉債建議) ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        ke = RF + beta * MRP # 投資人要求報酬率

        # --- 2. Gordon Model (合理價) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 3. Fama-French 因子邏輯 ---
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = market_cap > 0 and market_cap < 50000000000
        pb = ticker_info.get('priceToBook', 0)
        is_value_stock = pb > 0 and pb < 1.5
        
        # --- 4. Smart Beta & 技術買點 ---
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        ma20 = data['Close'].rolling(20).mean().iloc[-1] # 月線
        
        cgo_val = (current_price - ma100) / ma100
        volatility = stock_returns.std() * (252**0.5)
        
        # 建議買點：設定為月線 (MA20)，這是現貨波段操作常見的支撐點
        entry_price = round(ma20, 2)

        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波") 
        
        # --- 5. AI 綜合評分系統 ---
        score = 0.0
        factors = []
        
        if is_value_stock:
            score += 15
            factors.append("💎價值型")
        if not np.isnan(fair_value) and fair_value > current_price:
            score += 20
            factors.append("💰低估")
            
        if is_small_cap:
            score += 10
            
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 15
            factors.append("📈高成長")
            
        if current_price > ma20:
            score += 10 

        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append("👑高ROE")
            
        if volatility < 0.25:
            score += 15
            factors.append("🛡️籌碼穩")
        elif volatility > 0.5:
            score -= 10
            
        # AI 推薦語
        ai_eval = "🟡 觀察"
        if score >= 75:
            ai_eval = "🚀 強力買進"
        elif score >= 60:
            ai_eval = "🟢 積極佈局"
        elif score >= 50:
            ai_eval = "🔵 持有/觀望"

        if score >= 50:
            return {
                "代號": ticker_symbol.replace('.TW', '').replace('.TWO', ''),
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "AI建議": ai_eval,
                "現價": float(current_price),
                "建議買點": float(entry_price),
                "評分": round(score, 1),
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "CGO指標": round(cgo_val * 100, 1),
                "策略標籤": " ".join(strategy_tags),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.1", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.1 (現貨實戰版)")
st.markdown("""
本系統專注於 **現貨買入策略**，結合 AI 綜合評分與技術面支撐，篩選全市場最優質的標的。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統將篩選「Top 100」推薦個股，並計算建議買入點位。")
    if st.button("🚀 啟動 V9.1 智能掃描", type="primary"):
        with st.spinner("Step 1: 取得市場數據..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"開始分析 {len(tickers)} 檔股票...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in tickers}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed += 1
                if completed % 10 == 0:
                    progress_bar.progress(completed / len(tickers))
                    status_text.text(f"分析中: {completed}/{len(tickers)}")
                if data:
                    st.session_state['results'].append(data)

        status_text.text("✅ 分析完成！")

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊按鈕開始分析。")
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序與篩選：先按評分高低排序，取前 100 名
        df = df.sort_values(by=['評分'], ascending=False).head(100)
        
        st.subheader(f"🏆 AI 嚴選：最推薦優先買入 Top 100")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "AI建議", "現價", "建議買點", "合理價", "評分", "策略標籤", "CGO指標", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "AI建議": st.column_config.TextColumn(width="small", help="AI 根據財務與技術面綜合判斷"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="技術面支撐點位 (月線 MA20)，適合現貨佈局"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model 理論價值"),
                "評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="未實現獲利指標，越高代表籌碼越安定"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
