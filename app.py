import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock
import gc # 新增垃圾回收機制

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
    """【V8.3 價格修復版】"""
    price = None
    # 策略 1: yfinance History
    try:
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    # 策略 2: twstock Realtime
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
    """【Miniko V10.0 AI 旗艦運算核心 - 穩定版】"""
    try:
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 抓取數據
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None # 排除雞蛋水餃股

        # --- 1. 技術面與建議買點 ---
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
        if current_price > ma20:
            suggested_buy_point = ma20
        else:
            suggested_buy_point = current_price * 0.98
            
        suggested_buy_point = round(suggested_buy_point, 2)

        # --- 2. CAPM ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        ke = RF + beta * MRP
        operation_mode = "現貨持有"

        # --- 3. Gordon Model ---
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 4. Smart Beta & CGO ---
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        volatility = stock_returns.std() * (252**0.5)
        
        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波優選") 
        
        # --- 5. AI 綜合評分 ---
        score = 0.0
        factors = []
        
        pb = ticker_info.get('priceToBook', 0)
        if pb > 0 and pb < 1.5:
            score += 15
            factors.append("💎低PB價值")
        if not np.isnan(fair_value) and fair_value > current_price * 1.1:
            score += 20
            factors.append("💰低估潛力股")
            
        market_cap = ticker_info.get('marketCap', 0)
        if market_cap > 0 and market_cap < 50000000000:
            score += 10
            factors.append("🐟中小型爆發")
            
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 15
            factors.append("📈高成長")
        
        if current_price > ma20 and ma20 > ma60:
            score += 10
            factors.append("🐂多頭排列")

        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append("👑高ROE")
            
        if volatility < 0.25:
            score += 15
            factors.append("🛡️籌碼穩定")
        elif volatility > 0.5:
            score -= 10
            
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "建議買入點": suggested_buy_point,
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "操作模式": operation_mode,
                "CGO指標": round(cgo_val * 100, 1),
                "策略標籤": " ".join(strategy_tags),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V10.0", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V10.0 (AI 現貨防當機版)")
st.markdown("""
本系統整合 **AI 綜合評分、Gordon 模型** 與 **Smart Beta** 策略，專為 **現股買進** 打造。
*已優化雲端運算效能，避免記憶體溢出。*
""")

# --- 知識庫 Expander ---
with st.expander("📚 點此查看：投資理論與籌碼面分析教學 (Miniko 專屬)"):
    tab1, tab2, tab3 = st.tabs(["籌碼面六大指標", "Fama-French與多因子", "CGO與低波動策略"])
    
    with tab1:
        st.markdown("""
        ### 🕵️ 籌碼面六大指標
        1. **千張大戶持股**：>40% 代表集中。
        2. **內部人持股**：>40% 適合長期持有。
        3. **佔股本比重**：>3% 代表主力介入。
        4. **籌碼集中度**：60天 > 5% 為佳。
        5. **主力買賣超**：與股價同步為正常。
        6. **買賣家數差**：負數代表籌碼集中（必勝訊號）。
        """)
    with tab2:
        st.markdown("""
        ### 📈 Fama-French 三因子
        * **CAPM**：計算資金成本。
        * **SMB (規模)**：關注中小型股爆發力。
        * **HML (價值)**：關注低 PB 價值股。
        """)
    with tab3:
        st.markdown("""
        ### 🚀 CGO + 低波動
        * **CGO**：正值代表大家都在賺錢，惜售（支撐強）。
        * **低波動**：籌碼穩定，長線報酬佳。
        """)

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統將進行 AI 掃描 (分批運算以確保穩定)。")
    if st.button("🚀 啟動 AI 智能掃描", type="primary"):
        with st.spinner("Step 1: 準備市場數據..."):
            market_returns = get_market_data()
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"開始分析 {len(tickers)} 檔股票 (分批執行)...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- 核心修改：分批處理邏輯 ---
        BATCH_SIZE = 50 # 每次處理 50 檔
        total_processed = 0
        
        # 降低 workers 到 4 以避免記憶體不足 (RuntimeError)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i : i + BATCH_SIZE]
                
                # 送出這一批任務
                future_to_ticker = {
                    executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t 
                    for t in batch
                }
                
                # 收集這一批結果
                for future in concurrent.futures.as_completed(future_to_ticker):
                    data = future.result()
                    if data:
                        st.session_state['results'].append(data)
                    total_processed += 1
                
                # 更新進度條
                progress_bar.progress(min(total_processed / len(tickers), 1.0))
                status_text.text(f"AI 運算中: {total_processed}/{len(tickers)}")
                
                # 強制釋放記憶體
                gc.collect()

        status_text.text("✅ AI 分析完成！")

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊按鈕開始分析。")
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序：AI 評分優先，其次 CGO
        df = df.sort_values(by=['AI綜合評分', 'CGO指標'], ascending=[False, False])
        
        # 取 Top 100
        top_100_df = df.head(100)
        
        st.subheader(f"🏆 AI 嚴選 Top 100 強力買入清單")
        st.caption("篩選標準：AI 綜合評分最高的前 100 檔，現貨操作建議。")
        
        st.dataframe(
            top_100_df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "建議買入點", "AI綜合評分", "合理價", "策略標籤", "操作模式", "亮點"],
            column_config={
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "建議買入點": st.column_config.NumberColumn(format="$%.2f"),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
