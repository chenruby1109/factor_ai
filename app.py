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
        # 為了演示速度，這裡示範抓取部分熱門股，若要全市場請解開註解或使用完整 twstock.codes
        # 這裡示範抓取 twstock 內建的清單
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
    """
    【V8.3 終極價格修復版】
    解決週末/盤後價格為 0 或異常的問題。
    """
    price = None
    
    # --- 策略 1: yfinance 歷史數據 (最穩定，適合週末/盤後) ---
    try:
        # 抓 5 天是為了避開連假，取最後一筆非 NaN 的 Close
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except:
        pass

    # --- 策略 2: twstock (僅在平日盤中或 yf 失敗時做為輔助) ---
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
        except:
            pass

    return price

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V8.3 運算核心】
    """
    try:
        # 1. 獲取絕對正確的價格
        current_price = get_realtime_price_robust(ticker_symbol)
        
        # 如果價格還是抓不到或是 0，直接跳過
        if current_price is None or current_price <= 0: 
            return None

        # 2. 下載歷史數據
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 過濾雞蛋水餃股
        if current_price < 10: return None

        # --- A. CAPM 模型 ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        if len(aligned_data) < 60: return None

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        beta = covariance / market_variance if market_variance != 0 else 1.0
        expected_return = RF + beta * MRP

        # --- B. Gordon 模型 ---
        ticker_info = yf.Ticker(ticker_symbol).info
        dividend_rate = ticker_info.get('dividendRate', 0)
        
        if dividend_rate is None or dividend_rate == 0:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: dividend_rate = current_price * yield_val

        fair_value = np.nan
        upside_potential = np.nan
        
        k_minus_g = max(expected_return - G_GROWTH, 0.015) 
        
        if dividend_rate and dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
            upside_potential = (fair_value - current_price) / current_price

        # --- C. 數據準備 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        # --- D. 評分系統 ---
        score = 0.0
        factors = []
        
        # 1. 價值
        if not np.isnan(fair_value) and fair_value > current_price:
            val_score = min(upside_potential * 100, 30)
            score += val_score
            factors.append(f"💰低於合理價")
        
        # 2. 成長
        if rev_growth and rev_growth > 0:
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈營收高成長")

        # 3. 品質
        if roe and roe > 0:
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑高股東權益報酬")

        # 4. 價值 (PB)
        if pb_ratio and 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎低股價淨值比")
            
        # 5. 技術 (Momentum)
        if len(data) > 60:
            ma60 = data['Close'].rolling(60).mean().iloc[-1]
            bias = (current_price - ma60) / ma60
            if 0 < bias < 0.08:
                score += 20
                factors.append("🎯剛站上季線")
            elif bias > 0.2:
                score -= 10
        
        # 6. 風險
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: score -= 15
        
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "最新收盤價": float(current_price), 
                "綜合評分": round(score, 1),
                "理論合理價": fair_value if not np.isnan(fair_value) else None,
                "預估獲利空間": upside_potential if not np.isnan(upside_potential) else None,
                "資金成本": expected_return,
                "風險係數": float(beta),
                "亮點因子": " | ".join(factors)
            }

    except:
        return None
    return None

# --- Streamlit 頁面顯示區 (這是原本缺少的部分) ---

st.set_page_config(page_title="Miniko 戰情室 V8.4", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資理論實戰模型 V8.4 (修復版)")
st.markdown("""
本系統結合三大財務理論，為您計算股票真實價值。價格來源已修正為 **V8.3 雙重驗證 (History + Realtime)**。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 為了確保價格正確，系統優先採用歷史收盤價(適合週末)，盤中則切換為即時報價。")
    
    if st.button("🚀 啟動精準估值掃描", type="primary"):
        with st.spinner("Step 1: 計算大盤風險參數 (Beta基準)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入全台股清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"準備就緒！開始分析 {len(tickers)} 檔股票...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 平行運算
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                if completed_count % 10 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"分析進度: {completed_count}/{len(tickers)} | 發現潛力股: {found_count}")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    # 即時顯示部分結果
                    if found_count % 5 == 0:
                        df_realtime = pd.DataFrame(st.session_state['results'])
                        df_realtime = df_realtime.sort_values(by='綜合評分', ascending=False)
                        with result_placeholder.container():
                            st.subheader(f"🎯 掃描中... ({found_count} 檔)")
                            st.dataframe(df_realtime.head(10), use_container_width=True, hide_index=True)

        status_text.text("✅ 全市場分析完成！")
        
        if st.session_state['results']:
            # 這裡可以放發送 Telegram 的邏輯
            pass

with col2:
    if not st.session_state['results']:
        st.write("👈 請點擊左側按鈕，開始進行正確價格的理論估值。")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        
        st.dataframe(
            df_show.sort_values(by='綜合評分', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_order=["名稱", "代號", "最新收盤價", "理論合理價", "預估獲利空間", "綜合評分", "資金成本", "風險係數", "亮點因子"],
            column_config={
                "名稱": st.column_config.TextColumn("股票名稱"),
                "代號": st.column_config.TextColumn("代號"),
                "最新收盤價": st.column_config.NumberColumn("最新收盤價", format="$%.2f"),
                "理論合理價": st.column_config.NumberColumn("理論合理價", format="$%.2f"),
                "預估獲利空間": st.column_config.NumberColumn("獲利空間", format="%.2f%%"),
                "綜合評分": st.column_config.ProgressColumn("AI 評分", format="%.1f", min_value=0, max_value=100),
                "資金成本": st.column_config.NumberColumn("資金成本", format="%.2f%%"),
                "風險係數": st.column_config.NumberColumn("Beta", format="%.2f"),
            }
        )
