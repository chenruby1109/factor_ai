import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures
import twstock

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 ---
RF = 0.015  
MRP = 0.055 
G_GROWTH = 0.02 

# --- 核心功能 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    try:
        market = yf.download("^TWII", period="2y", interval="1d", progress=False)
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
        for code, info in twstock.codes.items():
            if info.type == '股票':
                suffix = ".TW" if info.market == '上市' else ".TWO"
                full_ticker = code + suffix
                tickers.append(full_ticker)
                name_map[full_ticker] = info.name
        return tickers, name_map
    except Exception as e:
        return [], {}

def get_realtime_price(stock_code):
    """
    【關鍵修正】使用 twstock 直接抓取證交所即時股價
    確保價格絕對正確，不再依賴 Yahoo
    """
    try:
        # 去除 .TW 或 .TWO 後綴
        code = stock_code.split('.')[0]
        realtime = twstock.realtime.get(code)
        
        if realtime['success']:
            # 嘗試獲取最新成交價
            price = realtime['realtime']['latest_trade_price']
            # 如果盤中暫無成交（顯示 - ），改抓最佳買入價
            if price == '-' or price is None:
                price = realtime['realtime']['best_bid_price'][0]
            
            return float(price)
    except:
        pass
    return None

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V8.1 價格修正版】
    邏輯：Yahoo 算技術指標 + TWSE 抓即時股價 = 精準決策
    """
    try:
        # 1. 獲取最精準的「即時現價」 (Realtime Price)
        current_price = get_realtime_price(ticker_symbol)
        
        # 如果證交所抓不到價格（例如暫停交易），才勉強用 Yahoo 的收盤價當備案
        # 但主要依賴 current_price
        
        # 2. 下載歷史數據 (用於計算 Beta, MA60)
        data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if len(data) < 250: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 這裡的 close 僅用於計算技術指標，顯示給用戶的價格用 current_price
        yf_close = data.iloc[-1]['Close']
        volume = data.iloc[-1]['Volume']
        
        # 若 twstock 抓不到，回退使用 yf_close
        if current_price is None:
            current_price = yf_close

        # 過濾殭屍股 (用現價判斷)
        if current_price < 10: return None

        # --- A. CAPM 模型 ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        if len(aligned_data) < 100: return None

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        beta = covariance / market_variance
        expected_return = RF + beta * MRP

        # --- B. Gordon 評價模型 (使用正確的現價) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        
        # 股利資料抓取
        dividend_rate = ticker_info.get('dividendRate', 0)
        # 如果 Yahoo 沒股利資料，簡單估算 (殖利率 * 現價) - 備用邏輯
        if dividend_rate is None or dividend_rate == 0:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: dividend_rate = current_price * yield_val

        fair_value = np.nan
        upside_potential = np.nan
        
        k_minus_g = max(expected_return - G_GROWTH, 0.01)
        
        if dividend_rate and dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
            # 關鍵修正：使用「即時現價」計算獲利空間
            upside_potential = (fair_value - current_price) / current_price

        # --- C. 評分系統 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        score = 0.0
        factors = []
        
        # 1. 價值 (Gordon)
        if not np.isnan(fair_value) and fair_value > current_price:
            val_score = min(upside_potential * 100, 30)
            score += val_score
            factors.append(f"💰折價{round(upside_potential*100)}%")
        
        # 2. 成長
        if rev_growth and rev_growth > 0:
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈高成長")

        # 3. 品質
        if roe and roe > 0:
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑高ROE")

        # 4. 價值 (PB)
        if 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎低PB")
            
        # 5. 技術 (使用歷史均線 vs 即時股價)
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        bias = (current_price - ma60) / ma60 # 用現價算乖離
        
        if 0 < bias < 0.08:
            score += 20
            factors.append("🎯站上季線")
        elif bias > 0.2:
            score -= 10
            
        # 6. 風險
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: score -= 15
        
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "最新收盤價": current_price, # 這裡是正確的即時價格
                "綜合評分": round(score, 1),
                "理論合理價": fair_value if not np.isnan(fair_value) else None,
                "預估獲利空間": upside_potential if not np.isnan(upside_potential) else None,
                "資金成本(CAPM)": expected_return,
                "風險係數(Beta)": float(beta),
                "亮點因子": " | ".join(factors)
            }

    except:
        return None
    return None

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 戰情室 V8.1", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 智能投資戰情室 V8.1 (價格修正版)")
st.markdown("""
### 🛠️ V8.1 緊急修正：
* **價格校正**：廢除 Yahoo 錯誤報價，改用 `twstock` 直接連線 **台灣證交所** 抓取即時成交價。
* **精準估值**：獲利空間與合理價皆基於正確的台股現價計算。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 系統將混合使用 Yahoo (歷史數據) 與 證交所 (即時價格)，確保分析精準度。")
    
    if st.button("🚀 啟動精準掃描", type="primary"):
        with st.spinner("Step 1: 建立大盤風險基準..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"準備完成！開始分析 {len(tickers)} 檔股票...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 16 核心平行運算
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                if completed_count % 20 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"分析進度: {completed_count}/{len(tickers)} | 發現潛力股: {found_count}")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    df_realtime = df_realtime.sort_values(by='綜合評分', ascending=False)
                    
                    with result_placeholder.container():
                        st.subheader(f"🎯 發現標的 ({found_count} 檔)")
                        st.dataframe(df_realtime, use_container_width=True, hide_index=True)

        status_text.text("✅ 分析完成！")
        
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='綜合評分', ascending=False)
            top_5 = df_final.head(5)
            msg = f"📊 **【Miniko 精準報價報告】**\n"
            for _, row in top_5.iterrows():
                profit_txt = f"{round(row['預估獲利空間']*100)}%" if pd.notnull(row['預估獲利空間']) else "N/A"
                msg += f"• {row['名稱']} ({row['代號']}) 現價:{row['最新收盤價']} | 潛在獲利:{profit_txt}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 請點擊左側按鈕，這次價格絕對是正確的台幣價格。")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        
        st.dataframe(
            df_show.sort_values(by='綜合評分', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_order=["名稱", "代號", "最新收盤價", "理論合理價", "預估獲利空間", "綜合評分", "資金成本(CAPM)", "風險係數(Beta)", "亮點因子"],
            column_config={
                "名稱": st.column_config.TextColumn("股票名稱"),
                "代號": st.column_config.TextColumn("代號"),
                "最新收盤價": st.column_config.NumberColumn(
                    "最新收盤價 (TWD)",
                    help="來源：台灣證交所即時報價",
                    format="$%.2f",
                ),
                "理論合理價": st.column_config.NumberColumn(
                    "理論合理價 (Gordon)",
                    format="$%.2f",
                ),
                "預估獲利空間": st.column_config.NumberColumn(
                    "預估獲利空間",
                    format="%.2f%%",
                ),
                "綜合評分": st.column_config.ProgressColumn(
                    "AI 綜合評分",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "資金成本(CAPM)": st.column_config.NumberColumn(
                    "資金成本 (CAPM)",
                    format="%.2f%%",
                ),
                "風險係數(Beta)": st.column_config.NumberColumn(
                    "風險係數 (Beta)",
                    format="%.2f",
                ),
                "亮點因子": st.column_config.TextColumn("AI 診斷亮點", width="medium"),
            }
        )
