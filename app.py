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

# --- 全局參數 (參考您的投資理論筆記) ---
RF = 0.015  # 無風險利率 (Risk-Free Rate, 假設 1.5%)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium, 假設 5.5%)
G_GROWTH = 0.02 # 股利長期成長率 (Gordon Growth Rate, 2%)

# --- 核心功能 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    """下載大盤指數 (TWII) - 用於計算系統性風險 Beta"""
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
    """從 twstock 獲取股票清單"""
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

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V8.0 智能戰情版】
    包含：中文化欄位、預估獲利空間、動態價格
    """
    try:
        # 1. 下載數據 (2年)
        data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if len(data) < 250: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        close = curr['Close']
        volume = curr['Volume']
        
        # 過濾殭屍股
        if volume < 100000 or close < 10: return None

        # --- A. CAPM 模型 (資本資產定價) ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        if len(aligned_data) < 100: return None

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        
        # Beta (系統性風險)
        beta = covariance / market_variance
        
        # 預期報酬率 (權益資金成本)
        expected_return = RF + beta * MRP

        # --- B. Gordon 評價模型 (合理股價) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        dividend_rate = ticker_info.get('dividendRate', 0)
        if dividend_rate is None: dividend_rate = 0
        
        fair_value = np.nan
        upside_potential = np.nan
        
        # P = D / (Re - g)
        k_minus_g = max(expected_return - G_GROWTH, 0.01)
        
        if dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
            # 計算潛在獲利空間
            upside_potential = (fair_value - close) / close

        # --- C. 數據準備 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        # --- D. 連續性評分系統 ---
        score = 0.0
        factors = []
        
        # 1. 價值 (Gordon Upside)
        if not np.isnan(fair_value) and fair_value > close:
            val_score = min(upside_potential * 100, 30)
            score += val_score
            factors.append(f"💰折價{round(upside_potential*100)}%")
        
        # 2. 成長 (Revenue)
        if rev_growth and rev_growth > 0:
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈高成長")

        # 3. 品質 (ROE)
        if roe and roe > 0:
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑高ROE")

        # 4. 價值 (PB)
        if 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎低PB")
            
        # 5. 技術 (Momentum)
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        bias = (close - ma60) / ma60
        if 0 < bias < 0.08:
            score += 20
            factors.append("🎯站上季線")
        elif bias > 0.2:
            score -= 10
            
        # 6. 風險 (Volatility)
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: 
            score -= 15
        
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "最新收盤價": close,
                "綜合評分": round(score, 1),
                "理論合理價": fair_value if not np.isnan(fair_value) else None,
                "預估獲利空間": upside_potential if not np.isnan(upside_potential) else None,
                "資金成本(CAPM)": expected_return, # 這裡存小數，顯示時轉百分比
                "風險係數(Beta)": float(beta),
                "亮點因子": " | ".join(factors)
            }

    except:
        return None
    return None

# --- Streamlit 頁面佈局 ---

st.set_page_config(page_title="Miniko 智能戰情室 V8", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 智能投資戰情室 V8")
st.markdown("""
此系統結合 **CAPM**、**Gordon Model** 與 **Fama-French** 理論，為您計算每檔股票的真實價值。
* **資料來源**：即時串接 Yahoo Finance (價格隨開盤浮動，約15分延遲)。
* **預估獲利空間**：(理論合理價 - 最新收盤價) / 最新收盤價。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 系統正在進行複雜的金融模型運算 (CAPM + Gordon)，分析全市場約需 20 分鐘。")
    
    if st.button("🚀 啟動全市場估值掃描", type="primary"):
        with st.spinner("Step 1: 建立大盤風險基準 (Market Risk)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票代號清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"準備完成！開始分析 {len(tickers)} 檔股票...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 平行運算
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                if completed_count % 50 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"分析進度: {completed_count}/{len(tickers)} | 發現潛力股: {found_count}")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    df_realtime = df_realtime.sort_values(by='綜合評分', ascending=False)
                    
                    # 即時顯示 (簡單版)
                    with result_placeholder.container():
                        st.subheader(f"🎯 發現標的 ({found_count} 檔)")
                        st.dataframe(df_realtime, use_container_width=True, hide_index=True)

        status_text.text("✅ 分析完成！")
        
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='綜合評分', ascending=False)
            top_5 = df_final.head(5)
            msg = f"📊 **【Miniko 估值報告】**\n"
            for _, row in top_5.iterrows():
                profit_txt = f"{round(row['預估獲利空間']*100)}%" if pd.notnull(row['預估獲利空間']) else "N/A"
                msg += f"• {row['名稱']} ({row['代號']}) 現價:{row['最新收盤價']} | 潛在獲利:{profit_txt}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 請點擊左側按鈕，開始尋找被低估的優質股。")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        
        # --- 關鍵修改：使用 column_config 進行中文化與視覺優化 ---
        st.dataframe(
            df_show.sort_values(by='綜合評分', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_order=["名稱", "代號", "最新收盤價", "理論合理價", "預估獲利空間", "綜合評分", "資金成本(CAPM)", "風險係數(Beta)", "亮點因子"],
            column_config={
                "名稱": st.column_config.TextColumn("股票名稱"),
                "代號": st.column_config.TextColumn("代號"),
                "最新收盤價": st.column_config.NumberColumn(
                    "最新收盤價",
                    help="即時更新的市場價格 (約15分延遲)",
                    format="$%.2f",
                ),
                "理論合理價": st.column_config.NumberColumn(
                    "理論合理價 (Gordon)",
                    help="根據 Gordon Model 估算的內在價值：股利 / (資金成本 - 成長率)",
                    format="$%.2f",
                ),
                "預估獲利空間": st.column_config.NumberColumn(
                    "預估獲利空間",
                    help="潛在漲幅 = (合理價 - 現價) / 現價。正值代表被低估。",
                    format="%.2f%%", # 百分比顯示
                ),
                "綜合評分": st.column_config.ProgressColumn(
                    "AI 綜合評分",
                    help="結合 F-G-M 模型 (基本面、成長、動能) 的總分，滿分約 100",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "資金成本(CAPM)": st.column_config.NumberColumn(
                    "資金成本 (CAPM)",
                    help="投資人要求的最低預期報酬率 (Re = Rf + Beta * MRP)",
                    format="%.2f%%",
                ),
                "風險係數(Beta)": st.column_config.NumberColumn(
                    "風險係數 (Beta)",
                    help="衡量相對於大盤的波動風險。Beta > 1 代表波動比大盤大；Beta < 1 代表較穩健。",
                    format="%.2f",
                ),
                "亮點因子": st.column_config.TextColumn(
                    "AI 診斷亮點",
                    help="符合的投資理論因子 (如：價值股、小型股溢酬、動能等)",
                    width="medium"
                ),
            }
        )
