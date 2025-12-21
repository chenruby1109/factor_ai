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

# --- 全局參數 (依據您的筆記設定) ---
RF = 0.015  # 無風險利率 (假設 1.5% 定存)
MRP = 0.05  # 市場風險溢酬 (Rm - Rf, 假設 5%)
G_GROWTH = 0.02 # 股利長期成長率假設 (保守估計 2%)

# --- 核心功能 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    """下載大盤指數 (TWII) 用於計算 Beta (CAPM)"""
    try:
        # 抓取 1 年數據以計算 Beta
        market = yf.download("^TWII", period="1y", interval="1d", progress=False)
        if isinstance(market.columns, pd.MultiIndex):
            market.columns = market.columns.get_level_values(0)
        # 計算日報酬率
        market['Return'] = market['Close'].pct_change()
        return market['Return'].dropna()
    except:
        return pd.Series()

@st.cache_data(ttl=3600) 
def get_all_tw_tickers():
    """使用 twstock 直接調用內建字典"""
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
    【Miniko V7 華爾街理論版】
    整合 CAPM, Gordon Model, Fama-French 三因子
    """
    try:
        # 1. 下載個股數據 (1年)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        
        if len(data) < 200: return None # 資料不足一年不計算
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 準備基本數據
        curr = data.iloc[-1]
        close = curr['Close']
        volume = curr['Volume']
        
        # 過濾殭屍股
        if volume < 200000 or close < 10: return None

        # --- A. CAPM 模型計算 (資本資產定價) ---
        # 1. 計算個股日報酬
        stock_returns = data['Close'].pct_change().dropna()
        
        # 2. 合併數據計算 Beta (共變異數 / 市場變異數)
        # 需確保日期對齊
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        beta = covariance / market_variance # 系統性風險係數
        
        # 3. 計算預期報酬率 E(Ri) = Rf + Beta * MRP
        expected_return = RF + beta * MRP #這就是投資人要求的權益資金成本

        # --- B. Gordon 評價模型 (合理股價) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        dividend_yield = ticker_info.get('dividendYield', 0)
        dividend_rate = ticker_info.get('dividendRate', 0)
        
        fair_value = "N/A"
        undervalued_pct = 0
        
        # 只有當預期報酬率 > 成長率，Gordon 模型才有效
        if dividend_rate and dividend_rate > 0 and expected_return > G_GROWTH:
            # P = D / (K - g)
            theoretical_price = dividend_rate / (expected_return - G_GROWTH)
            fair_value = round(theoretical_price, 2)
            # 計算折價幅度 (正值代表被低估)
            undervalued_pct = (theoretical_price - close) / close

        # --- C. Fama-French 三因子準備 ---
        # SMB (規模): 市值
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = market_cap < 50000000000 # 假設 500億以下為中小型
        
        # HML (價值): 淨值市價比 (B/M) = 1 / PB
        pb_ratio = ticker_info.get('priceToBook', 0)
        is_value_stock = 0 < pb_ratio < 1.5 # 低 PB 代表價值型

        # --- D. MPT 風險 (標準差) ---
        volatility = stock_returns.std() * (252**0.5) # 年化波動率

        # --- E. 技術面 (Momentum) ---
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        bias_60 = (close - ma60) / ma60
        vol_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
        
        # --- 評分系統 (Weighted Score) ---
        score = 0
        factors = []
        
        # 1. 估值因子 (Gordon Model & Value)
        if isinstance(fair_value, float) and fair_value > close:
            score += 25
            factors.append(f"💰 低於理論價({fair_value})")
        
        if is_value_stock: # Fama-French HML
            score += 15
            factors.append(f"💎 價值股(PB {round(pb_ratio, 2)})")

        # 2. 規模因子 (SMB)
        if is_small_cap: # Fama-French SMB
            score += 10 # 根據統計，小型股有超額報酬
            factors.append("🔹 小型股溢酬")

        # 3. 獲利因子 (Quality)
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append(f"👑 高ROE({round(roe*100)}%)")

        # 4. 技術因子 (Momentum & Sniper)
        # 剛站上季線且乖離不大
        if close > ma60 and 0 < bias_60 < 0.10:
            score += 20
            factors.append("🎯 剛站上季線")
        
        # 5. 籌碼因子 (Volume)
        if volume > 1.5 * vol_ma5:
            score += 15
            factors.append("🔥 量能放大")

        # 6. 風險調整 (Risk Penalty)
        if volatility > 0.5: # 波動太大扣分
            score -= 10
            factors.append("⚠️ 高波動")

        # 門檻
        if score >= 60:
            return {
                "Ticker": ticker_symbol,
                "Name": name_map.get(ticker_symbol, ticker_symbol),
                "Close": round(close, 2),
                "Score": score,
                "Fair_Value": fair_value, # 合理股價
                "Beta": round(beta, 2), # 系統風險
                "Exp_Return": f"{round(expected_return*100, 1)}%", # 要求報酬
                "Factors": " | ".join(factors)
            }

    except:
        return None
    return None

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 投資理論實戰版 V7", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資理論實戰模型 V7")
st.markdown("""
### 📚 應用理論模型：
* **CAPM (資本資產定價)**：計算 Beta 與 預期報酬率 (資金成本)。
* **Gordon Model (股利折現)**：利用 CAPM 算出的成本，推導 **合理股價 (Fair Value)**。
* **Fama-French (三因子)**：加權 **小型股 (SMB)** 與 **價值股 (HML)**。
* **MPT (現代投資組合)**：監控波動率 ($\sigma$)，優化風險回報。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 系統將計算全台股的 Beta 值與理論價格，運算量較大，約需 20-25 分鐘。")
    
    if st.button("🚀 啟動理論模型掃描", type="primary"):
        with st.spinner("Step 1: 下載大盤指數計算 Beta 基準..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 讀取 twstock 字典..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"基準設定完成！開始分析 {len(tickers)} 檔股票的 CAPM 與估值...")
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
                
                if completed_count % 50 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"分析進度: {completed_count}/{len(tickers)} | 符合理論標的: {found_count}")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    df_realtime = df_realtime.sort_values(by='Score', ascending=False)
                    
                    with result_placeholder.container():
                        st.subheader(f"🎯 理論價值選股 ({found_count} 檔)")
                        # 顯示包含理論數值的表格
                        st.dataframe(
                            df_realtime[['Name', 'Ticker', 'Close', 'Fair_Value', 'Score', 'Exp_Return', 'Beta', 'Factors']], 
                            use_container_width=True,
                            hide_index=True
                        )

        status_text.text("✅ 全市場理論分析完成！")
        
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='Score', ascending=False)
            top_5 = df_final.head(5)
            msg = f"📊 **【Miniko 理論模型報告】**\n"
            for _, row in top_5.iterrows():
                msg += f"• {row['Name']} ({row['Ticker']}) 現價:{row['Close']} | 合理價:{row['Fair_Value']}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊左側按鈕，讓 AI 用華爾街模型幫您算股價！")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
