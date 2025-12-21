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
RF = 0.015  # 無風險利率 (1.5%)
MRP = 0.055 # 市場風險溢酬 (稍微調高至 5.5% 以拉大差異)
G_GROWTH = 0.02 # 長期成長率 (2%)

# --- 核心功能 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    """下載大盤指數 (TWII) - 改用 2 年數據以提升 Beta 精準度"""
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

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V7.1 精細運算版】
    改進：分數連續化、Beta樣本擴大、個別化呈現
    """
    try:
        # 1. 下載個股數據 (擴大到 2 年，讓 Beta 更獨特)
        data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if len(data) < 250: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        close = curr['Close']
        volume = curr['Volume']
        
        # 過濾殭屍股
        if volume < 100000 or close < 10: return None

        # --- A. CAPM 模型 (精細版) ---
        stock_returns = data['Close'].pct_change().dropna()
        
        # 對齊數據
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        # 確保有足夠的重疊交易日才計算
        if len(aligned_data) < 100: return None

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        
        # Beta 計算 (保留 4 位小數運算，最後再顯示 2 位)
        beta = covariance / market_variance
        
        # 預期報酬率 E(Ri)
        expected_return = RF + beta * MRP

        # --- B. Gordon 評價模型 ---
        ticker_info = yf.Ticker(ticker_symbol).info
        
        # 嘗試獲取更精確的股利數據
        dividend_rate = ticker_info.get('dividendRate', 0)
        if dividend_rate is None: dividend_rate = 0
        
        fair_value = np.nan
        
        # 如果股利 > 0 且 要求報酬率 > 成長率，才能算合理價
        # 為了避免分母過小導致價格無限大，設定分母最小值
        k_minus_g = max(expected_return - G_GROWTH, 0.01)
        
        if dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
        
        # --- C. 數據準備 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        peg = ticker_info.get('pegRatio', None)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        # --- D. 連續性評分系統 (Continuous Scoring) ---
        # 不再只是 +10 或 +20，而是根據強度給分
        
        score = 0.0 # 改用浮點數
        factors = []
        
        # 1. 價值分數 (Gordon 模型折價幅度)
        if not np.isnan(fair_value) and fair_value > close:
            upside = (fair_value - close) / close
            # 折價越多越高分，最高給 30 分
            val_score = min(upside * 100, 30)
            score += val_score
            factors.append(f"💰 折價{round(upside*100)}%")
        
        # 2. 成長分數 (營收成長率)
        if rev_growth and rev_growth > 0:
            # 成長 20% 得 20 分，最高 25 分
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈 高成長")

        # 3. 品質分數 (ROE)
        if roe and roe > 0:
            # ROE 15% 得 15 分，最高 20 分
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑 ROE{round(roe*100)}%")

        # 4. 價值分數 (PB Ratio)
        if 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎 低PB({round(pb_ratio, 1)})")
            
        # 5. 技術面微調 (剛站上季線)
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        bias = (close - ma60) / ma60
        if 0 < bias < 0.08: # 剛站上 0~8%
            score += 20
            factors.append("🎯 剛站上季線")
        elif bias > 0.2: # 漲太多扣分
            score -= 10
            
        # 6. Beta 調整 (風險調整)
        # 根據您的筆記：低 Beta (防守) 或 高 Beta (攻擊) 
        # 這裡我們假設偏好「波動不要太大」的穩健股
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: 
            score -= 15 # 波動太大扣分
        
        # 最終門檻
        if score >= 50:
            return {
                "Ticker": ticker_symbol,
                "Name": name_map.get(ticker_symbol, ticker_symbol),
                "Close": round(close, 2),
                "Score": round(score, 1), # 顯示小數點後一位
                "Fair_Value": fair_value if not np.isnan(fair_value) else "N/A",
                "Beta": round(beta, 3), # 顯示三位小數，區分差異
                "Exp_Return": f"{round(expected_return*100, 2)}%", # 顯示兩位小數
                "Factors": " | ".join(factors)
            }

    except:
        return None
    return None

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 理論實戰 V7.1", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資理論實戰模型 V7.1 (精細版)")
st.markdown("""
### 🚀 V7.1 更新特點：
* **個別化 Beta**：採用 2 年數據運算，精準區分每檔股票的風險係數，不再出現重複數值。
* **連續性評分**：分數不再是整數，而是根據 ROE 與成長率的強弱給予 **精確小數點評分** (例如 82.5 分)。
* **動態估值**：Gordon 模型參數優化，呈現每檔股票獨特的合理價。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 因運算精度提高，分析約需 20 分鐘。請耐心等待，結果將具備高度個別化特徵。")
    
    if st.button("🚀 啟動精細運算", type="primary"):
        with st.spinner("Step 1: 下載大盤 2 年數據建立 CAPM 基準..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"基準建立完成！開始為 {len(tickers)} 檔股票進行個別化定價...")
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
                    status_text.text(f"分析進度: {completed_count}/{len(tickers)} | 價值發現: {found_count}")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    # 按照分數排序
                    df_realtime = df_realtime.sort_values(by='Score', ascending=False)
                    
                    with result_placeholder.container():
                        st.subheader(f"🎯 個別化理論選股 ({found_count} 檔)")
                        st.dataframe(
                            df_realtime[['Name', 'Ticker', 'Close', 'Fair_Value', 'Score', 'Exp_Return', 'Beta', 'Factors']], 
                            use_container_width=True,
                            hide_index=True
                        )

        status_text.text("✅ 精細分析完成！")
        
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='Score', ascending=False)
            top_5 = df_final.head(5)
            msg = f"📊 **【Miniko V7.1 精選】**\n"
            for _, row in top_5.iterrows():
                msg += f"• {row['Name']} ({row['Ticker']}) 分數:{row['Score']} | 合理價:{row['Fair_Value']}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊左側按鈕，觀看個別化的股票估值運算結果。")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
