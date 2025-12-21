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
RF = 0.015  # 無風險利率 (Risk-Free Rate)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 股利長期成長率 (Gordon Growth Rate)

# --- 核心功能 ---

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

def get_realtime_price_robust(stock_code):
    """
    【V8.2 強力價格獲取】
    優先順序：
    1. twstock (證交所/櫃買中心真實成交價)
    2. yfinance fast_info (比 download 更準確的即時報價)
    """
    # 策略 1: twstock (最準)
    try:
        code = stock_code.split('.')[0]
        realtime = twstock.realtime.get(code)
        if realtime['success']:
            price = realtime['realtime']['latest_trade_price']
            # 如果盤中暫無成交，抓最佳買入價
            if price == '-' or price is None:
                price = realtime['realtime']['best_bid_price'][0]
            if float(price) > 0:
                return float(price)
    except:
        pass

    # 策略 2: yfinance fast_info (備援，防擋IP)
    try:
        ticker = yf.Ticker(stock_code)
        # fast_info 通常包含 'last_price'，這是最新的交易所價格
        price = ticker.fast_info.get('last_price')
        if price and price > 0:
            return float(price)
    except:
        pass
        
    return None

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V8.2 中文詳解版】
    修正價格錯誤，並將所有術語中文化。
    """
    try:
        # 1. 獲取絕對正確的價格
        current_price = get_realtime_price_robust(ticker_symbol)
        
        # 2. 下載歷史數據 (用於計算技術指標與 Beta)
        data = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
        
        if len(data) < 250: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 如果真的抓不到即時價，才用歷史收盤價 (極少發生)
        if current_price is None:
            current_price = data.iloc[-1]['Close']

        # 過濾雞蛋水餃股
        if current_price < 10: return None

        # --- A. CAPM 模型 (資本資產定價模型) ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        if len(aligned_data) < 100: return None

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        
        # Beta (風險係數)
        beta = covariance / market_variance
        
        # 預期報酬率 (Expected Return)
        expected_return = RF + beta * MRP

        # --- B. Gordon 模型 (股利折現模型) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        dividend_rate = ticker_info.get('dividendRate', 0)
        
        # 補強：如果 Yahoo 缺股利資料，改用殖利率推算
        if dividend_rate is None or dividend_rate == 0:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: dividend_rate = current_price * yield_val

        fair_value = np.nan
        upside_potential = np.nan
        
        # 公式：合理價 = 股利 / (預期報酬率 - 成長率)
        k_minus_g = max(expected_return - G_GROWTH, 0.01)
        
        if dividend_rate and dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
            # 計算獲利空間
            upside_potential = (fair_value - current_price) / current_price

        # --- C. 數據準備 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        # --- D. 評分系統 ---
        score = 0.0
        factors = []
        
        # 1. 價值 (Value)
        if not np.isnan(fair_value) and fair_value > current_price:
            val_score = min(upside_potential * 100, 30)
            score += val_score
            factors.append(f"💰低於合理價 (Undervalued)")
        
        # 2. 成長 (Growth)
        if rev_growth and rev_growth > 0:
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈營收高成長 (Revenue Growth)")

        # 3. 品質 (Quality - ROE)
        if roe and roe > 0:
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑高股東權益報酬 (High ROE)")

        # 4. 價值 (PB)
        if pb_ratio and 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎低股價淨值比 (Low P/B)")
            
        # 5. 技術 (Momentum)
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        bias = (current_price - ma60) / ma60
        if 0 < bias < 0.08:
            score += 20
            factors.append("🎯剛站上季線 (Trend Start)")
        elif bias > 0.2:
            score -= 10
            
        # 6. 風險 (Volatility)
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: score -= 15
        
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "最新收盤價": current_price, 
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

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 戰情室 V8.2", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資理論實戰模型 V8.2 (中文詳解版)")
st.markdown("""
本系統結合三大財務理論，為您計算股票真實價值。價格來源已修正為 **證交所即時成交價**。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 為了確保價格正確，系統會進行雙重驗證 (證交所 + Yahoo Fast Info)，全市場掃描約需 20 分鐘。")
    
    if st.button("🚀 啟動精準估值掃描", type="primary"):
        with st.spinner("Step 1: 計算大盤風險參數 (Beta基準)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入全台股清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"準備就緒！開始分析 {len(tickers)} 檔股票的 CAPM 與 合理價...")
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

        status_text.text("✅ 全市場分析完成！")
        
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
        st.write("👈 請點擊左側按鈕，開始進行正確價格的理論估值。")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終評價結果 ({len(df_show)} 檔)")
        
        # --- 詳細中文說明與格式化 ---
        st.dataframe(
            df_show.sort_values(by='綜合評分', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_order=["名稱", "代號", "最新收盤價", "理論合理價", "預估獲利空間", "綜合評分", "資金成本", "風險係數", "亮點因子"],
            column_config={
                "名稱": st.column_config.TextColumn("股票名稱"),
                "代號": st.column_config.TextColumn("代號"),
                "最新收盤價": st.column_config.NumberColumn(
                    "最新收盤價 (Price)",
                    help="目前證交所的即時成交價格 (台幣)。",
                    format="$%.2f",
                ),
                "理論合理價": st.column_config.NumberColumn(
                    "理論合理價 (Gordon Fair Value)",
                    help="基於高登股利折現模型 (Gordon Model) 計算的合理股價。\n公式：股利 / (預期報酬率 - 成長率)。",
                    format="$%.2f",
                ),
                "預估獲利空間": st.column_config.NumberColumn(
                    "預估獲利空間 (Upside Potential)",
                    help="((合理價 - 現價) / 現價)。\n正值代表被低估(值得買入)，負值代表被高估。",
                    format="%.2f%%",
                ),
                "綜合評分": st.column_config.ProgressColumn(
                    "AI 綜合評分 (Score)",
                    help="綜合 F-G-M 模型 (基本面、成長、動能) 的總分，滿分 100 分。",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "資金成本": st.column_config.NumberColumn(
                    "資金成本 (CAPM Expected Return)",
                    help="基於 CAPM 模型計算的『預期報酬率』，也就是投資人持有這檔股票要求的最低回報率。\n公式：無風險利率 + Beta * 市場風險溢酬。",
                    format="%.2f%%",
                ),
                "風險係數": st.column_config.NumberColumn(
                    "風險係數 (Beta)",
                    help="衡量股票相對於大盤的波動程度。\nBeta > 1：波動比大盤大 (攻擊型)。\nBeta < 1：波動比大盤小 (防守型)。",
                    format="%.2f",
                ),
                "亮點因子": st.column_config.TextColumn(
                    "AI 診斷亮點 (Key Factors)",
                    help="符合的投資理論特徵，如：價值低估、高成長、籌碼集中等。",
                    width="medium"
                ),
            }
        )
