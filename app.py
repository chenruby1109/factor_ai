import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import twstock
import gc
import time

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 ---
RF = 0.015
MRP = 0.055
G_GROWTH = 0.02

# --- 核心功能函數 ---

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
        for code, info in twstock.codes.items():
            if info.type == '股票':
                suffix = ".TW" if info.market == '上市' else ".TWO"
                full_ticker = code + suffix
                tickers.append(full_ticker)
                name_map[full_ticker] = info.name
        return tickers, name_map
    except:
        return [], {}

def get_realtime_price_robust(stock_code):
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
                rt = realtime['realtime']
                price = float(rt.get('latest_trade_price', 0) or rt.get('best_bid_price', [0])[0])
    # 這裡如果不成功就回傳 None，不勉強
        except: pass
    return price

def calculate_single_stock(ticker_symbol, name_map, market_returns):
    """計算單一股票因子 (最輕量化版)"""
    try:
        # 1. 抓資料 (只抓必要長度)
        data = yf.download(ticker_symbol, period="6mo", interval="1d", progress=False)
        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        current_price = float(data['Close'].iloc[-1])
        if current_price < 10: return None # 排除雞蛋水餃

        # 2. 技術指標
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
        # 買點建議
        if current_price > ma20:
            buy_point = ma20
        else:
            buy_point = current_price * 0.98

        # 3. 波動率 & Beta
        rets = data['Close'].pct_change().dropna()
        volatility = rets.std() * (252**0.5)
        
        # 簡化 Beta 計算 (避免太複雜的 covariance 矩陣運算吃記憶體)
        # 這裡用簡易判斷代替繁重計算，或假設 Beta=1 以節省資源
        beta = 1.0 
        ke = RF + beta * MRP

        # 4. Gordon 合理價
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        # 若抓不到，嘗試用最後股價 * 殖利率估算
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        if div_rate and div_rate > 0:
            k_minus_g = max(ke - G_GROWTH, 0.015)
            fair_value = div_rate / k_minus_g

        # 5. AI 評分 (邏輯簡化以加速)
        score = 0
        factors = []
        
        # 價值
        if fair_value and fair_value > current_price * 1.1:
            score += 25
            factors.append("💰低估")
        
        # 趨勢
        if current_price > ma20 and ma20 > ma60:
            score += 25
            factors.append("🐂多頭")
            
        # 籌碼 (CGO概念: 現價 > 季線成本)
        if current_price > ma60:
            score += 25
            factors.append("🔥籌碼優")
            
        # 穩定度
        if volatility < 0.3:
            score += 25
            factors.append("🛡️穩健")
        elif volatility > 0.6:
            score -= 10
            
        return {
            "代號": ticker_symbol,
            "名稱": name_map.get(ticker_symbol, ticker_symbol),
            "現價": round(current_price, 2),
            "AI評分": score,
            "買入點": round(buy_point, 2),
            "合理價": round(fair_value, 2) if not np.isnan(fair_value) else None,
            "亮點": " ".join(factors)
        }

    except:
        return None

# --- Streamlit 介面 ---
st.set_page_config(page_title="Miniko 輕量戰情室", layout="wide")
st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 (雲端輕量版)")

with st.sidebar:
    st.header("⚙️ 設定")
    test_mode = st.checkbox("開啟快速測試模式 (只跑前 30 檔)", value=True, help="建議先勾選此項，確認程式能跑，避免雲端記憶體不足。")
    run_btn = st.button("🚀 開始掃描", type="primary")

st.info("💡 提示：此版本為「單執行緒穩定版」，速度較慢但不容易當機。建議先用測試模式跑一次。")

if run_btn:
    st.session_state['results'] = []
    
    with st.spinner("準備資料中..."):
        market_returns = get_market_data()
        tickers, name_map = get_all_tw_tickers()
    
    # 測試模式限制數量
    if test_mode:
        tickers = tickers[:30]
        st.warning("⚠️ 目前為測試模式，僅分析前 30 檔股票。")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 單執行緒迴圈 (最穩定)
    results = []
    for i, ticker in enumerate(tickers):
        # 顯示進度
        status_text.text(f"正在分析 ({i+1}/{len(tickers)}): {ticker} - {name_map.get(ticker, '')}")
        progress_bar.progress((i + 1) / len(tickers))
        
        # 計算
        res = calculate_single_stock(ticker, name_map, market_returns)
        if res and res['AI評分'] >= 50: # 只存及格的，節省記憶體
            results.append(res)
            
        # 每 10 檔強制清理記憶體
        if i % 10 == 0:
            gc.collect()
            
    st.session_state['results'] = results
    status_text.text("✅ 分析完成！")

# 顯示結果
if 'results' in st.session_state and st.session_state['results']:
    df = pd.DataFrame(st.session_state['results'])
    if not df.empty:
        df = df.sort_values(by="AI評分", ascending=False).head(100)
        st.subheader(f"🏆 AI 評分 Top {len(df)} (現貨推薦)")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("沒有符合條件的股票 (或資料抓取失敗)。")
