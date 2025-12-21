import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
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
    price = None
    try:
        # 1. 嘗試 yfinance (歷史數據最後一筆)
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    if price is None:
        try:
            # 2. 備用嘗試 twstock
            code = stock_code.split('.')[0]
            realtime = twstock.realtime.get(code)
            if realtime['success']:
                rt = realtime['realtime']
                p = rt.get('latest_trade_price', '-')
                if p == '-' or not p:
                    p = rt.get('best_bid_price', ['-'])[0]
                if p and p != '-':
                    price = float(p)
        except: pass
    return price

def calculate_single_stock(ticker_symbol, name_map):
    """計算單一股票因子 (極簡化版以求穩定)"""
    try:
        # 1. 抓價格
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price < 10: return None # 排除資料錯誤或雞蛋水餃

        # 2. 抓數據 (限制 1 年)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 60: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 技術面
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
        # 買點
        if current_price > ma20:
            buy_point = ma20
        else:
            buy_point = current_price * 0.98
            
        # 波動率與 CGO
        stock_returns = data['Close'].pct_change().dropna()
        volatility = stock_returns.std() * (252**0.5)
        
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        # 基本面估值 (Gordon)
        ke = RF + 1.0 * MRP
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            y = ticker_info.get('dividendYield', 0)
            if y: div_rate = current_price * y
            
        fair_value = np.nan
        if div_rate and div_rate > 0:
            fair_value = div_rate / max(ke - G_GROWTH, 0.015)

        # AI 評分
        score = 0
        factors = []
        
        # 價值
        if pb := ticker_info.get('priceToBook', 0):
            if 0 < pb < 1.5: 
                score += 15
                factors.append("💎低PB")
        
        if not np.isnan(fair_value) and fair_value > current_price * 1.1:
            score += 20
            factors.append("💰低估")
            
        # 籌碼與技術
        if cgo_val > 0.1:
            score += 15
            factors.append("🔥籌碼優")
            
        if current_price > ma20 and ma20 > ma60:
            score += 15
            factors.append("🐂多頭")
            
        if volatility < 0.3:
            score += 15
            factors.append("🛡️穩健")

        # 回傳資料
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": round(float(current_price), 2),
                "AI評分": score,
                "買入點": round(buy_point, 2),
                "合理價": round(fair_value, 2) if not np.isnan(fair_value) else None,
                "CGO": round(cgo_val * 100, 1),
                "亮點": " ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---
st.set_page_config(page_title="Miniko 穩定分流版", layout="wide")
st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 (穩定分流版)")

# --- 側邊欄控制 ---
with st.sidebar:
    st.header("⚙️ 掃描設定")
    
    # 預先載入清單以取得總數
    tickers_all, name_map = get_all_tw_tickers()
    total_count = len(tickers_all)
    
    st.write(f"全市場共 {total_count} 檔股票")
    
    # 關鍵：讓使用者選擇範圍，避免一次跑掛
    start_idx = st.number_input("起始順序", min_value=0, max_value=total_count, value=0, step=100)
    end_idx = st.number_input("結束順序", min_value=0, max_value=total_count, value=min(200, total_count), step=100)
    
    st.info(f"本次將掃描第 {start_idx} 到 {end_idx} 檔 (共 {end_idx - start_idx} 檔)")
    st.warning("建議每次掃描不超過 300 檔，以免雲端伺服器斷線。")
    
    run_btn = st.button("🚀 開始掃描選定範圍", type="primary")

# --- 主程式 ---
if run_btn:
    if end_idx <= start_idx:
        st.error("結束順序必須大於起始順序！")
    else:
        # 切割清單
        target_tickers = tickers_all[start_idx : end_idx]
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 執行緒數量設為 2，非常保守以求穩定
        MAX_WORKERS = 2 
        processed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_ticker = {
                executor.submit(calculate_single_stock, t, name_map): t 
                for t in target_tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res:
                    st.session_state['results'].append(res)
                
                processed_count += 1
                progress_bar.progress(processed_count / len(target_tickers))
                status_text.text(f"分析中: {processed_count}/{len(target_tickers)}")
                
                # 每 10 檔強制清理記憶體
                if processed_count % 10 == 0:
                    gc.collect()

        st.success("✅ 掃描完成！")

# --- 顯示結果 ---
if 'results' in st.session_state and st.session_state['results']:
    df = pd.DataFrame(st.session_state['results'])
    
    if not df.empty:
        df = df.sort_values(by=['AI評分', 'CGO'], ascending=[False, False])
        
        st.subheader(f"🏆 掃描結果 Top {min(100, len(df))}")
        st.dataframe(
            df.head(100),
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI評分", "買入點", "合理價", "CGO", "亮點"],
            column_config={
                "AI評分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
                "CGO": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
    else:
        st.warning("在此範圍內沒有找到符合條件的股票。")
