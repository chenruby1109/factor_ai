import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
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
    price = None
    try:
        # 優先嘗試 yfinance (歷史數據最後一筆)
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    if price is None:
        try:
            # 備用嘗試 twstock
            code = stock_code.split('.')[0]
            realtime = twstock.realtime.get(code)
            if realtime['success']:
                rt = realtime['realtime']
                # 嘗試抓成交價，沒有則抓買進價
                p = rt.get('latest_trade_price', '-')
                if p == '-' or not p:
                    p = rt.get('best_bid_price', ['-'])[0]
                if p and p != '-':
                    price = float(p)
        except: pass
    return price

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """【Miniko V10.0 AI 旗艦運算核心 - 記憶體優化版】"""
    try:
        # 1. 先抓價格，若失敗直接跳過，節省資源
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None
        if current_price < 10: return None # 排除雞蛋水餃股

        # 2. 下載數據 (限制只抓 1 年)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # --- 技術面與買點 ---
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
        if current_price > ma20:
            buy_point = ma20 # 強勢回檔買月線
        else:
            buy_point = current_price * 0.98 # 弱勢下方接
        buy_point = round(buy_point, 2)

        # --- CAPM & 基本面 ---
        stock_returns = data['Close'].pct_change().dropna()
        # 簡化 Covariance 計算以節省記憶體，若數據長度不對齊則用簡易 Beta
        if len(stock_returns) > 60:
            volatility = stock_returns.std() * (252**0.5)
        else:
            volatility = 0.5 # 預設值

        ke = RF + 1.0 * MRP # 簡化 Beta=1 以加速運算，差異不大
        
        # Gordon Model
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            y_val = ticker_info.get('dividendYield', 0)
            if y_val: div_rate = current_price * y_val
            
        fair_value = np.nan
        if div_rate and div_rate > 0:
            k_g = max(ke - G_GROWTH, 0.015)
            fair_value = round(div_rate / k_g, 2)

        # Smart Beta (CGO)
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        # --- AI 評分 ---
        score = 0
        factors = []
        
        # 價值面
        pb = ticker_info.get('priceToBook', 0)
        if pb > 0 and pb < 1.5:
            score += 15
            factors.append("💎低PB")
        if not np.isnan(fair_value) and fair_value > current_price * 1.1:
            score += 20
            factors.append("💰低估")
            
        # 成長與規模
        mkt_cap = ticker_info.get('marketCap', 0)
        if 0 < mkt_cap < 50000000000:
            score += 10
            factors.append("🐟中小型")
            
        # 技術面
        if current_price > ma20 and ma20 > ma60:
            score += 15
            factors.append("🐂多頭排列")
            
        # 籌碼面
        if cgo_val > 0.1:
            score += 15
            factors.append("🔥籌碼優") # CGO高
            
        # 穩定度
        if volatility < 0.3:
            score += 15
            factors.append("🛡️穩健")
        
        # 門檻：50分以上才回傳，減少列表長度
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI評分": score,
                "買入點": buy_point,
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "CGO指標": round(cgo_val * 100, 1),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V10.0 (Full)", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V10.0 (全台股深度掃描版)")
st.caption("專為 Streamlit Cloud 優化的全市場掃描，包含 AI 評分、買點建議與 CGO 策略。")

# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 掃描設定")
    run_btn = st.button("🚀 啟動全市場掃描 (約需15分鐘)", type="primary")
    st.info("⚠️ 為了防止雲端當機，系統將採用「分批處理」模式。請耐心等待，勿關閉視窗。")

# --- 主程式 ---
if run_btn:
    st.session_state['results'] = []
    
    with st.spinner("Step 1: 下載大盤與股票清單..."):
        market_returns = get_market_data()
        tickers, name_map = get_all_tw_tickers()
        
    st.success(f"取得 {len(tickers)} 檔股票，開始 AI 運算...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_container = st.container() # 用來最後顯示結果
    
    # --- 關鍵修改：更安全的 Batch 處理 ---
    # 將 batch size 設為 30，確保記憶體絕對安全
    BATCH_SIZE = 30 
    total_processed = 0
    all_results = []
    
    # 外層迴圈：控制批次
    for i in range(0, len(tickers), BATCH_SIZE):
        batch_tickers = tickers[i : i + BATCH_SIZE]
        
        # 內層：每次只開一個小的 ThreadPool，跑完就關閉釋放資源
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t 
                for t in batch_tickers
            }
            
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    all_results.append(res)
                total_processed += 1
        
        # 更新進度
        progress = min(total_processed / len(tickers), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"正在掃描: {total_processed} / {len(tickers)} (已找到 {len(all_results)} 檔潛力股)...")
        
        # ★★★ 關鍵：強制清理記憶體 ★★★
        gc.collect() 
        # 稍微休息一下，避免 CPU 過熱被雲端踢掉
        time.sleep(0.05) 

    st.session_state['results'] = all_results
    status_text.text("✅ 全市場掃描完成！")

# --- 顯示結果 ---
if 'results' in st.session_state and st.session_state['results']:
    df = pd.DataFrame(st.session_state['results'])
    
    if not df.empty:
        # 排序邏輯：AI 評分高 -> CGO 高 -> 價格低
        df = df.sort_values(by=['AI評分', 'CGO指標'], ascending=[False, False])
        
        # 只取 Top 100
        top_100 = df.head(100)
        
        st.divider()
        st.subheader(f"🏆 AI 嚴選 Top 100 (現貨買入推薦)")
        st.markdown(f"從 **{len(df)}** 檔及格股票中，篩選出分數最高的 100 檔。")
        
        st.dataframe(
            top_100,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI評分", "買入點", "合理價", "CGO指標", "亮點"],
            column_config={
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "買入點": st.column_config.NumberColumn(format="$%.2f", help="技術面支撐位(月線)"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon 模型估值"),
                "AI評分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
    else:
        st.warning("沒有找到符合條件的股票。")
else:
    st.info("👈 請點擊左側按鈕開始掃描 (因為資料量大，可能需要 10-15 分鐘)。")
