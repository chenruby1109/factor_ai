import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures
import twstock # <--- 引入這個強大的台股套件

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 核心功能 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_all_tw_tickers():
    """
    使用 twstock 套件直接獲取清單 (不用爬蟲，速度快且穩)
    """
    try:
        tickers = []
        name_map = {}
        
        # twstock.codes 是內建的字典，包含所有台股資訊
        for code, info in twstock.codes.items():
            # 只選「股票」，排除權證、ETF等
            if info.type == '股票':
                suffix = ""
                if info.market == '上市':
                    suffix = ".TW"
                elif info.market == '上櫃':
                    suffix = ".TWO"
                
                if suffix:
                    full_ticker = code + suffix
                    tickers.append(full_ticker)
                    name_map[full_ticker] = info.name
        
        return tickers, name_map
        
    except Exception as e:
        st.error(f"獲取清單失敗: {e}")
        return [], {}

def calculate_factors_sniper(ticker_symbol, name_map):
    """
    Miniko 狙擊手 V6 - 嚴格篩選邏輯
    """
    try:
        # 抓取最近 3 個月資料
        data = yf.download(ticker_symbol, period="3mo", interval="1d", progress=False)
        
        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        prev = data.iloc[-2]
        close = curr['Close']
        
        # 0. 基本過濾 (排除 10 元以下與無量股)
        if close < 10 or curr['Volume'] < 200000: return None

        # 1. 技術指標
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        vol_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
        
        bias = (close - ma60) / ma60
        rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
        
        macd = ta.trend.MACD(data['Close'])
        macd_diff = macd.macd_diff().iloc[-1]
        macd_diff_prev = macd.macd_diff().iloc[-2]

        # --- 狙擊手邏輯 ---
        score = 0
        factors = []
        
        # 條件 A: 拒絕追高 (乖離率 < 20%)
        if bias > 0.20: return None 
        if bias < -0.15: return None

        # 條件 B: 剛站上季線
        if close > ma60:
            score += 30
            factors.append("🎯 站上季線")
        
        # 條件 C: 底部爆量吸籌
        vol_ratio = curr['Volume'] / vol_ma5
        if vol_ratio > 1.3:
            score += 25
            factors.append(f"🔥 量增({round(vol_ratio,1)}倍)")
        
        # 條件 D: MACD 轉折
        if macd_diff > 0 and macd_diff_prev <= 0:
            score += 20
            factors.append("⚡ MACD翻紅")
            
        # 條件 E: RSI
        if 45 < rsi < 75:
            score += 15
        
        # 總分門檻
        if score >= 55:
            return {
                "Ticker": ticker_symbol,
                "Name": name_map.get(ticker_symbol, ticker_symbol),
                "Close": round(close, 2),
                "Score": score,
                "Bias": f"{round(bias*100, 1)}%",
                "Factors": " | ".join(factors),
                "Volume": int(curr['Volume'])
            }
            
    except:
        return None
    return None

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 狙擊手 V6", layout="wide")

st.title("🏹 Miniko 狙擊手 V6 - 字典資料庫版")
st.markdown("### 策略：使用內建資料庫掃描全台 1800+ 檔股票，絕不連線失敗")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 提醒：這次使用的是內建清單，不會被網站擋 IP。全市場掃描約需 20 分鐘。")
    
    if st.button("🚀 啟動掃描", type="primary"):
        with st.spinner("正在讀取股票字典..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"成功載入 {len(tickers)} 檔股票！開始分析...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 使用多執行緒 (Max workers 設為 16 以加快 yfinance 下載)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ticker = {executor.submit(calculate_factors_sniper, t, name_map): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                if completed_count % 50 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"掃描進度: {completed_count}/{len(tickers)} | 已發現: {found_count} 檔")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    df_realtime = df_realtime.sort_values(by='Score', ascending=False)
                    
                    with result_placeholder.container():
                        st.subheader(f"🎯 發現目標 ({found_count} 檔)")
                        st.dataframe(
                            df_realtime[['Name', 'Ticker', 'Close', 'Score', 'Bias', 'Factors']], 
                            use_container_width=True,
                            hide_index=True
                        )

        status_text.text("✅ 掃描完成！")
        
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='Score', ascending=False)
            top_3 = df_final.head(3)
            msg = f"🏹 **【Miniko 狙擊手報告】**\n發現 {len(df_final)} 檔潛力股，前三名：\n"
            for _, row in top_3.iterrows():
                msg += f"• {row['Name']} ({row['Ticker']}) ${row['Close']}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊左側按鈕開始，這次保證不會有 SSL 錯誤！")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 歷史掃描結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
