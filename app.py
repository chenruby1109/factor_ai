import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures

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

@st.cache_data(ttl=3600) # 快取 1 小時，不用每次都重抓
def get_all_tw_tickers():
    """
    從證交所與櫃買中心獲取所有上市櫃股票代號
    """
    ticker_list = []
    
    try:
        # 1. 上市股票 (Mode=2)
        url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        res_twse = pd.read_html(url_twse)[0]
        # 整理資料：只留股票代號，並加上 .TW
        res_twse.columns = res_twse.iloc[0]
        res_twse = res_twse.iloc[1:]
        res_twse = res_twse[res_twse['有價證券別'] == '股票']
        # 格式化代號
        tickers_twse = res_twse['有價證券代號及名稱'].apply(lambda x: x.split()[0] + ".TW").tolist()
        names_twse = res_twse['有價證券代號及名稱'].apply(lambda x: x.split()[0] + " " + x.split()[-1]).tolist()
        
        # 2. 上櫃股票 (Mode=4)
        url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
        res_tpex = pd.read_html(url_tpex)[0]
        res_tpex.columns = res_tpex.iloc[0]
        res_tpex = res_tpex.iloc[1:]
        res_tpex = res_tpex[res_tpex['有價證券別'] == '股票']
        # 格式化代號
        tickers_tpex = res_tpex['有價證券代號及名稱'].apply(lambda x: x.split()[0] + ".TWO").tolist()
        names_tpex = res_tpex['有價證券代號及名稱'].apply(lambda x: x.split()[0] + " " + x.split()[-1]).tolist()

        # 合併與建立對照表
        all_tickers = tickers_twse + tickers_tpex
        all_names = names_twse + names_tpex
        
        # 建立 {代號: 中文名} 的字典
        name_map = {}
        for item in all_names:
            code, name = item.split()
            # 判斷是上市還是上櫃來決定後綴
            suffix = ".TW" if code + ".TW" in tickers_twse else ".TWO"
            name_map[code + suffix] = name
            
        return all_tickers, name_map
        
    except Exception as e:
        st.error(f"無法自動抓取股票清單: {e}")
        return [], {}

def calculate_factors_sniper(ticker_symbol, name_map):
    """
    Miniko 狙擊手 V5 - 嚴格篩選邏輯
    """
    try:
        # 只抓最近 3 個月資料以節省時間
        data = yf.download(ticker_symbol, period="3mo", interval="1d", progress=False)
        
        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        prev = data.iloc[-2]
        close = curr['Close']
        
        # 0. 基本過濾：排除雞蛋水餃股 (股價 < 10 元 或 成交量 < 200 張)
        if close < 10 or curr['Volume'] < 200000: return None

        # 1. 技術指標
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        vol_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
        
        # 乖離率 (Bias)
        bias = (close - ma60) / ma60
        
        # RSI
        rsi = ta.momentum.rsi(data['Close'], window=14).iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        macd_diff = macd.macd_diff().iloc[-1]
        macd_diff_prev = macd.macd_diff().iloc[-2]

        # --- 狙擊手邏輯 ---
        score = 0
        factors = []
        
        # 條件 A: 拒絕追高 (乖離率不能太大)
        # 第一浪通常剛突破，乖離率應該在 0% ~ 15% 之間
        if bias > 0.20: return None # 漲太多了，Pass
        if bias < -0.10: return None # 還在空頭，Pass

        # 條件 B: 剛站上季線 (生命線)
        if close > ma60:
            score += 30
            factors.append("🎯 站上季線")
        
        # 條件 C: 底部爆量吸籌 (最關鍵！)
        # 成交量放大 1.5 倍，但股價沒噴出 (>3% <8%) 或 穩健上漲
        price_chg = (close - prev['Close']) / prev['Close']
        vol_ratio = curr['Volume'] / vol_ma5
        
        if vol_ratio > 1.5:
            score += 25
            factors.append(f"🔥 量能放大({round(vol_ratio,1)}倍)")
        
        # 條件 D: MACD 轉折
        if macd_diff > 0 and macd_diff_prev <= 0:
            score += 20
            factors.append("⚡ MACD翻紅")
            
        # 條件 E: RSI 健康區
        if 50 < rsi < 70:
            score += 15
        
        # 門檻：至少要 60 分才回傳
        if score >= 60:
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

st.set_page_config(page_title="Miniko 全市場狙擊 V5", layout="wide")

st.title("🏹 Miniko 狙擊手 V5 - 全市場地毯式搜查")
st.markdown("### 策略：掃描全台 1800+ 檔股票，尋找「剛站上季線 + 爆量」的起漲股")

# 初始化 session state 用來存結果
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 提醒：掃描全市場約需 15~20 分鐘，請耐心等待。結果會即時出現在右側。")
    
    if st.button("🚀 啟動全市場掃描", type="primary"):
        # 1. 抓股票清單
        with st.spinner("正在下載全台股清單 (上市+上櫃)..."):
            tickers, name_map = get_all_tw_tickers()
            
        if not tickers:
            st.error("無法取得股票清單，請檢查網路或 requirements.txt")
        else:
            st.success(f"成功取得 {len(tickers)} 檔股票，開始分析...")
            st.session_state['results'] = [] # 清空舊資料
            
            # 進度條
            progress_bar = st.progress(0)
            status_text = st.empty()
            result_placeholder = col2.empty() # 用來即時更新表格
            
            # 使用多執行緒加速 (小心 Yahoo 封鎖，設 4-8 即可)
            # 為了讓結果即時顯示，我們不用 executor.map，改用 as_completed
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                future_to_ticker = {executor.submit(calculate_factors_sniper, t, name_map): t for t in tickers}
                
                completed_count = 0
                found_count = 0
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    data = future.result()
                    completed_count += 1
                    
                    # 更新進度
                    if completed_count % 10 == 0:
                        progress_bar.progress(completed_count / len(tickers))
                        status_text.text(f"已掃描: {completed_count}/{len(tickers)} | 發現潛力股: {found_count} 檔")
                    
                    # 如果有找到好股票
                    if data:
                        found_count += 1
                        st.session_state['results'].append(data)
                        
                        # 即時更新右側表格 (每發現一檔就更新一次)
                        df_realtime = pd.DataFrame(st.session_state['results'])
                        # 排序
                        df_realtime = df_realtime.sort_values(by='Score', ascending=False)
                        
                        with result_placeholder.container():
                            st.subheader(f"🎯 發現目標 ({found_count} 檔)")
                            st.dataframe(
                                df_realtime[['Name', 'Ticker', 'Close', 'Score', 'Bias', 'Factors']], 
                                use_container_width=True,
                                hide_index=True
                            )

            status_text.text("✅ 全市場掃描完成！")
            
            # 掃描結束後，如果有 Telegram 設定，發送前三名
            if st.session_state['results']:
                df_final = pd.DataFrame(st.session_state['results']).sort_values(by='Score', ascending=False)
                top_3 = df_final.head(3)
                msg = f"🏹 **【Miniko 全市場掃描完成】**\n共發現 {len(df_final)} 檔潛力股，前三名：\n"
                for _, row in top_3.iterrows():
                    msg += f"• {row['Name']} ({row['Ticker']}) ${row['Close']}\n"
                send_telegram_message(msg)

with col2:
    # 初始畫面或結果顯示區
    if not st.session_state['results']:
        st.write("👈 點擊左側按鈕開始掃描，搜尋結果將會在此即時顯示...")
    else:
        # 這裡保留是為了 refresh 後還能看到資料
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 歷史掃描結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
