import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures
import ssl

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

# --- 備用方案：內建熱門股清單 (萬一爬蟲失敗時使用) ---
def get_fallback_tickers():
    # 這裡預先列出市值前 150 大與熱門題材股
    STOCK_MAP = {
        '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2308.TW': '台達電', 
        '2382.TW': '廣達', '2412.TW': '中華電', '2881.TW': '富邦金', '2882.TW': '國泰金', 
        '2886.TW': '兆豐金', '2891.TW': '中信金', '1216.TW': '統一', '1301.TW': '台塑', 
        '1303.TW': '南亞', '1326.TW': '台化', '2002.TW': '中鋼', '2207.TW': '和泰車', 
        '2303.TW': '聯電', '2327.TW': '國巨', '2357.TW': '華碩', '2379.TW': '瑞昱', 
        '2395.TW': '研華', '2408.TW': '南亞科', '2603.TW': '長榮', '2609.TW': '陽明', 
        '2615.TW': '萬海', '2880.TW': '華南金', '2883.TW': '開發金', '2884.TW': '玉山金', 
        '2885.TW': '元大金', '2890.TW': '永豐金', '2892.TW': '第一金', '2912.TW': '統一超', 
        '3008.TW': '大立光', '3034.TW': '聯詠', '3037.TW': '欣興', '3045.TW': '台灣大', 
        '3231.TW': '緯創', '3443.TW': '創意', '3661.TW': '世芯-KY', '3711.TW': '日月光', 
        '4904.TW': '遠傳', '4938.TW': '和碩', '5871.TW': '中租-KY', '5876.TW': '上海商銀', 
        '5880.TW': '合庫金', '6415.TW': '矽力-KY', '6505.TW': '台塑化', '6669.TW': '緯穎', 
        '8046.TW': '南電', '9910.TW': '豐泰', '8299.TW': '群聯', '4927.TW': '泰鼎-KY',
        '3035.TW': '智原', '3529.TW': '力旺', '2360.TW': '致茂', '6278.TW': '台表科',
        '2356.TW': '英業達', '2376.TW': '技嘉', '2388.TW': '威盛', '2455.TW': '全新', 
        '3105.TW': '穩懋', '8086.TW': '宏捷科', '6213.TW': '聯茂', '3017.TW': '奇鋐',
        '3324.TW': '雙鴻', '1513.TW': '中興電', '1519.TW': '華城', '1503.TW': '士電',
        '1605.TW': '華新', '9958.TW': '世紀鋼', '6488.TW': '環球晶', '5483.TW': '中美晶',
        '6147.TW': '頎邦', '8069.TW': '元太', '5347.TW': '世界'
    }
    return list(STOCK_MAP.keys()), STOCK_MAP

@st.cache_data(ttl=3600) 
def get_all_tw_tickers():
    """
    獲取股票清單 (V5.2 強力版)
    先嘗試爬蟲，失敗則切換到內建清單
    """
    try:
        # 使用 requests 強制忽略 SSL 驗證
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        # 1. 上市
        url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        r_twse = requests.get(url_twse, headers=headers, verify=False, timeout=10) # 關鍵: verify=False
        df_twse = pd.read_html(r_twse.text)[0]
        
        df_twse.columns = df_twse.iloc[0]
        df_twse = df_twse.iloc[1:]
        df_twse = df_twse[df_twse['有價證券別'] == '股票']
        tickers_twse = df_twse['有價證券代號及名稱'].apply(lambda x: x.split()[0] + ".TW").tolist()
        names_twse = df_twse['有價證券代號及名稱'].apply(lambda x: x.split()[0] + " " + x.split()[-1]).tolist()
        
        # 2. 上櫃
        url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
        r_tpex = requests.get(url_tpex, headers=headers, verify=False, timeout=10)
        df_tpex = pd.read_html(r_tpex.text)[0]
        
        df_tpex.columns = df_tpex.iloc[0]
        df_tpex = df_tpex.iloc[1:]
        df_tpex = df_tpex[df_tpex['有價證券別'] == '股票']
        tickers_tpex = df_tpex['有價證券代號及名稱'].apply(lambda x: x.split()[0] + ".TWO").tolist()
        names_tpex = df_tpex['有價證券代號及名稱'].apply(lambda x: x.split()[0] + " " + x.split()[-1]).tolist()

        all_tickers = tickers_twse + tickers_tpex
        all_names = names_twse + names_tpex
        
        name_map = {}
        for item in all_names:
            code, name = item.split()
            suffix = ".TW" if code + ".TW" in tickers_twse else ".TWO"
            name_map[code + suffix] = name
            
        return all_tickers, name_map
        
    except Exception as e:
        st.warning(f"自動抓取全市場清單失敗 (SSL 阻擋)，已自動切換至「精選熱門股模式」繼續執行。")
        return get_fallback_tickers()

def calculate_factors_sniper(ticker_symbol, name_map):
    """
    Miniko 狙擊手 V5.2 - 嚴格篩選邏輯
    """
    try:
        # 只抓最近 3 個月資料
        data = yf.download(ticker_symbol, period="3mo", interval="1d", progress=False)
        
        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        prev = data.iloc[-2]
        close = curr['Close']
        
        # 0. 基本過濾
        if close < 10 or curr['Volume'] < 100000: return None # 稍微放寬成交量

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
        
        # 條件 A: 拒絕追高
        if bias > 0.20: return None 
        if bias < -0.15: return None # 放寬一點空頭容忍度

        # 條件 B: 剛站上季線
        if close > ma60:
            score += 30
            factors.append("🎯 站上季線")
        
        # 條件 C: 底部爆量吸籌
        vol_ratio = curr['Volume'] / vol_ma5
        if vol_ratio > 1.3: # 稍微放寬到 1.3 倍
            score += 25
            factors.append(f"🔥 量增({round(vol_ratio,1)}倍)")
        
        # 條件 D: MACD 轉折
        if macd_diff > 0 and macd_diff_prev <= 0:
            score += 20
            factors.append("⚡ MACD翻紅")
            
        # 條件 E: RSI
        if 45 < rsi < 75:
            score += 15
        
        if score >= 55: # 門檻微調至 55 分
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

st.set_page_config(page_title="Miniko 狙擊手 V5.2", layout="wide")

st.title("🏹 Miniko 狙擊手 V5.2 - 強力掃描版")
st.markdown("### 策略：尋找「剛站上季線 + 爆量」的起漲股 (內建防當機機制)")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 提醒：若全市場連線不穩，系統會自動切換為「精選熱門股」掃描，確保您一定能看到結果。")
    
    if st.button("🚀 啟動掃描", type="primary"):
        with st.spinner("正在初始化數據庫..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"目標鎖定：準備掃描 {len(tickers)} 檔股票...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 使用多執行緒
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ticker = {executor.submit(calculate_factors_sniper, t, name_map): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                if completed_count % 10 == 0:
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
        st.write("👈 點擊左側按鈕開始，結果會即時顯示...")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 歷史掃描結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
