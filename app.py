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
    使用 twstock 直接調用內建字典，獲取全台 1800+ 檔股票代號
    優點：速度快、不需要連線證交所、絕對不會有 SSL 錯誤
    """
    tickers = []
    name_map = {}
    
    try:
        # 遍歷 twstock 資料庫
        for code, info in twstock.codes.items():
            # 過濾條件：只抓「股票」，排除權證(W)、ETF(00)、特別股等
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
        st.error(f"字典讀取失敗: {e}")
        return [], {}

def calculate_fgm_score(ticker_symbol, name_map):
    """
    【Miniko F-G-M 大戶模型】
    F (Fundamentals): ROE, PEG (價值與品質)
    G (Growth): 營收成長 (動能來源)
    M (Momentum): 剛站上季線, MACD翻紅, 量能異常 (狙擊進場點)
    """
    try:
        # 1. 下載數據 (抓取半年數據以計算季線)
        data = yf.download(ticker_symbol, period="6mo", interval="1d", progress=False)
        
        # 資料防呆
        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        curr = data.iloc[-1]
        prev = data.iloc[-2]
        close = curr['Close']
        volume = curr['Volume']

        # --- 0. 初步過濾 (Filter) ---
        # 排除流動性太差的股票 (成交量 < 300張 或 股價 < 10元)
        if volume < 300000 or close < 10: return None

        # --- 1. 計算基本面與成長因子 (Fundamentals & Growth) ---
        # 由於 yfinance 台股財報常缺漏，我們用「估算」方式
        ticker_info = yf.Ticker(ticker_symbol).info
        
        # G: 營收成長 (Revenue Growth)
        rev_growth = ticker_info.get('revenueGrowth', 0) # 0.25 = 25%
        
        # F: ROE (股東權益報酬率) - 代表公司賺錢效率
        roe = ticker_info.get('returnOnEquity', 0)
        
        # F: PEG (本益成長比) - 大戶找便宜的關鍵
        # 如果抓不到 PEG，我們嘗試自己算: PE / (Growth*100)
        peg = ticker_info.get('pegRatio', None)
        pe = ticker_info.get('trailingPE', None)
        if peg is None and pe and rev_growth > 0:
            peg = pe / (rev_growth * 100)

        # --- 2. 計算技術面因子 (Momentum) ---
        # 均線
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1] # 生命線
        
        # 乖離率 (Bias): 用來判斷是否「追高」
        bias_60 = (close - ma60) / ma60
        
        # 成交量均線
        vol_ma5 = data['Volume'].rolling(5).mean().iloc[-1]
        
        # MACD
        macd = ta.trend.MACD(data['Close'])
        macd_diff = macd.macd_diff().iloc[-1]
        macd_diff_prev = macd.macd_diff().iloc[-2]

        # --- 3. 大戶評分系統 (Scoring) ---
        score = 0
        factors = []
        
        # === 守門員：乖離率濾網 ===
        # 如果股價已經離季線太遠 (> 25%)，大戶不會追，我們也不追
        if bias_60 > 0.25: return None
        # 如果還在深海空頭排列 (季線下方 > 15%)，也不是好買點
        if bias_60 < -0.15: return None

        # === 因子加分區 ===
        
        # [G] 成長因子: 營收高成長 (+20分)
        if rev_growth and rev_growth > 0.20:
            score += 20
            factors.append(f"📈 營收爆發(+{round(rev_growth*100)}%)")
            
        # [F] 價值因子: PEG 低估 (+15分)
        if peg and 0 < peg < 1.0:
            score += 15
            factors.append(f"💎 價值低估(PEG {round(peg, 2)})")
            
        # [F] 品質因子: 高 ROE (+10分)
        if roe and roe > 0.15:
            score += 10
            factors.append(f"👑 高效能(ROE {round(roe*100)}%)")

        # [M] 狙擊手因子 1: 剛站上季線 (+20分)
        # 這是第一浪/第二浪轉強的特徵
        if close > ma60 and (close - ma60)/ma60 < 0.05:
            score += 20
            factors.append("🎯 剛站上季線")
        elif close > ma60:
            score += 10 # 站上但有點距離

        # [M] 狙擊手因子 2: 主力吸籌 (+20分)
        # 量增 (1.5倍) 但價穩 (漲幅 < 5%) -> 大戶偷偷買
        pct_change = (close - prev['Close']) / prev['Close']
        vol_ratio = volume / vol_ma5
        if vol_ratio > 1.5 and 0 < pct_change < 0.05:
            score += 20
            factors.append(f"🤫 主力吸籌(量增{round(vol_ratio,1)}倍)")
        elif vol_ratio > 2.0:
            score += 10
            factors.append("🔥 爆量攻擊")

        # [M] 狙擊手因子 3: MACD 轉折 (+15分)
        if macd_diff > 0 and macd_diff_prev <= 0:
            score += 15
            factors.append("⚡ MACD翻紅")

        # 總分門檻 (稍微放寬到 50 分，確保有結果，然後我們看排名)
        if score >= 50:
            return {
                "Ticker": ticker_symbol,
                "Name": name_map.get(ticker_symbol, ticker_symbol),
                "Close": round(close, 2),
                "Score": score,
                "Bias": f"{round(bias_60*100, 1)}%",
                "Factors": " | ".join(factors),
                "PEG": round(peg, 2) if peg else "N/A",
                "Growth": f"{round(rev_growth*100)}%" if rev_growth else "N/A"
            }
            
    except:
        return None
    return None

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko FGM 狙擊手 V6", layout="wide")

st.title("🏹 Miniko & 曜鼎豐 - 全市場 F-G-M 狙擊手")
st.markdown("""
### 策略邏輯：
* **F (基本面)**：尋找被低估 (PEG<1) 且高效能 (ROE>15%) 的好公司。
* **G (成長面)**：營收年增率 > 20%，確保動能。
* **M (技術面)**：**拒絕追高！** 鎖定剛站上季線、主力吸籌的起漲點 (第一浪)。
""")

if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 3])

with col1:
    st.info("💡 說明：掃描全台 1800+ 檔股票約需 20 分鐘。只要發現符合 FGM 模型的好股，右側會即時顯示。")
    
    if st.button("🚀 啟動全市場 FGM 掃描", type="primary"):
        with st.spinner("正在讀取 twstock 字典資料庫..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"成功載入 {len(tickers)} 檔股票！開始大戶邏輯分析...")
        st.session_state['results'] = [] 
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        result_placeholder = col2.empty() 
        
        # 開啟多執行緒加速 (16核心)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_ticker = {executor.submit(calculate_fgm_score, t, name_map): t for t in tickers}
            
            completed_count = 0
            found_count = 0
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed_count += 1
                
                # 每 50 檔更新一次進度條
                if completed_count % 50 == 0:
                    progress_bar.progress(completed_count / len(tickers))
                    status_text.text(f"掃描進度: {completed_count}/{len(tickers)} | 已發現: {found_count} 檔")
                
                if data:
                    found_count += 1
                    st.session_state['results'].append(data)
                    
                    # 即時排序並顯示
                    df_realtime = pd.DataFrame(st.session_state['results'])
                    df_realtime = df_realtime.sort_values(by='Score', ascending=False)
                    
                    with result_placeholder.container():
                        st.subheader(f"🎯 發現 FGM 潛力股 ({found_count} 檔)")
                        # 顯示關鍵欄位
                        st.dataframe(
                            df_realtime[['Name', 'Ticker', 'Close', 'Score', 'Bias', 'Factors', 'PEG', 'Growth']], 
                            use_container_width=True,
                            hide_index=True
                        )

        status_text.text("✅ 全市場掃描完成！")
        
        # 發送 Telegram 通知
        if st.session_state['results']:
            df_final = pd.DataFrame(st.session_state['results']).sort_values(by='Score', ascending=False)
            top_5 = df_final.head(5)
            msg = f"🏹 **【Miniko FGM 狙擊報告】**\n發現 {len(df_final)} 檔潛力股，前五名：\n"
            for _, row in top_5.iterrows():
                msg += f"• {row['Name']} ({row['Ticker']}) {row['Close']}元 | 分數:{row['Score']}\n"
            send_telegram_message(msg)

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊左側按鈕開始，這次保證能跑出全市場結果！")
    else:
        df_show = pd.DataFrame(st.session_state['results'])
        st.subheader(f"🎯 最終篩選結果 ({len(df_show)} 檔)")
        st.dataframe(
            df_show.sort_values(by='Score', ascending=False), 
            use_container_width=True, 
            hide_index=True
        )
