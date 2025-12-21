import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures

# --- 設定區 (Configuration) ---

# 1. 擴充觀察名單：包含權值股、AI供應鏈、熱門中型股
# 為了抓到大戶佈局，範圍要夠廣，但又不能是成交量太小的殭屍股
STOCK_MAP = {
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2308.TW': '台達電', 
    '2382.TW': '廣達', '2412.TW': '中華電', '2881.TW': '富邦金', '2882.TW': '國泰金', 
    '2303.TW': '聯電', '2379.TW': '瑞昱', '2395.TW': '研華', '2603.TW': '長榮', 
    '2609.TW': '陽明', '2615.TW': '萬海', '3008.TW': '大立光', '3034.TW': '聯詠', 
    '3037.TW': '欣興', '3231.TW': '緯創', '3443.TW': '創意', '3661.TW': '世芯-KY', 
    '6669.TW': '緯穎', '8299.TW': '群聯', '4927.TW': '泰鼎-KY', '3035.TW': '智原', 
    '3529.TW': '力旺', '2360.TW': '致茂', '6278.TW': '台表科', '2356.TW': '英業達', 
    '2376.TW': '技嘉', '2388.TW': '威盛', '2455.TW': '全新', '3105.TW': '穩懋', 
    '8086.TW': '宏捷科', '6213.TW': '聯茂', '2368.TW': '金像電', '6274.TW': '台燿',
    '3017.TW': '奇鋐', '3324.TW': '雙鴻', '2421.TW': '建準', '5274.TW': '信驊',
    '6415.TW': '矽力-KY', '6770.TW': '力積電', '5347.TW': '世界', '3711.TW': '日月光',
    '2344.TW': '華邦電', '2408.TW': '南亞科', '6147.TW': '頎邦', '3532.TW': '台勝科',
    '6488.TW': '環球晶', '5483.TW': '中美晶', '8069.TW': '元太', '9958.TW': '世紀鋼',
    '1513.TW': '中興電', '1519.TW': '華城', '1503.TW': '士電', '1504.TW': '東元'
}
TICKERS = list(STOCK_MAP.keys())

# 2. Telegram 設定
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 核心功能模組 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def get_stock_name(ticker):
    return STOCK_MAP.get(ticker, ticker)

def calculate_factors_sniper(ticker_symbol, stock_df, market_df=None):
    """
    【Miniko 狙擊手版 V4.0】專抓第一浪起漲點
    特點：
    1. 拒絕追高：嚴格的乖離率濾網
    2. 底部吸籌：量價背離偵測
    3. 低檔轉折：MACD 水下金叉
    """
    if len(stock_df) < 60: return None 

    # 取最近一筆與前一筆數據
    curr = stock_df.iloc[-1]
    prev = stock_df.iloc[-2]
    current_price = curr['Close']
    
    # --- 0. 基本面濾網 (只要不爛就好，不用太嚴苛，因為轉機股通常財報還沒爆發) ---
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        eps = info.get('trailingEps', None)
        if eps is None: eps = info.get('forwardEps', 0)
        
        # 成長率 (若資料缺失給予預設值，避免錯殺轉機股)
        revenue_growth = info.get('revenueGrowth', 0) 
        
        # PEG 計算
        peg_ratio = None
        if eps and eps > 0 and revenue_growth and revenue_growth > 0:
            pe_ratio = current_price / eps
            peg_ratio = pe_ratio / (revenue_growth * 100)
            
    except:
        peg_ratio = revenue_growth = None

    # --- 1. 技術指標運算 ---
    stock_df['MA20'] = ta.trend.sma_indicator(stock_df['Close'], window=20)
    stock_df['MA60'] = ta.trend.sma_indicator(stock_df['Close'], window=60) # 季線(生命線)
    
    # 乖離率 (Bias): (股價 - 60MA) / 60MA
    # 這是判斷是否為「第一浪」的關鍵。如果 > 20%，通常已經是第三浪了。
    bias_60 = (current_price - curr['MA60']) / curr['MA60']

    # MACD
    macd = ta.trend.MACD(stock_df['Close'])
    stock_df['MACD_Line'] = macd.macd()
    stock_df['MACD_Signal'] = macd.macd_signal()
    stock_df['MACD_Diff'] = macd.macd_diff()
    
    # 成交量平均 (5日均量)
    stock_df['Vol_MA5'] = stock_df['Volume'].rolling(window=5).mean()
    
    # RSI
    stock_df['RSI'] = ta.momentum.rsi(stock_df['Close'], window=14)

    # --- 2. 狙擊手評分系統 (Scoring) ---
    score = 0
    factors = [] 

    # === 第一關：絕對過濾 (Filter) ===
    # 如果股價離季線太遠 (> 25%)，直接淘汰 (拒絕追高 Wave 3/5)
    if bias_60 > 0.25: 
        return None # 直接不看這檔
    
    # === 第二關：多因子加分 ===

    # F1. 潛伏期突破 (剛站上季線)
    # 邏輯：股價在季線附近 (-5% ~ +10%) 且站上季線
    if -0.05 <= bias_60 <= 0.10 and current_price > curr['MA60']:
        score += 30
        factors.append("🎯 剛站上季線 (起漲點)")

    # F2. 底部吸籌 (量價結構)
    # 邏輯：成交量大增 (> 1.5倍均量) 但 股價漲幅不大 (< 4%) -> 主力壓低吃貨
    price_change_pct = (curr['Close'] - prev['Close']) / prev['Close']
    vol_ratio = curr['Volume'] / curr['Vol_MA5'] if curr['Vol_MA5'] > 0 else 0
    
    if vol_ratio > 1.5 and abs(price_change_pct) < 0.04:
        score += 25
        factors.append(f"🤫 主力吸籌 (量增價穩)")
    elif vol_ratio > 2.0 and price_change_pct > 0.0:
        score += 20
        factors.append(f"🔥 爆量攻擊")

    # F3. 技術面轉折 (Reversal)
    # 邏輯：MACD 剛翻紅 或是 RSI 從低檔翻揚 (40-60)
    if curr['MACD_Diff'] > 0 and prev['MACD_Diff'] <= 0:
        score += 20
        factors.append("⚡ MACD翻紅轉折")
    
    if 40 < curr['RSI'] < 65: # 剛睡醒，還沒過熱
        score += 10
        factors.append("📈 RSI甦醒區")
    elif curr['RSI'] > 75: # 過熱扣分
        score -= 10
        factors.append("⚠️ RSI過熱")

    # F4. 價值保護 (PEG)
    # 既然要買第一浪，最好買在還有價值低估的時候
    if peg_ratio and peg_ratio < 1.0:
        score += 15
        factors.append(f"💎 價值低估 (PEG {round(peg_ratio, 2)})")

    # 總分過濾
    if score < 50: return None

    return {
        "Ticker": ticker_symbol,
        "Name": get_stock_name(ticker_symbol),
        "Close": round(current_price, 2),
        "Score": score,
        "Bias": f"{round(bias_60*100, 1)}%", # 顯示乖離率
        "Factors": " | ".join(factors),
        "PEG": round(peg_ratio, 2) if peg_ratio else "N/A"
    }

def run_analysis_parallel():
    """多執行緒加速"""
    results = []
    status_text = st.empty()
    bar = st.progress(0)
    
    # 這次不需要大盤資料，專注個股型態
    
    def analyze_one(ticker):
        try:
            data = yf.download(ticker, period="6mo", interval="1d", progress=False)
            if data.empty: return None
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return calculate_factors_sniper(ticker, data)
        except: return None

    status_text.text(f"正在執行「第一浪狙擊」掃描 ({len(TICKERS)} 檔)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_ticker = {executor.submit(analyze_one, ticker): ticker for ticker in TICKERS}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            data = future.result()
            if data:
                results.append(data)
            completed += 1
            bar.progress(completed / len(TICKERS))

    status_text.text("掃描完成！")
    
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # 欄位調整，把乖離率 (Bias) 放前面方便檢查
        cols = ['Name', 'Ticker', 'Close', 'Score', 'Bias', 'Factors', 'PEG']
        df_res = df_res[cols].sort_values(by='Score', ascending=False)
        return df_res
    return pd.DataFrame()

# --- Streamlit 頁面 ---

st.set_page_config(page_title="Miniko 狙擊手 V4", layout="wide")

st.title("🏹 Miniko & 曜鼎豐 - 第一浪狙擊手 (V4)")
st.caption("策略目標：尋找剛站上季線、主力低檔吸籌、尚未噴出的潛力股 (拒絕追高)")
st.markdown("---")

col1, col2 = st.columns([1, 4])

with col1:
    st.header("戰情中心")
    if st.button("🏹 啟動狙擊掃描", type="primary"):
        with st.spinner('正在過濾高檔股，尋找底部起漲點...'):
            result_df = run_analysis_parallel()
            
            if not result_df.empty:
                st.session_state['data'] = result_df
                st.success(f"發現 {len(result_df)} 檔潛伏股！")
                
                # 發送 Telegram
                top_picks = result_df[result_df['Score'] >= 70]
                if not top_picks.empty:
                    msg = f"🏹 **【Miniko 狙擊訊號 (第一浪)】** 🏹\n\n"
                    for _, row in top_picks.iterrows():
                        msg += f"• {row['Name']} ({row['Ticker']}) ${row['Close']}\n  得分: {row['Score']} | 乖離: {row['Bias']}\n  {row['Factors']}\n"
                    msg += f"\n時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    send_telegram_message(msg)
            else:
                st.warning("目前沒有發現符合「底部起漲」條件的股票，市場可能過熱或過冷。")

with col2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # 顯示高分狙擊名單
        st.subheader("🎯 最佳狙擊目標 (Score >= 70)")
        st.write("特徵：剛突破季線 + 籌碼進駐 + 價值被低估")
        st.dataframe(
            df[df['Score'] >= 70].style.highlight_max(axis=0, color='#fff3cd'), 
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("👀 觀察中 (蓄勢待發)")
        st.dataframe(
            df[(df['Score'] < 70)], 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("👈 請點擊「啟動狙擊掃描」")
        st.markdown("""
        **V4 狙擊手版本特點：**
        1. **拒絕追高濾網**：只要股價離季線太遠 (>25%)，直接剔除，避免買在第五浪。
        2. **抓轉折**：鎖定「MACD 翻紅」且「剛站上季線」的黃金時機。
        3. **量價秘密**：偵測「量增價穩」的主力吸籌訊號。
        4. **適合標的**：此模式選出的股票通常看起來「剛睡醒」，這才是大戶進場的位置。
        """)
