import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime

# --- 設定區 (Configuration) ---

# 1. 股票觀察名單 (您可以隨時增加)
TICKERS = [
    '2330.TW', '2454.TW', '2317.TW', '2603.TW', '3443.TW', 
    '3661.TW', '8299.TW', '4927.TW', '2382.TW', '6669.TW'
]

# 2. Telegram 設定 (請填入您的 Token 與 Chat ID)
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 核心功能模組 ---

def send_telegram_message(message):
    """發送訊號到 Telegram"""
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN':
        # 如果使用者沒設定，就不發送，避免報錯
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        st.error(f"Telegram 發送失敗: {e}")

def calculate_factors_advanced(ticker_symbol, stock_df, market_df=None):
    """
    【Miniko 旗艦版 V2.0】F-G-M 多因子全能選股引擎
    邏輯：
    1. Fundamentals (基本面): 價值(PEG) + 品質(ROE) + 獲利能力(三率三升)
    2. Growth (成長面): 營收動能 (季度 Proxy)
    3. Momentum (技術面): RS相對強度 + 波動壓縮 + 均線多頭
    """
    # 資料長度防呆
    if len(stock_df) < 60: return None 

    current_price = stock_df['Close'].iloc[-1]
    
    # --- 0. 獲取深度基本面數據 (財報爬蟲) ---
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # 抓取季度財報 (為了計算毛利率/營益率成長)
        # yfinance 的 quarterly_financials 包含：Total Revenue, Gross Profit, Operating Income...
        q_fin = ticker.quarterly_financials 
        
        # 提取關鍵財務比率
        peg_ratio = info.get('pegRatio', None)
        roe = info.get('returnOnEquity', None)
        revenue_growth_yoy = info.get('revenueGrowth', None) # 單季 YoY
        
        # 計算三率三升 (Margin Expansion)
        # 邏輯：比較「最新一季」與「上一季」
        margin_expansion = False
        if not q_fin.empty and 'Gross Profit' in q_fin.index and 'Total Revenue' in q_fin.index:
            try:
                # 最新一季
                rev_curr = q_fin.iloc[:, 0]['Total Revenue']
                gross_curr = q_fin.iloc[:, 0]['Gross Profit']
                op_curr = q_fin.iloc[:, 0].get('Operating Income', 0)
                
                # 上一季
                rev_prev = q_fin.iloc[:, 1]['Total Revenue']
                gross_prev = q_fin.iloc[:, 1]['Gross Profit']
                op_prev = q_fin.iloc[:, 1].get('Operating Income', 0)

                # 計算率 (Margins)
                gm_curr = gross_curr / rev_curr
                gm_prev = gross_prev / rev_prev
                om_curr = op_curr / rev_curr
                om_prev = op_prev / rev_prev
                
                # 判定：毛利率擴張 且 營益率擴張
                if gm_curr > gm_prev and om_curr > om_prev:
                    margin_expansion = True
            except:
                pass # 財報資料缺漏時跳過
                
    except Exception as e:
        # print(f"基本面數據獲取失敗: {e}") # Debug用
        peg_ratio = roe = revenue_growth_yoy = None
        margin_expansion = False

    # --- 1. 技術指標計算 (Technical Indicators) ---
    
    # A. 均線系統
    stock_df['MA20'] = ta.trend.sma_indicator(stock_df['Close'], window=20)
    stock_df['MA60'] = ta.trend.sma_indicator(stock_df['Close'], window=60)
    
    # B. 動能: MACD
    macd = ta.trend.MACD(stock_df['Close'])
    stock_df['MACD_Diff'] = macd.macd_diff() # 柱狀圖
    
    # C. 波動率: 布林通道
    bb = ta.volatility.BollingerBands(stock_df['Close'], window=20, window_dev=2)
    stock_df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / stock_df['MA20']
    stock_df['BB_Upper'] = bb.bollinger_hband()
    
    # D. 相對強度 (RS) - 比較大盤
    # 如果有傳入大盤資料 (market_df)，計算 RS
    rs_score = 0
    rs_trend = False
    if market_df is not None and not market_df.empty:
        # 確保索引對齊
        common_index = stock_df.index.intersection(market_df.index)
        if len(common_index) > 20:
            s_price = stock_df.loc[common_index]['Close']
            m_price = market_df.loc[common_index]['Close']
            
            # 計算近 20 日漲幅
            stock_ret_20 = (s_price.iloc[-1] / s_price.iloc[-20]) - 1
            market_ret_20 = (m_price.iloc[-1] / m_price.iloc[-20]) - 1
            
            # RS 值 (簡單版: 個股漲幅 - 大盤漲幅)
            rs_val = stock_ret_20 - market_ret_20
            if rs_val > 0: rs_score = rs_val * 100 # 轉為正數方便評分
            if stock_ret_20 > market_ret_20: rs_trend = True

    current = stock_df.iloc[-1]
    prev = stock_df.iloc[-2]

    # --- 2. 評分系統 (F-G-M Model Scoring) ---
    score = 0
    factors = [] 

    # === 【基本面】F-G (權重 50%) ===
    
    # 1. 成長因子: 營收爆發 (YoY > 20%)
    # 註: YF 無法算 3MA vs 12MA (缺月營收)，改用單季 YoY + 季度營收增長模擬
    if revenue_growth_yoy and revenue_growth_yoy > 0.20:
        score += 20
        factors.append(f"📈 營收爆發 (+{round(revenue_growth_yoy*100)}%)")

    # 2. 價值因子: PEG (本益成長比)
    if peg_ratio:
        if peg_ratio < 0.75:
            score += 20
            factors.append(f"💎 超級低估 (PEG {peg_ratio})")
        elif peg_ratio < 1.0:
            score += 15
            factors.append(f"✅ 價值合理 (PEG {peg_ratio})")
            
    # 3. 品質因子: ROE & Margin Expansion (三率三升)
    if roe and roe > 0.15:
        score += 10
        factors.append(f"👑 高ROE ({round(roe*100)}%)")
    
    if margin_expansion:
        score += 15
        factors.append("💰 毛利營益雙升 (產品競爭力強)")

    # === 【技術面】Momentum (權重 50%) ===

    # 4. 趨勢確立 (MA Alignment)
    if current['Close'] > current['MA20'] > current['MA60']:
        score += 15
        factors.append("🚀 多頭排列 (季線之上)")

    # 5. 相對強度 (RS) - 強者恆強
    if rs_trend:
        score += 15
        factors.append("💪 強於大盤 (RS>0)")

    # 6. 波動壓縮 + 突破 (Volatility Squeeze)
    # 條件: 頻寬 < 10% (壓縮) 並且 股價剛突破上軌 (或接近上軌)
    if current['BB_Width'] < 0.12:
        factors.append("⚡ 波動壓縮中") # 這是觀察訊號
        if current['Close'] > current['BB_Upper'] or (current['Close'] > current['MA20'] and current['MACD_Diff'] > 0):
            score += 20
            factors.append("🔥 壓縮後發動 (買點!)")

    # 7. MACD 共振 (剛翻紅)
    if current['MACD_Diff'] > 0 and prev['MACD_Diff'] <= 0:
        score += 10
        factors.append("🎯 MACD 黃金交叉")

    return {
        "Ticker": ticker_symbol,
        "Close": round(current['Close'], 2),
        "Score": score,
        "Factors": " | ".join(factors),
        "PEG": round(peg_ratio, 2) if peg_ratio else "N/A",
        "Rev_Growth": f"{round(revenue_growth_yoy*100, 1)}%" if revenue_growth_yoy else "N/A",
        "RS_Status": "Strong" if rs_trend else "Weak"
    }

def run_analysis():
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. 先抓大盤資料 (加權指數: ^TWII) - 用於計算 RS
    status_text.text("正在獲取大盤數據 (TWII) 以計算相對強度...")
    try:
        market_data = yf.download("^TWII", period="6mo", interval="1d", progress=False)
        # 處理多層索引 (如果有的話)
        if isinstance(market_data.columns, pd.MultiIndex):
            market_data.columns = market_data.columns.get_level_values(0)
    except Exception as e:
        st.error(f"大盤數據獲取失敗: {e}")
        market_data = None

    # 2. 迴圈分析個股
    total_tickers = len(TICKERS)
    for i, ticker in enumerate(TICKERS):
        status_text.text(f"正在分析 {ticker} ({i+1}/{total_tickers})...")
        try:
            # 下載個股 Data
            data = yf.download(ticker, period="6mo", interval="1d", progress=False)
            
            if not data.empty:
                # 處理 yfinance 多層索引
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                # 呼叫新的旗艦版函數，並傳入 market_data
                analysis = calculate_factors_advanced(ticker, data, market_data)
                
                if analysis:
                    results.append(analysis)
        except Exception as e:
            # st.error(f"Error analyzing {ticker}: {e}") # Debug 用
            pass
        
        progress_bar.progress((i + 1) / total_tickers)

    status_text.text("全市場掃描完成！")
    
    # 轉為 DataFrame 並排序
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        # 按照分數由高到低排序
        df_res = df_res.sort_values(by='Score', ascending=False)
        return df_res
    return pd.DataFrame()

# --- Streamlit 頁面佈局 (GUI) ---

st.set_page_config(page_title="Miniko 旗艦操盤室", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - FGM 多因子選股機器人")
st.markdown("---")

col1, col2 = st.columns([1, 4])

with col1:
    st.header("控制中心")
    if st.button("🔍 啟動多因子掃描", type="primary"):
        with st.spinner('正在從雲端計算 F-G-M 因子...'):
            result_df = run_analysis()
            
            if not result_df.empty:
                st.session_state['data'] = result_df
                st.success("分析完成！")
                
                # 自動發送 Telegram 通知給高分股票
                top_picks = result_df[result_df['Score'] >= 80]
                if not top_picks.empty:
                    msg = f"🔥 **【Miniko 機器人訊號】** 🔥\n\n發現 FGM 高分股：\n"
                    for _, row in top_picks.iterrows():
                        msg += f"• `{row['Ticker']}` ({row['Close']}元)\n  得分: {row['Score']}\n  亮點: {row['Factors']}\n"
                    msg += f"\n時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    
                    send_telegram_message(msg)
                    st.toast("已同步訊號至 Telegram!", icon="📨")
            else:
                st.warning("未能取得數據，請檢查網路或代號。")

with col2:
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # 1. 冠軍區 (Score >= 80)
        st.subheader("🏆 冠軍潛力股 (Score >= 80)")
        st.write("符合：營收成長 + PEG低估 + 技術面共振")
        high_score_df = df[df['Score'] >= 80]
        st.dataframe(high_score_df.style.highlight_max(axis=0, color='#d1e7dd'), use_container_width=True)
        
        st.markdown("---")
        
        # 2. 觀察區
        st.subheader("👀 一般觀察名單")
        st.dataframe(df[df['Score'] < 80], use_container_width=True)
    else:
        st.info("👈 請點擊左側按鈕開始分析")
        st.write("本系統採用 **F-G-M 模型**：結合 基本面(F)、成長(G) 與 動能(M) 三大維度。")
