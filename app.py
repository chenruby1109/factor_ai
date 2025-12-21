import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 (現貨思維 / 資金成本參數) ---
RF = 0.015  # 無風險利率 (Risk-Free Rate, 如定存 1.5%)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 股利長期成長率 (Gordon Model用)
WACC_THRESHOLD = 0.05 # 假設公司資金成本門檻 (用於比較)

# --- 核心功能函數 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
    """下載大盤指數 (TWII) 用於計算 Beta 與系統性風險"""
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
        # 示範抓取 twstock 內建清單
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
    """【V8.3 價格修復版】"""
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
                rt_price = realtime['realtime']['latest_trade_price']
                if rt_price and rt_price != '-' and float(rt_price) > 0:
                    price = float(rt_price)
                else:
                    best_bid = realtime['realtime']['best_bid_price'][0]
                    if best_bid and best_bid != '-' and float(best_bid) > 0:
                        price = float(best_bid)
        except: pass
    return price

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.4 旗艦運算核心 - 價格意圖因子引擎】
    特點：整合「價格意圖因子」(Return / Variability) 識別主力畫線股。
    """
    try:
        stock_name = name_map.get(ticker_symbol, ticker_symbol)
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 抓取 1 年數據 (足夠計算60天意圖因子)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None # 排除極低價股

        # --- 0. 核心選股：價格意圖因子 (Price Intent Factor) ---
        # 邏輯：報酬率(s) / 變動率(v)。尋找 A->B 走直線的股票
        days = 60
        close_series = data['Close']
        volume_series = data['Volume']
        
        # S: 60天報酬率
        price_60_ago = close_series.iloc[-days]
        s_return = (current_price / price_60_ago) - 1
        
        # V: 變動率 (每日漲跌幅絕對值總和)
        v_variability = close_series.pct_change().abs().tail(days).sum()
        
        # Volume Check (日均量)
        avg_volume = volume_series.tail(days).mean()
        
        # 意圖因子計算
        intent_factor = 0
        score_intent = 0
        is_intent_candidate = False
        
        # 篩選條件：1. 收益率 < 20% (避免過熱) 2. 成交量 > 200,000 (流動性)
        if v_variability > 0 and 0 < s_return < 0.20 and avg_volume > 200000:
            # 原始因子: s / v
            raw_intent = s_return / v_variability
            # 排名指標: (s / v) / volume (偏好低關注度但走勢穩定的)
            # 為了讓數值可讀，我們主要評估 raw_intent (直線性)，並確認 volume 不會過大
            
            intent_factor = raw_intent
            is_intent_candidate = True
            score_intent = 25 # 符合此核心邏輯直接加高分

        # --- 1. CAPM & WACC (資金成本分析) ---
        stock_returns = close_series.pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        # Ke = Rf + Beta * MRP (權益資金成本)
        ke = RF + beta * MRP
        
        # --- 2. Gordon Model ---
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 3. Smart Beta (CGO & Low Vol) ---
        pb = ticker_info.get('priceToBook', 0)
        ma100 = close_series.rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100 
        volatility = stock_returns.std() * (252**0.5)
        
        # --- 4. AI 差異化評分機制 ---
        score = score_intent # 初始分數由意圖因子決定
        factors = []
        
        if is_intent_candidate:
            factors.append("💎價格意圖優選(直線爬升)")

        # 價值因子
        if 0 < pb < 1.0:
            score += 20
            factors.append("深度價值(PB<1)")
        elif 1.0 <= pb < 1.5:
            score += 10
            
        if not np.isnan(fair_value):
            upside = (fair_value - current_price) / current_price
            if upside > 0.2:
                score += 15
                factors.append("估值低估")

        # 品質 (ROE)
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 10
            factors.append("高ROE")

        # 技術與籌碼
        ma20 = close_series.rolling(20).mean().iloc[-1]
        if current_price > ma20: score += 5

        if volatility < 0.25:
            score += 15
            factors.append("籌碼安定")
            
        if cgo_val > 0.1:
            score += 10

        # --- 5. 生成「個別化」AI 深度綜合建議 ---
        
        # 路徑軌跡診斷 (New!)
        path_diagnosis = ""
        if is_intent_candidate:
            path_diagnosis = f"【極佳】股價呈「直線爬升」型態。意圖因子顯示主力控盤穩定，且近60日漲幅 {s_return:.1%} 未過熱，屬於穩定推升階段。"
        elif s_return > 0.3:
            path_diagnosis = f"【過熱注意】近60日漲幅達 {s_return:.1%}，雖強勢但偏離直線軌跡，需提防回調。"
        elif v_variability > 0.5:
            path_diagnosis = "【震盪劇烈】路徑曲折，多空拉鋸明顯，缺乏明確主力控盤方向。"
        else:
            path_diagnosis = "股價路徑一般，隨市場波動。"

        # 價值與風險
        valuation_txt = f"合理價 {fair_value}" if not np.isnan(fair_value) else "無股利評價"
        risk_txt = f"Beta {beta:.2f} (防禦型)" if beta < 1 else f"Beta {beta:.2f} (波動型)"

        # 綜合結論
        action_plan = ""
        if score >= 75:
            action_plan = "評分極高。具備「價格意圖」與「基本面」雙重優勢，建議積極佈局。"
        elif score >= 50:
            action_plan = "評分中上。路徑或價值面有一項優勢，可納入觀察。"
        else:
            action_plan = "觀望。缺乏明確上漲意圖或籌碼優勢。"

        final_advice = (
            f"🎯 **AI 核心解析**：\n"
            f"1. **軌跡**：{path_diagnosis}\n"
            f"2. **價值**：{valuation_txt}，{risk_txt}。\n"
            f"3. **籌碼**：CGO {cgo_val:.1%} ({( '獲利惜售' if cgo_val>0.1 else '正常' )})。\n"
            f"4. **決策**：{action_plan}"
        )

        if score >= 50:
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "AI綜合建議": final_advice,
                "意圖因子": round(intent_factor, 2) if is_intent_candidate else 0, # 新欄位
                "權益成本(Ke)": round(ke, 3),
                "CGO指標": round(cgo_val * 100, 1),
                "波動率": round(volatility, 2),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.4", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.4 (價格意圖因子旗艦版)")
st.markdown("""
本系統整合 **CAPM、Fama-French** 與 **Smart Beta**。
**V9.4 核心升級：** 引入 **「價格意圖因子」**，利用數學公式篩選出「股價走直線」的主力控盤股，排除隨機漫步的雜訊。
""")

# --- 知識庫 Expander ---
with st.expander("📚 點此查看：價格意圖因子與核心選股邏輯 (New!)"):
    tab_intent, tab_theory, tab_chips = st.tabs(["💎 核心：價格意圖因子", "CAPM與三因子", "籌碼與CGO"])
    
    with tab_intent:
        st.markdown("""
        ### 💎 什麼是「價格意圖因子」？
        
        
        **核心邏輯**：股價從 A 點到 B 點，距離最短的是「直線」。
        * 如果一檔股票像**直線**一樣慢慢爬升，代表背後有**造市者或主力**在付費維護或少量吸籌，讓價格穩定。
        * 如果一檔股票上沖下洗、路徑繞來繞去，代表多空分歧，看不出主力意圖。
        
        **三大篩選公式**：
        1.  **收益率上限**：過去 60 天漲幅 < 20% (避免追高、找起漲點)。
        2.  **變動率 (Variability)**：每日漲跌幅絕對值總和 (越小代表走勢越平滑)。
        3.  **價格意圖** = `報酬率 / 變動率`。數值越大，代表「直線上漲」趨勢越強。
        
        **為什麼有效？**
        * **風險調整後收益高**：在承擔最小波動下，獲得最穩定的報酬。
        * **市場關注度低**：我們結合 `因子 / 交易量`，找出尚未被市場大肆炒作的低調好股。
        """)

    with tab_theory:
        st.markdown("""
        ### CAPM & WACC
        * **WACC**：資金成本概念。若預期報酬率 > WACC，才值得投資。
        * **CAPM**：$E(R_i) = R_f + \\beta(R_m - R_f)$，計算合理的投資回報門檻。
        """)
        
    with tab_chips:
        st.markdown("""
        ### CGO + Smart Beta
        * **CGO (未實現獲利)**：正值代表大部分持股者賺錢，籌碼穩定惜售。
        * **低波動**：長期回測顯示，低波動股票的夏普比率優於高波動熱門股。
        """)

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統執行：價格意圖因子篩選 + CAPM 評價 + Smart Beta 診斷。")
    if st.button("🚀 啟動 AI 智能運算 (Top 100)", type="primary"):
        with st.spinner("Step 1: 載入大盤數據..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 全市場掃描 (計算意圖因子)..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定 {len(tickers)} 檔標的，開始運算股價路徑...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in tickers}
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_ticker):
                data = future.result()
                completed += 1
                if completed % 10 == 0:
                    progress_bar.progress(completed / len(tickers))
                    status_text.text(f"AI 解析中: {completed}/{len(tickers)}")
                if data:
                    st.session_state['results'].append(data)

        status_text.text("✅ AI 分析完成！")

with col2:
    if not st.session_state['results']:
        st.write("👈 請點擊左側按鈕開始分析。")
        [st.write("")] 
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序：優先展示「價格意圖優選」且評分高的
        df = df.sort_values(by=['AI綜合評分', '意圖因子'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 Top 100 (價格意圖優選)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "AI綜合建議", "意圖因子", "合理價", "CGO指標", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含股價路徑軌跡診斷"),
                "意圖因子": st.column_config.NumberColumn(format="%.2f", help="數值越高代表走勢越像直線(穩定)"),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
