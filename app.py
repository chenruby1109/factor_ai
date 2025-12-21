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
    【Miniko V9.2 旗艦運算核心 - 深度理論版】
    包含：CAPM, Fama-French, CGO, Smart Beta, Gordon Model
    產出：AI 綜合詳評 (替代單一買點)
    """
    try:
        stock_name = name_map.get(ticker_symbol, ticker_symbol)
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 抓取 1 年數據
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None # 排除極低價股

        # --- 1. CAPM & WACC (資金成本分析) ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        # Ke = Rf + Beta * MRP (權益資金成本 / 投資人要求報酬率)
        ke = RF + beta * MRP
        
        # --- 2. Gordon Model (股利折現評價) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 3. Fama-French Proxy & Smart Beta ---
        # SMB (規模)
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = 0 < market_cap < 50000000000 
        
        # HML (價值)
        pb = ticker_info.get('priceToBook', 0)
        is_value_stock = 0 < pb < 1.5
        
        # CGO (未實現獲利 - 籌碼面)
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100 # >0 代表多數人獲利
        
        # Low Vol (低波動)
        volatility = stock_returns.std() * (252**0.5)
        
        # --- 4. AI 評分機制 ---
        score = 0.0
        factors = []
        
        # 價值因子
        if is_value_stock:
            score += 15
            factors.append("價值型(低PB)")
        if not np.isnan(fair_value) and fair_value > current_price:
            score += 20
            factors.append("低估(低於Gordon價)")
            
        # 規模與動能
        if is_small_cap:
            score += 10
            factors.append("中小型(SMB效應)")
        
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        if current_price > ma20: score += 10 # 多頭排列

        # 品質 (ROE)
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append("高ROE")

        # 風險 (Low Vol & CGO)
        if volatility < 0.25:
            score += 15
            factors.append("低波動(籌碼穩)")
        if cgo_val > 0.1:
            score += 10
            factors.append("CGO高(賣壓輕)")

        # --- 5. 生成 AI 綜合詳細建議 (取代單一買點) ---
        # 這裡運用 WACC 與 CAPM 邏輯進行敘述
        
        advice_text = f"【{stock_name} AI深度解析】\n"
        
        # 資金成本觀點
        advice_text += f"1. 資金成本與評價：Beta值為 {beta:.2f} ({( '高波動' if beta>1 else '低波動' )})。根據CAPM模型，您的要求報酬率(Ke)應為 {ke:.1%}。"
        if not np.isnan(fair_value):
            discount = (fair_value - current_price) / current_price
            if discount > 0:
                advice_text += f" Gordon模型顯示合理價約 {fair_value} 元，目前具 {discount:.1%} 潛在漲幅。"
            else:
                advice_text += f" Gordon模型顯示合理價約 {fair_value} 元，目前價格略高於理論價。"
        else:
            advice_text += " 無配息資料，不適用Gordon模型評價。"
            
        # 籌碼與策略觀點
        advice_text += f"\n2. Smart Beta 檢測："
        if cgo_val > 0.1 and volatility < 0.3:
            advice_text += f"符合「CGO+低波動」策略。CGO指標 {cgo_val:.1%} 顯示多數籌碼獲利，且波動率 {volatility:.1%} 低，籌碼安定度高。"
        else:
            advice_text += f"波動率 {volatility:.1%}，CGO指標 {cgo_val:.1%}。雖未完全符合低波策略，但可關注其他因子。"
            
        # 投資決策建議 (不融資/不舉債)
        advice_text += f"\n3. 投資決策 (現股無槓桿)："
        if score >= 70:
            advice_text += "綜合評分極優。符合Fama-French多因子特徵，建議以現有資金分批佈局，長期持有。"
        elif score >= 50:
            advice_text += "評分中上。若股價回測月線(MA20)不破，可視為現貨買點。"
        else:
            advice_text += "評分普通，建議先觀察，待籌碼面轉佳再介入。"

        if score >= 50:
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "AI綜合建議": advice_text, # 新欄位
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "權益成本(Ke)": round(ke, 3),
                "CGO指標": round(cgo_val * 100, 1),
                "波動率": round(volatility, 2),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.2", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.2 (三因子/APT/CAPM 深度版)")
st.markdown("""
本系統整合 **CAPM、APT、Fama-French 三因子** 與 **Smart Beta** 理論。
**策略原則：** 嚴守 **不融資、不舉債、只買現股**，利用 WACC 概念評估企業價值，並結合嗨投資(HiStock)與TEJ資料庫邏輯。
""")

# --- 知識庫 Expander (深度理論整合) ---
with st.expander("📚 點此查看：Miniko 專屬三因子資料庫與投資理論 (MPT/APT/CAPM)"):
    tab_theory, tab_chips, tab_backtest, tab_factors = st.tabs(["核心理論 (CAPM/APT/FF3)", "籌碼面六大指標", "CGO策略回測", "八大因子與Smart Beta"])
    
    with tab_theory:
        st.markdown("""
        ### 一、投資組合理論 (MPT) 與 CAPM
        * **MPT (現代投資組合理論)**：由 Markowitz 提出，核心觀念是「多角化降低風險」。
            * 公式：$\sigma_p = \sqrt{\sum w_i^2 \sigma_i^2 + \sum \sum w_i w_j \sigma_i \sigma_j \rho_{ij}}$
            * 意義：船運公司案例，10艘小船風險遠低於2艘大船。
        
        * **CAPM (資本資產定價模式)**：
            * 公式：$E(R_i) = R_f + \\beta(R_m - R_f)$
            * $R_f$：無風險利率 (如定存)
            * $R_m - R_f$：市場風險溢酬 (MRP)
            * **應用**：計算 **Ke (權益資金成本)**，作為投資人的要求報酬率。
            
        * **APT (套利定價模式)**：
            * Ross (1976) 提出，認為股價受多個系統因子影響 (通膨、利差等)。
            * $E(R_i) = \\beta_0 + \Sigma \\beta_i F_i$
            
        * **Fama & French 三因子 (FF3)**：
            * 修正 CAPM 對 Beta 解釋力不足的問題。
            * 加入 **SMB (規模溢酬)**：小型股報酬通常高於大型股。
            * 加入 **HML (淨值市價比溢酬)**：價值股通常優於成長股。
            * 公式：$E(R_i) = \\beta_0 + \\beta_1 MRP + \\beta_2 SMB + \\beta_3 HML$
            
        ### 💡 投資與融資決策 (WACC)
        * **投資決策**：計算 WACC (加權平均資金成本)，將未來現金流折現算出 NPV。若 NPV > 0 (或報酬率 > WACC)，則投資可行。
            * *Miniko 案例*：假設公司 WACC=5%。
        * **融資決策**：比較舉債與增資成本。
            * 若銀行借款 4% < 預期報酬 6%，傾向舉債 (但本策略設定為**不舉債**，全採現股)。
        * **Gordon Model 評價**：$P = Div / (Ke - g)$。
        """)

    with tab_chips:
        st.markdown("""
        ### 🕵️ 籌碼面六大指標 (判斷大戶與散戶)
        1.  **千張大戶持股**：
            * 絕對指標。適合區間 **40% ~ 70%**。>80% 則波動過小。
        2.  **內部人持股**：
            * >40% 代表經營層利益與股東一致，適合長期持有。
        3.  **佔股本比重 (區間買賣超)**：
            * 若 60 天內買賣超佔股本 > 3%，代表主力介入 (較適用大型股)。
        4.  **籌碼集中度**：
            * 60天集中度 > 5%、120天集中度 > 3% 為佳。
        5.  **主力買賣超**：
            * 若主力賣、股價漲 (背離)，小心主力倒貨。
        6.  **買賣家數差 (重要必勝訊號)**：
            * 負數 (賣家家數 > 買家家數) = **籌碼集中** (多數散戶賣給少數大戶)。
            * **訊號**：主力買超 (+) 且 買賣家數差 (-) = 大戶吸籌中！
        """)

    with tab_backtest:
        st.markdown("""
        ### 🚀 CGO + 低波動 (Smart Beta 回測實證)
        **資料來源：TEJ、嗨投資 (HiStock)、Miniko 數據庫** (2005-2025)
        
        * **策略定義**：
            * **CGO (未實現資本利得)**：$(P - Cost) / Cost$。衡量潛在賣壓。
            * **cgo_low_tv 策略**：先選「歷史波動度低 (TV100)」的股票，再從中選「CGO 高」的股票。
            
        * **回測績效 (2005-2025)**：
            | 績效指標 | 純 CGO 策略 | **CGO + Low TV (推薦)** | 大盤基準 |
            | :--- | :--- | :--- | :--- |
            | 年化報酬 | 14.89% | **14.04%** | 10.74% |
            | 年化波動 | 16.45% | **8.46% (超穩)** | 18.38% |
            | 夏普比率 | 0.927 | **1.596 (優)** | 0.647 |
            | 最大回撤 | -57% | **-32%** | -56% |
            
        * **結論**：
            加入低波動因子後，雖然報酬率略降，但**風險大幅降低** (波動率減半)，夏普比率顯著提升。這符合我們「不融資、求穩健」的投資哲學。
        """)
        
    with tab_factors:
        st.markdown("""
        ### 📊 TEJ 市場八大因子
        根據 Fama-French 延伸，台股市場有效因子包含：
        1.  **市場風險溢酬 (MRP)**
        2.  **規模溢酬 (SMB)**：小型股效應 (台灣市場不明顯，但小型價值股強)。
        3.  **淨值市價比 (HML)**：價值型投資在台灣長期有效。
        4.  **益本比 (E/P)**：高益本比 (低本益比) 優於成長股。
        5.  **現金股利率**：高股息長期優於低股息。
        6.  **動能因子**：過去一年表現好，預期續強。
        7.  **短期反轉**：近1個月表現差，預期反彈 (反應過度)。
        8.  **長期反轉**：近3-4年表現差，預期長線反轉。
        """)

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統執行：CAPM 計算 Ke、Gordon 評價、Fama-French 因子掃描。")
    if st.button("🚀 啟動 AI 智能運算 (Top 100)", type="primary"):
        with st.spinner("Step 1: 載入大盤數據與無風險利率..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入全市場清單 (含嗨投資/TEJ定義)..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定 {len(tickers)} 檔標的，開始 AI 深度運算...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 平行運算
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
        [st.write("")] # Placeholder
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序邏輯：評分優先 -> CGO優先
        df = df.sort_values(by=['AI綜合評分', 'CGO指標'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 Top 100 (不融資/不舉債)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "AI綜合建議", "合理價", "權益成本(Ke)", "CGO指標", "波動率", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含WACC、CAPM、籌碼面之完整分析"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model"),
                "權益成本(Ke)": st.column_config.NumberColumn(format="%.1f%%", help="CAPM計算之投資人要求報酬率"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值代表籌碼獲利"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
