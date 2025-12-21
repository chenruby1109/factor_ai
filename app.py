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

# --- 全局參數 (調整為現貨思維) ---
RF = 0.015  # 無風險利率 (Risk-Free Rate, 如定存)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 股利長期成長率

# --- 核心功能函數 (完全保留原有邏輯) ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

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
    """【V8.3 價格修復版】(History + Realtime 雙重驗證)"""
    price = None
    # 策略 1: yfinance History
    try:
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    # 策略 2: twstock Realtime
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
    【Miniko V9.1 旗艦運算核心 - 現貨版】
    整合 CAPM, Fama-French, CGO, Smart Beta
    """
    try:
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None 

        # --- 1. CAPM ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        
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

        # --- 3. Fama-French Logic ---
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = market_cap > 0 and market_cap < 50000000000 
        
        pb = ticker_info.get('priceToBook', 0)
        is_value_stock = pb > 0 and pb < 1.5
        
        # --- 4. Smart Beta: CGO + Low Vol ---
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        volatility = stock_returns.std() * (252**0.5)
        
        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波優選")
        
        # --- 5. AI Score ---
        score = 0.0
        factors = []
        
        if is_value_stock:
            score += 15
            factors.append("💎價值型(低PB)")
        if not np.isnan(fair_value) and fair_value > current_price:
            score += 20
            factors.append("💰低於Gordon合理價")
            
        if is_small_cap:
            score += 10
            factors.append("🐟中小型股")
            
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 15
            factors.append("📈高成長")
            
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        if current_price > ma20:
            score += 10
        else:
            score -= 5 

        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append("👑高ROE")
            
        if volatility < 0.25:
            score += 15
            factors.append("🛡️低波動")
        elif volatility > 0.5:
            score -= 10
            
        # --- 6. Buy Suggestion ---
        buy_suggestion = ma20 
        buy_note = "MA20支撐"
        
        if not np.isnan(fair_value) and fair_value < ma20:
            buy_suggestion = fair_value
            buy_note = "合理價支撐"

        if score >= 50:
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "建議買點": float(buy_suggestion),
                "買點說明": buy_note,
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "波動率": volatility,
                "CGO指標": round(cgo_val * 100, 1),
                "策略標籤": " ".join(strategy_tags),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面與新增內容 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.2 Plus", layout="wide")

# Sidebar: 外部連結與計算機
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3310/3310624.png", width=100)
    st.title("Miniko 戰情室工具箱")
    
    st.markdown("### 🔗 外部資源")
    st.link_button("前往 嗨投資 (HiStock)", "https://histock.tw/")
    
    st.markdown("---")
    st.markdown("### 🧮 財務決策模擬")
    st.info("假設公司資金成本(WACC) = 5%")
    
    # NPV 計算機 (依照您的範例)
    st.markdown("**NPV 案例試算**")
    wacc_input = st.number_input("資金成本率 (%)", value=5.0, step=0.1) / 100
    
    # 方案 A
    cf_a = [1000, 1000, 1000, 1000]
    npv_a = sum([cf / ((1+wacc_input)**(i+1)) for i, cf in enumerate(cf_a)])
    
    # 方案 B
    cf_b = [1000, 500, 1500, 1000]
    npv_b = sum([cf / ((1+wacc_input)**(i+1)) for i, cf in enumerate(cf_b)])
    
    st.write(f"🅰️ 方案A NPV: **{npv_a:.2f}**")
    st.write(f"🅱️ 方案B NPV: **{npv_b:.2f}**")
    
    if npv_a > npv_b:
        st.success("建議選擇：方案 A")
    else:
        st.success("建議選擇：方案 B")
        
    st.markdown("---")
    st.markdown("**融資決策判斷**")
    st.text("銀行借款利率: 4%")
    st.text("預期投資報酬率: 6%")
    st.caption("決策：應傾向舉債(Leverage)而非增資，因為借款成本(4%) < 報酬(6%)。")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.2 Plus")
st.markdown("### 整合 CAPM、Fama-French、CGO 與 籌碼面分析的 AI 決策系統")

# --- 新增：Miniko 投資學院 (教學與理論區) ---
with st.expander("📚 點此進入：Miniko 投資學院 (理論、籌碼、策略)", expanded=False):
    
    course_tab1, course_tab2, course_tab3, course_tab4 = st.tabs([
        "💰 金融理論與定價模型", 
        "🕵️ 籌碼面六大指標", 
        "📈 Fama-French 與八大因子", 
        "🚀 CGO 與 Smart Beta 策略"
    ])
    
    # --- TAB 1: 金融理論 ---
    with course_tab1:
        st.header("一、投資組合理論與定價模型")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("1. 投資組合理論 (MPT)")
            st.markdown("""
            **Markowitz (1950s) 多角化概念量化：**
            * 核心概念：透過多角化投資，當資產數量(n)增加，非系統性風險($\sigma$)下降。
            * 
            * **案例：船運公司風險分散**
                * **情境**：總運費200萬，貨物價值1000萬，出事機率0.1。
                * **2艘大船**：風險(標準差) $\sigma = 260.22$
                * **10艘小船**：風險(標準差) $\sigma = 116.62$
                * **結論**：拆成多艘小船運輸（多角化），預期利潤相同，但風險顯著降低。
            """)
            
            st.subheader("2. 資本資產定價模式 (CAPM)")
            st.latex(r"E(R_i) = R_f + \beta \times (R_m - R_f)")
            st.markdown("""
            * $R_f$：無風險利率 (如定存)
            * $R_m - R_f$：市場風險溢酬 (MRP)
            * $\beta$：系統性風險係數
            * **應用**：計算 **Ke (權益資金成本)**，作為投資人要求的最低回報率。
            * **批評**：假設市場完美、投資人理性，且無法解釋「規模效應」或「價值效應」。
            """)
            st.write("")

        with col_t2:
            st.subheader("3. 套利定價模式 (APT)")
            st.markdown("""
            **Ross (1976)** 提出。主張個股報酬受「多個」系統因子影響，而非僅有市場風險。
            * 公式：$E(R_i) = \beta_0 + \Sigma \beta_i \times F_i$
            * 因子包含：通膨、利差、工業生產指數等。
            * **特點**：利用套利行為達成市場均衡。
            """)
            
            st.subheader("4. 評價模型：Gordon Model")
            st.latex(r"P = \frac{Div}{K - g}")
            st.markdown("""
            * **應用**：股利折現模型，計算合理股價。
            * **範例**：
                * 每年發放股利 3 元
                * 預期報酬率 (K) 6%
                * 合理股價 = $3 / 0.06 = 50$ 元。
            """)

    # --- TAB 2: 籌碼面 ---
    with course_tab2:
        st.header("二、籌碼面分析：判斷大戶動向")
        st.markdown("可以用這 **6項指標** 來看這一檔股票是否是「籌碼集中股」！")
        
        st.info("前三個是「絕對指標」 (判斷是否已被大戶擁抱)")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 1. 千張大戶持股")
            st.markdown("""
            * **< 40%**：籌碼不集中。
            * **> 80%**：過於集中，股價難有波動。
            * **40% ~ 70%**：**最佳交易區間**。
            * 人數越少，越有炒作優勢。
            """)
        with c2:
            st.markdown("#### 2. 內部人持股")
            st.markdown("""
            * **> 40%**：算高。
            * 代表老闆與股東利益一致。
            * 不易暴漲暴跌，適合長期持有，隨公司獲利成長。
            """)
        with c3:
            st.markdown("#### 3. 佔股本比重")
            st.markdown("""
            * **定義**：區間買賣超佔股本比重。
            * **訊號**：60天內買賣超佔股本 **> 3%**。
            * 代表有主力大戶介入 (較適用於大型股)。
            """)
            
        st.info("後三個是「相對指標」 (發現明日之星)")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown("#### 4. 籌碼集中度")
            st.markdown("""
            * 60天集中度：**> 5%** 為佳。
            * 120天集中度：**> 3%** 為佳。
            * 單日集中度：**> 20%** 可能有特定人收集。
            """)
        with c5:
            st.markdown("#### 5. 主力買賣超")
            st.markdown("""
            * **正常**：主力買、股價漲。
            * **警訊**：主力賣、股價漲 (可能是主力倒貨給散戶，自己左手賣右手炒高)。
            * 需搭配「買賣家數差」一起看。
            """)
        with c6:
            st.markdown("#### 6. 買賣家數差")
            st.markdown("""
            * **負數 (賣家 > 買家)**：代表**籌碼集中** (多數散戶賣給少數大戶)。
            * **必勝訊號**：
                1. 主力買超 (+)
                2. 買賣家數差 (-) 
                3. **大戶正在吸籌！**
            """)

    # --- TAB 3: Fama-French ---
    with course_tab3:
        st.header("三、Fama-French 三因子與多因子模型")
        st.markdown("")
        st.markdown("""
        **Fama & French (1992)** 發現 $\\beta$ 對報酬解釋力不足，因此加入規模與價值因子：
        $$E(R_i) = R_f + \beta_1(MRP) + \beta_2(SMB) + \beta_3(HML)$$
        """)
        
        st.subheader("三大核心因子")
        st.markdown("""
        1.  **市場風險 (MRP)**：$R_m - R_f$。
        2.  **規模溢酬 (SMB - Small Minus Big)**：長期來看，**小型股**報酬率高於大型股。
        3.  **價值溢酬 (HML - High Minus Low)**：**高淨值市價比 (Value)** 股票報酬優於成長股。
        """)
        
        st.subheader("TEJ 八大因子體系")
        st.table(pd.DataFrame({
            "因子名稱": ["市場風險", "規模", "淨值市價比", "益本比", "現金股利率", "動能", "短期反轉", "長期反轉"],
            "投資邏輯": [
                "承擔市場波動的補償",
                "小型股具爆發力 (SMB)",
                "價值型投資 (HML)",
                "高益本比 (便宜) 優於低益本比",
                "高殖利率保護",
                "強者恆強 (近1年表現好)",
                "跌深反彈 (近1月表現差)",
                "長期回歸均值 (近3-4年表現差)"
            ]
        }))
        
        st.subheader("台灣市場實證結論 (1995-2009)")
        st.success("""
        * **價值型投資有效**：以「益本比」及「現金股利」區分效果最佳。
        * **小型價值股最強**：過去曾創造近 3 倍報酬 (年化約 10%)。
        * **反轉效應**：台灣市場在短期與長期皆有「反應過度」現象 (適合反向操作)。
        """)

    # --- TAB 4: Smart Beta & CGO ---
    with course_tab4:
        st.header("四、Smart Beta 與 CGO 策略")
        st.markdown("""
        **Smart Beta** 是介於主動與被動之間的策略，透過選取特定因子 (Factor) 來獲取超額報酬 (Alpha)。
        """)
        
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            st.subheader("CGO (Capital Gains Overhang)")
            st.markdown("""
            * **定義**：未實現資本利得。
            * **公式概念**：$(現價 - 參考成本) / 參考成本$。
            * **行為財務學意義**：
                * 當 CGO 高 (大家都在賺錢)：投資人惜售，賣壓小，支撐強。
                * 當 CGO 低 (大家都在賠錢)：解套賣壓重。
            """)
        with col_sb2:
            st.subheader("低波動 (Low Volatility)")
            st.markdown("""
            * **現象**：長期而言，低波動股票的「風險調整後報酬」優於高波動股票。
            * **原因**：避免大幅回撤 (Max Drawdown)，複利效果更佳。
            """)
            
        st.markdown("---")
        st.subheader("🏆 Miniko 推薦策略：CGO + Low Vol (序貫排序法)")
        st.markdown("""
        **策略邏輯 (cgo_low_tv)：**
        1.  **第一步 (Filter)**：先篩選全市場 **歷史波動度最低** 的 10% 股票 (剔除高風險雜訊)。
        2.  **第二步 (Select)**：在低波動池中，買入 **CGO 值最高** (籌碼最穩定、獲利中) 的 50 檔。
        
        **回測績效 (2005-2025/06)：**
        * **年化報酬率**：14.04% (優於大盤 10.74%)
        * **波動率**：降至 8.46% (大盤為 18.38%)
        * **夏普比率 (Sharpe Ratio)**：**1.596** (顯著優於純 CGO 的 0.927)
        * **結論**：低波動篩選能有效「提純」CGO 因子的獲利能力，降低 Beta，提升 Alpha。
        """)
        st.line_chart(pd.DataFrame({'Strategy': [100, 114, 130, 145, 1281], 'Market': [100, 110, 120, 115, 668]}, index=[2005, 2010, 2015, 2020, 2025]))
        st.caption("示意圖：策略累積報酬率 vs 大盤 (參考數據)")

# --- 主程式區 (保留原有掃描邏輯) ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統將執行 AI 綜合評估，篩選全市場最值得買入的現貨標的。")
    if st.button("🚀 啟動 AI 智能掃描 (Top 100)", type="primary"):
        with st.spinner("Step 1: 計算市場風險參數 (Beta/MRP)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"開始分析 {len(tickers)} 檔股票的財務因子...")
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
                    status_text.text(f"AI 分析中: {completed}/{len(tickers)}")
                if data:
                    st.session_state['results'].append(data)

        status_text.text("✅ AI 分析完成！")

with col2:
    if not st.session_state['results']:
        st.write("👈 點擊按鈕開始分析。")
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # --- AI 篩選邏輯 ---
        df['SortKey'] = df['策略標籤'].apply(lambda x: 100 if "CGO" in x else 0)
        df['TotalScore'] = df['AI綜合評分'] + df['SortKey']
        
        # 取前 100 名
        df_top100 = df.sort_values(by=['TotalScore', 'AI綜合評分'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 推薦優先買入 Top 100 ({len(df_top100)} 檔)")
        
        st.dataframe(
            df_top100,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "建議買點", "買點說明", "合理價", "策略標籤", "CGO指標", "波動率", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(help="股票代碼"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100, help="綜合基本面與技術面的AI評分"),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="根據技術支撐(MA20)或合理價計算的建議掛單點"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model 計算之合理股價"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值代表多數人獲利(支撐強)"),
                "波動率": st.column_config.NumberColumn(format="%.2f", help="越低代表籌碼越穩定"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
