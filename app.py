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



# --- 全局參數 (根據使用者提供資料調整) ---

RF = 0.015  # 無風險利率 (Risk-Free Rate, 如定存)

MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)

G_GROWTH = 0.02 # 股利長期成長率

COST_OF_DEBT = 0.04 # 假設平均借款利率 (4%)



# --- 核心功能函數 ---



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

        # 示範抓取 twstock 內建清單 (全市場掃描建議分批)

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

    # 策略 1: yfinance History (適合盤後/週末)

    try:

        ticker = yf.Ticker(stock_code)

        hist = ticker.history(period="5d")

        if not hist.empty:

            price = float(hist['Close'].iloc[-1])

    except: pass



    # 策略 2: twstock Realtime (適合盤中)

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

    【Miniko V9.0 旗艦運算核心】

    整合 CAPM, Fama-French, CGO, Smart Beta

    """

    try:

        current_price = get_realtime_price_robust(ticker_symbol)

        if current_price is None or current_price <= 0: return None



        # 抓取 1 年數據 (用於計算波動率與 Beta)

        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)

        if len(data) < 100: return None 

        if isinstance(data.columns, pd.MultiIndex):

            data.columns = data.columns.get_level_values(0)

        

        if current_price < 10: return None # 排除雞蛋水餃股



        # --- 1. CAPM & WACC (權益資金成本) ---

        stock_returns = data['Close'].pct_change().dropna()

        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()

        aligned.columns = ['Stock', 'Market']

        

        if len(aligned) < 60: return None



        cov = aligned.cov().iloc[0, 1]

        mkt_var = aligned['Market'].var()

        beta = cov / mkt_var if mkt_var != 0 else 1.0

        

        # Ke = Rf + Beta * MRP (投資人要求報酬率)

        ke = RF + beta * MRP

        

        # 融資決策建議

        financing_msg = "增資/舉債皆可"

        if COST_OF_DEBT < ke:

            financing_msg = "建議舉債 (利率低)"

        else:

            financing_msg = "建議增資 (風險高)"



        # --- 2. Gordon Model (股利折現) ---

        ticker_info = yf.Ticker(ticker_symbol).info

        div_rate = ticker_info.get('dividendRate', 0)

        if not div_rate:

            yield_val = ticker_info.get('dividendYield', 0)

            if yield_val: div_rate = current_price * yield_val



        fair_value = np.nan

        # 保護機制：避免分母過小

        k_minus_g = max(ke - G_GROWTH, 0.015) 

        if div_rate and div_rate > 0:

            fair_value = round(div_rate / k_minus_g, 2)



        # --- 3. Fama-French 三因子邏輯模擬 ---

        # SMB (規模): 這裡簡單用市值判斷

        market_cap = ticker_info.get('marketCap', 0)

        is_small_cap = market_cap > 0 and market_cap < 50000000000 # 假設小於500億為中小型

        

        # HML (價值): 用 PB (淨值市價比的倒數) 判斷

        pb = ticker_info.get('priceToBook', 0)

        is_value_stock = pb > 0 and pb < 1.5

        

        # --- 4. Smart Beta: CGO (未實現獲利) + Low Vol ---

        # CGO Proxy: (現價 - 成本) / 成本。這裡假設過去 100 天均價為市場持倉成本

        ma100 = data['Close'].rolling(100).mean().iloc[-1]

        cgo_val = (current_price - ma100) / ma100

        

        # 波動率 (Volatility)

        volatility = stock_returns.std() * (252**0.5)

        

        # 策略標籤：CGO + Low Vol (使用者提到的 cgo_low_tv)

        strategy_tags = []

        if cgo_val > 0.1 and volatility < 0.3:

            strategy_tags.append("🔥CGO低波優選") # 獲利中且波動低

        

        # --- 5. 評分系統 (V9.0 升級版) ---

        score = 0.0

        factors = []

        

        # 價值因子 (Value)

        if is_value_stock:

            score += 15

            factors.append("💎價值型(低PB)")

        if not np.isnan(fair_value) and fair_value > current_price:

            score += 20

            factors.append("💰低於Gordon合理價")

            

        # 規模因子 (Size - SMB)

        if is_small_cap:

            score += 10

            factors.append("🐟中小型股(具爆發力)")

            

        # 成長/動能因子 (Growth/Momentum)

        rev_growth = ticker_info.get('revenueGrowth', 0)

        if rev_growth > 0.2:

            score += 15

            factors.append("📈高成長")

            

        ma20 = data['Close'].rolling(20).mean().iloc[-1]

        if current_price > ma20:

            score += 10 # 短期動能



        # 品質因子 (Quality)

        roe = ticker_info.get('returnOnEquity', 0)

        if roe > 0.15:

            score += 15

            factors.append("👑高ROE")

            

        # 風險控制 (Low Vol)

        if volatility < 0.25:

            score += 15

            factors.append("🛡️低波動(籌碼穩)")

        elif volatility > 0.5:

            score -= 10

            

        if score >= 50:

            return {

                "代號": ticker_symbol,

                "名稱": name_map.get(ticker_symbol, ticker_symbol),

                "現價": float(current_price),

                "評分": round(score, 1),

                "合理價": fair_value if not np.isnan(fair_value) else None,

                "資金成本(Ke)": ke,

                "融資建議": financing_msg,

                "波動率": volatility,

                "CGO指標": round(cgo_val * 100, 1),

                "策略標籤": " ".join(strategy_tags),

                "亮點": " | ".join(factors)

            }

    except:

        return None

    return None



# --- Streamlit 介面 ---



st.set_page_config(page_title="Miniko 投資戰情室 V9.0", layout="wide")



st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.0 (旗艦策略版)")

st.markdown("""

本系統整合 **CAPM、Fama-French 三因子、Gordon 模型** 與 **Smart Beta (CGO+低波動)** 策略。

""")



# --- 知識庫 Expander ---

with st.expander("📚 點此查看：投資理論與籌碼面分析教學 (Miniko 專屬)"):

    tab1, tab2, tab3 = st.tabs(["籌碼面六大指標", "Fama-French與多因子", "CGO與低波動策略"])

    

    with tab1:

        st.markdown("""

        ### 🕵️ 籌碼面六大指標 (判斷大戶動向)

        1. **千張大戶持股**：>40% 代表集中，>80% 過於集中波動小。適合區間 40%~70%。

        2. **內部人持股**：>40% 算高，代表老闆與股東利益一致，適合長期持有。

        3. **佔股本比重**：區間買賣超佔股本 >3%，代表有主力介入 (較適用大型股)。

        4. **籌碼集中度**：

           - 60天集中度 > 5% 為佳

           - 120天集中度 > 3% 為佳

        5. **主力買賣超**：與股價同步為正常；若主力賣、股價漲，小心是主力倒貨給散戶。

        6. **買賣家數差**：

           - 負數 (賣家 > 買家) = **籌碼集中** (多數人賣給少數人)。

           - **必勝訊號**：主力買超 (+) 且 買賣家數差 (-) = 大戶正在吸籌！

        """)

    

    with tab2:

        st.markdown("""

        ### 📈 Fama-French 三因子與多因子模型

        * **CAPM 模型**：$E(R_i) = R_f + \\beta(R_m - R_f)$。

            - 應用：計算 **Ke (權益資金成本)**。若 Ke (e.g. 6%) > 銀行借款利率 (4%)，公司應選擇 **舉債**。

        * **Fama-French 三因子**：除了市場風險，還加入：

            - **SMB (規模)**：小型股通常報酬高於大型股 (Small Minus Big)。

            - **HML (價值)**：高淨值市價比(價值股) 通常優於成長股。

        * **八大因子**：包含 動能、反轉、股利率、波動率等。

        """)

        

    with tab3:

        st.markdown("""

        ### 🚀 CGO + 低波動 (Smart Beta 策略)

        * **CGO (未實現資本利得)**：衡量市場上的「潛在賣壓」或「惜售心理」。

        * **低波動 (Low Vol)**：長期來看，低波動股票的風險調整後報酬往往優於高波動股票。

        * **Miniko 精選策略 (cgo_low_tv)**：

            1. 先篩選 **歷史波動度低** 的股票 (籌碼穩定)。

            2. 再從中選 **CGO 高** (大部分持股者都賺錢，惜售) 的股票。

            - **回測結果**：年化報酬與夏普比率顯著提升，Beta 降低，Alpha 提升。

        """)



# --- 主程式區 ---

if 'results' not in st.session_state:

    st.session_state['results'] = []



col1, col2 = st.columns([1, 4])



with col1:

    st.info("💡 系統將計算 WACC 融資決策與 CGO 策略指標。")

    if st.button("🚀 啟動 V9.0 全面掃描", type="primary"):

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

                    status_text.text(f"分析中: {completed}/{len(tickers)}")

                if data:

                    st.session_state['results'].append(data)



        status_text.text("✅ 分析完成！")



with col2:

    if not st.session_state['results']:

        st.write("👈 點擊按鈕開始分析。")

    else:

        df = pd.DataFrame(st.session_state['results'])

        st.subheader(f"🎯 策略選股結果 ({len(df)} 檔)")

        

        # 排序：優先顯示 CGO 策略股，其次按評分

        df['SortKey'] = df['策略標籤'].apply(lambda x: 1 if "CGO" in x else 0)

        df = df.sort_values(by=['SortKey', '評分'], ascending=[False, False])

        

        st.dataframe(

            df,

            use_container_width=True,

            hide_index=True,

            column_order=["名稱", "現價", "合理價", "評分", "策略標籤", "CGO指標", "資金成本(Ke)", "融資建議", "波動率", "亮點"],

            column_config={

                "現價": st.column_config.NumberColumn(format="$%.2f"),

                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model 計算之合理股價"),

                "評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),

                "資金成本(Ke)": st.column_config.NumberColumn(format="%.2f%%", help="CAPM 計算之股東要求報酬率"),

                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值代表多數人獲利(支撐強)，負值代表多數人虧損(賣壓重)"),

                "波動率": st.column_config.NumberColumn(format="%.2f", help="越低代表籌碼越穩定"),

                "亮點": st.column_config.TextColumn(width="medium"),

            }

        )
