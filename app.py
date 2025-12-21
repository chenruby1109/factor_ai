import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock
from bs4 import BeautifulSoup # 新增：用於爬取嗨投資數據

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 (調整為現貨思維) ---
RF = 0.015  # 無風險利率 (Risk-Free Rate, 如定存)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 股利長期成長率

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

@st.cache_data(ttl=3600)
def get_histock_data(stock_code):
    """
    【新增數據源：嗨投資 HiStock】
    嘗試爬取該股票在 HiStock 的基本資料 (如產業或殖利率補充)
    """
    try:
        code = stock_code.split('.')[0]
        url = f"https://histock.tw/stock/{code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=3)
        
        data = {"HiStock_Yield": None, "HiStock_Industry": ""}
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # 嘗試抓取殖利率 (範例邏輯，視網頁結構而定)
            # 這裡簡單抓取網頁標題或特定區塊作為示範
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                content = meta_desc.get('content', '')
                if '殖利率' in content:
                    # 簡單解析字串中的殖利率
                    parts = content.split('殖利率')
                    if len(parts) > 1:
                        # 嘗試提取數字
                        data["HiStock_Info"] = "已連結"
            
            # 抓取產業分類 (通常在麵包屑導航)
            breadcrumbs = soup.find_all('li', class_='breadcrumb-item')
            if breadcrumbs and len(breadcrumbs) > 1:
                data["HiStock_Industry"] = breadcrumbs[-1].text.strip()
                
        return data
    except:
        return {"HiStock_Yield": None, "HiStock_Industry": "N/A"}

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.2 旗艦運算核心 - 現貨 + HiStock 版】
    整合 CAPM, Fama-French, CGO, Smart Beta
    新增：HiStock 數據整合、Gordon Model 詳細參數
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

        # 1. 取得 HiStock 補充資料 (新增)
        histock_info = get_histock_data(ticker_symbol)

        # --- 2. CAPM (權益資金成本 Ke) ---
        # 投資決策：用以計算 WACC，將未來現金流量折現
        # Ke = Rf + Beta * (Rm - Rf)
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        ke = RF + beta * MRP # 投資人要求報酬率
        
        # --- 3. Gordon Model (股利折現) ---
        # P = Div / (Ke - g)
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        fair_value = np.nan
        # 保護機制：避免分母過小或為負
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 4. Fama-French 三因子邏輯模擬 ---
        # SMB (規模): 預期小型股報酬率高於大型股
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = market_cap > 0 and market_cap < 50000000000 
        
        # HML (價值): 預期高淨值市價比(低PB)優於低淨值市價比
        pb = ticker_info.get('priceToBook', 0)
        is_value_stock = pb > 0 and pb < 1.5
        
        # --- 5. Smart Beta: CGO (未實現獲利) + Low Vol ---
        # CGO Proxy: (現價 - 成本) / 成本。
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        # 波動率 (Volatility)
        volatility = stock_returns.std() * (252**0.5)
        
        # 策略標籤：CGO + Low Vol (Miniko cgo_low_tv)
        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波優選") 
        
        # --- 6. AI 綜合評分系統 (V9.2) ---
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
            factors.append("🐟中小型股")
            
        # 成長/動能因子
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 15
            factors.append("📈高成長")
            
        # 技術面動能
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        if current_price > ma20:
            score += 10 
        else:
            score -= 5 

        # 品質因子 (Quality)
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15:
            score += 15
            factors.append("👑高ROE")
            
        # 風險控制 (Low Vol)
        if volatility < 0.25:
            score += 15
            factors.append("🛡️低波動")
        elif volatility > 0.5:
            score -= 10
            
        # 買點計算
        buy_suggestion = ma20 
        buy_note = "MA20支撐"
        
        if not np.isnan(fair_value) and fair_value < ma20:
            buy_suggestion = fair_value
            buy_note = "合理價支撐"

        # 篩選門檻
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
                "產業(HiStock)": histock_info.get("HiStock_Industry", ""),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.2", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.2 (含 HiStock 數據源)")
st.markdown("""
本系統整合 **CAPM、APT、Fama-French 三因子** 與 **Gordon 模型**。
**【V9.2 更新】** 加入 **HiStock (嗨投資)** 數據連結與 **NPV/WACC** 決策教學。
""")

# --- 知識庫 Expander (整合您的教學內容) ---
with st.expander("📚 Miniko 專屬：投資理論與籌碼面教學資料庫"):
    
    theory_tab1, theory_tab2, theory_tab3, theory_tab4 = st.tabs([
        "籌碼面六大指標", "資產定價與三因子", "CGO與低波動策略(Smart Beta)", "財務決策(NPV/WACC)"
    ])
    
    with theory_tab1:
        st.markdown("""
        ### 🕵️ 籌碼面六大指標 (判斷是否為籌碼集中股)
        
        **前三個是「絕對指標」(大戶是否擁抱)：**
        1. **千張大戶持股**：
           - < 40%：籌碼不集中。
           - > 80%：過於集中，波動小。
           - **最佳區間：40% ~ 70%** (人數越少越有炒作優勢)。
        2. **內部人持股**：
           - > 40% 算高。代表老闆認真做事，利益與股東一致。
        3. **佔股本比重 (區間買賣超)**：
           - 若 60 天內買賣超佔股本 > 3%，代表有主力介入 (較適用大型股)。
           
        **後三個是「相對指標」(尋找明日之星)：**
        4. **籌碼集中度 (%)**：
           - 60天集中度 > 5% 為佳。
           - 120天集中度 > 3% 為佳。
           - 單日 > 20% 代表特定人收集。
        5. **主力買賣超**：
           - 觀察主力是否在買。
           - **注意**：若主力賣、股價漲 (買回自己賣的高價股)，可能是為了拉高出貨。
        6. **買賣家數差**：
           - 負數 (賣家家數 > 買家家數) = **籌碼集中** (多數人賣給少數人)。
           - **必勝訊號**：主力買超 (+) 且 買賣家數差 (-) = 大戶正在吸籌！
        """)

    with theory_tab2:
        st.markdown("""
        ### 📈 現代投資組合理論與定價模型
        
        #### 1. 資本資產定價模式 (CAPM)
        * 公式：$E(R_i) = R_f + \\beta(R_m - R_f)$
        * **用途**：計算 **權益資金成本 (Ke)**，即投資人要求的最低報酬率。
        * $R_f$：無風險利率 (如定存)。
        * $R_m - R_f$：市場風險溢酬 (MRP)。
        
        #### 2. 套利定價模式 (APT)
        * 由 Ross 提出 (1976)。
        * 主張報酬率由**多個系統因子**決定 (如通膨、利差、GNP等)，而非單一 Beta。
        * 公式：$E(R_i) = \\beta_0 + \\sum \\beta_i F_i$
        
        #### 3. Fama & French 三因子模式 (FF3)
        * CAPM 的 $\\beta$ 解釋力不足，FF 加入兩個因子：
            1.  **MRP (市場風險)**
            2.  **SMB (規模溢酬)**：預期小型股報酬 > 大型股。
            3.  **HML (淨值市價比溢酬)**：預期價值股 (高B/P) > 成長股。
        * **TEJ 八大因子**：延伸包含益本比、現金股利率、動能、反轉等。
        * **實證**：在台灣市場，**益本比**與**現金股利率**區分的價值型投資長期有效。
        """)

    with theory_tab3:
        st.markdown("""
        ### 🚀 Smart Beta：CGO + 低波動策略
        
        #### 什麼是多因子選股？
        結合基本面、技術面、動能、風險等多個指標。單一因子易受市場週期影響，多因子可提高穩定性。
        
        #### Miniko 精選策略：CGO + Low Vol (cgo_low_tv)
        本策略採用「序貫排序 (Sequential Sort)」法：
        1.  **第一步 (Risk Filter)**：篩選 **歷史波動度 (TV100)** 最低的 10% 股票 (籌碼穩定)。
        2.  **第二步 (Alpha Selection)**：從中選取 **CGO (未實現獲利)** 最高的 50 檔。
        
        #### 回測績效 (2005~2025)
        | 策略 | 年化報酬 | 波動率 | 夏普比率 | 最大回撤 | Alpha |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        | **cgo_low_tv** | **14.04%** | **8.46%** | **1.596** | -32.91% | 0.096 |
        | 大盤 (Benchmark) | 10.74% | 18.38% | 0.647 | -56.02% | - |
        
        **結論**：加入低波動篩選後，雖然報酬率略低於純 CGO 策略，但**風險大幅降低**，夏普比率(CP值)顯著提升。
        """)
        
    with theory_tab4:
        st.markdown("""
        ### 💰 財務決策與評價模型
        
        #### Gordon Model (高登模型)
        * 用於評估合理股價。
        * 公式：$P = \\frac{Div}{K_e - g}$
        * **範例**：假設每年發股利 3 元，預期報酬率 ($K_e$) 6%，則合理股價 = $3 / 0.06 = 50$ 元。
        
        #### 投資決策：NPV (淨現值)
        * 計算 WACC (加權平均資金成本) 後，將未來現金流折現。
        * **您的範例計算** (假設 WACC=5%?):
            * **A方案** (平均流): 1000, 1000, 1000, 1000 -> NPV = 3545.95
            * **B方案** (波動流): 1000, 500, 1500, 1000 -> NPV = 3524.35
            * **決策**：A 方案 NPV 較高，應優先選擇。
            
        #### 融資決策
        * 比較舉債成本 vs. 增資成本 (權益成本)。
        * **範例**：銀行借款利率 4% < 預期報酬率(權益成本) 6%。
        * **決策**：應傾向 **舉債** (成本較低)。
        """)
        
        # 簡易 NPV 計算機
        st.markdown("---")
        st.write("#### 🧮 簡易 NPV 計算機")
        col_cal1, col_cal2 = st.columns(2)
        with col_cal1:
            rate = st.number_input("折現率 (WACC) %", value=5.0) / 100
        with col_cal2:
            flows = st.text_input("未來現金流 (逗號分隔)", "1000, 1000, 1000, 1000")
        
        if flows:
            try:
                cf_list = [float(x.strip()) for x in flows.split(',')]
                npv = sum([cf / ((1+rate)**(i+1)) for i, cf in enumerate(cf_list)])
                st.write(f"**計算結果 NPV:** :red[{npv:.2f}]")
            except:
                st.write("請輸入正確格式")

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統將執行 AI 綜合評估，整合 Gordon Model 合理價與 HiStock 產業資訊。")
    if st.button("🚀 啟動 AI 智能掃描 (Top 100)", type="primary"):
        with st.spinner("Step 1: 計算市場風險參數 (Beta/MRP)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入股票清單 & 連線 HiStock..."):
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
        # 1. 根據 AI 綜合評分從高到低排序
        # 2. 如果分數相同，優先選擇有 CGO 策略標籤的
        df['SortKey'] = df['策略標籤'].apply(lambda x: 100 if "CGO" in x else 0)
        df['TotalScore'] = df['AI綜合評分'] + df['SortKey']
        
        # 取前 100 名
        df_top100 = df.sort_values(by=['TotalScore', 'AI綜合評分'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 推薦優先買入 Top 100 ({len(df_top100)} 檔)")
        
        st.dataframe(
            df_top100,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "建議買點", "買點說明", "合理價", "策略標籤", "產業(HiStock)", "CGO指標", "波動率", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(help="股票代碼"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100, help="綜合基本面與技術面的AI評分"),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="根據技術支撐(MA20)或合理價計算的建議掛單點"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model: Div / (Ke - g)"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值代表多數人獲利(支撐強)"),
                "波動率": st.column_config.NumberColumn(format="%.2f", help="越低代表籌碼越穩定 (Low Vol Strategy)"),
                "產業(HiStock)": st.column_config.TextColumn(help="來自嗨投資的產業分類"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
