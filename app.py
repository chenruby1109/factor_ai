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
    【Miniko V9.4 旗艦運算核心 - 同步決策版】
    整合 CAPM, Fama-French, CGO, Smart Beta
    V9.4 修正：確保「建議買點」與「AI決策」邏輯一致，消除落差感。
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

        # --- 1. CAPM (權益資金成本 - 僅作評估用) ---
        stock_returns = data['Close'].pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: return None

        cov = aligned.cov().iloc[0, 1]
        mkt_var = aligned['Market'].var()
        beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        # Ke = Rf + Beta * MRP (投資人要求報酬率)
        ke = RF + beta * MRP
        
        # --- 2. Gordon Model (股利折現 - 用於計算合理價) ---
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
        # SMB (規模)
        market_cap = ticker_info.get('marketCap', 0)
        is_small_cap = market_cap > 0 and market_cap < 50000000000 # 假設小於500億為中小型
        
        # HML (價值)
        pb = ticker_info.get('priceToBook', 0)
        is_value_stock = pb > 0 and pb < 1.5
        
        # --- 4. Smart Beta: CGO (未實現獲利) + Low Vol ---
        # CGO Proxy: (現價 - 成本) / 成本。這裡假設過去 100 天均價為市場持倉成本
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        # 波動率 (Volatility)
        volatility = stock_returns.std() * (252**0.5)
        
        # 策略標籤：CGO + Low Vol (Miniko cgo_low_tv)
        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波優選") # 獲利中且波動低
        
        # --- 5. AI 綜合評分系統 (V9.4 升級版) ---
        score = 0.0
        factors = []

        # 準備多週期均線 (用於評分與買點計算)
        ma5 = data['Close'].rolling(5).mean().iloc[-1]
        ma10 = data['Close'].rolling(10).mean().iloc[-1]
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
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
            factors.append("🐟中小型股(爆發力)")
            
        # 成長/動能因子 (Growth/Momentum)
        rev_growth = ticker_info.get('revenueGrowth', 0)
        if rev_growth > 0.2:
            score += 15
            factors.append("📈高成長")
            
        # 技術面動能
        if current_price > ma20:
            score += 10 # 短期多頭排列
        else:
            score -= 5  # 跌破月線扣分

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
            
        # --- 6. 建議買入點位與 AI 決策同步計算 (修正邏輯) ---
        
        # 步驟 1: 先找出「技術上的理想支撐點」(Technical Anchor)
        # 這不一定是最後的建議買點，而是作為衡量乖離率的基準
        bias_ma20 = (current_price - ma20) / ma20 
        anchor_price = ma20
        anchor_note = "MA20"

        if current_price > ma20:
            if bias_ma20 > 0.1: # 強勢噴出
                anchor_price = ma10
                anchor_note = "MA10"
            elif bias_ma20 > 0.04: # 正常多頭
                anchor_price = ma20
                anchor_note = "MA20"
            else: # 貼近月線
                anchor_price = ma20
                anchor_note = "MA20"
        else:
             # 跌破月線
            if current_price > ma60:
                anchor_price = ma60
                anchor_note = "MA60"
            else:
                anchor_price = current_price * 0.95 # 超跌緩衝
                anchor_note = "超跌區"
                if not np.isnan(fair_value) and (current_price * 0.85) < fair_value < current_price:
                     anchor_price = fair_value
                     anchor_note = "合理價"

        # 步驟 2: 計算現價與理想支撐的距離 (Gap)
        # 用來判斷要「追價(現價買)」還是「等待(掛單買)」
        gap_percent = (current_price - anchor_price) / current_price
        
        # 步驟 3: 綜合決策與定價
        ai_advice = "⏳ 觀望"
        final_buy_price = anchor_price # 預設為支撐價
        final_buy_note = f"{anchor_note}支撐"

        if score >= 60:
            # 邏輯修正：如果在支撐上方 3% 以內，視為「可進場區間」
            if gap_percent <= 0.03: 
                if score >= 80:
                    ai_advice = "🚀 強力買進"
                else:
                    ai_advice = "✅ 建議買進"
                
                # 關鍵修正：既然建議買進，買點就不能是遙遠的支撐，而是「現價」或「現價略低」
                # 這樣使用者可以直接操作，不會困惑
                final_buy_price = current_price 
                final_buy_note = f"現價進場 (防守{anchor_note})"
            
            else:
                # 距離支撐太遠 (>3%)，建議等待
                wait_percent = round(gap_percent * 100, 1)
                ai_advice = f"📉 等回檔 ({wait_percent}%)"
                # 此時買點維持在 anchor_price，告訴使用者要等
                final_buy_price = anchor_price
                final_buy_note = f"乖離大，等{anchor_note}"
        else:
             ai_advice = "😐 暫不推薦"
             final_buy_note = "評分不足"

        # 篩選門檻
        if score >= 50:
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""), 
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI決策": ai_advice, 
                "AI綜合評分": round(score, 1), 
                "建議買點": float(round(final_buy_price, 2)), # 修正後的同步買點
                "買點說明": final_buy_note, # 修正後的同步說明
                "合理價": fair_value if not np.isnan(fair_value) else None,
                "波動率": volatility,
                "CGO指標": round(cgo_val * 100, 1),
                "策略標籤": " ".join(strategy_tags),
                "亮點": " | ".join(factors)
            }
    except:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.4", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.4 (AI決策同步版)")
st.markdown("""
本系統整合 **CAPM、Fama-French 三因子、Gordon 模型** 與 **Smart Beta (CGO+低波動)** 策略。
**【V9.4 更新】** 優化決策邏輯，確保 **「AI 決策」與「建議買點」同步**，消除操作落差感。
""")

# --- 知識庫 Expander (更新內容) ---
with st.expander("📚 點此查看：投資理論與籌碼面分析教學 (Miniko 專屬)"):
    tab1, tab2, tab3 = st.tabs(["籌碼面六大指標", "Fama-French與多因子", "CGO與低波動策略"])
    
    with tab1:
        st.markdown("""
        ### 🕵️ 籌碼面六大指標 (判斷大戶動向)
        
        **前三個是「絕對指標」**：符合代表已被大戶擁抱。
        **後三個是「相對指標」**：用來發現未來的明日之星。

        #### 1. 千張大戶持股
        * **標準**：>40% 代表集中；>80% 過於集中波動小。
        * **操作**：適合區間 **40% ~ 70%** 作為交易對象。
        * **比較**：若持股比率相同，持有人數越少，炒作優勢越大。

        #### 2. 內部人持股
        * **標準**：>40% 算高。
        * **意義**：代表老闆與股東利益一致，不易暴漲但抗跌。適合長期持有，公司賺錢股票更值錢。

        #### 3. 佔股本比重 (區間買賣超)
        * **標準**：60天內買賣超佔股本 > 3%。
        * **意義**：代表有特定大戶介入 (籌碼集中到特定券商)。
        * **注意**：此指標較適用於 **大型股**，小股易被操控。

        #### 4. 籌碼集中度 (%)
        * **60天集中度**：> 5% 為佳。
        * **120天集中度**：> 3% 為佳。
        * **單日**：> 20% 代表可能有特定人在收集籌碼。

        #### 5. 主力買賣超
        * **正常**：與股價同步。
        * **異常**：主力賣、股價漲 (主力倒貨給散戶，呈現反向走勢時要小心)。
        * **應用**：讓你知道目前是主力在買還是賣。

        #### 6. 買賣家數差 (必勝訊號)
        * **定義**：賣出家數 > 買進家數 (數值為負)。
        * **解讀**：多數人賣給少數人 = **籌碼集中**。
        * **🔥 必勝訊號**：**「主力買賣超」連續買進 (+) 且 「買賣家數差」為負數 (-)** = 大戶正在吸籌！
        """)
    
    with tab2:
        st.markdown("""
        ### 📈 投資組合理論與定價模型

        #### 1. 現代投資組合理論 (MPT)
        * **核心**：多角化投資可降低風險 (n↑, σ↓)。
        * **例子**：
            * 2艘大船 (總運費200萬)：風險(σ) = 260.22
            * 10艘小船 (總運費200萬)：風險(σ) = 116.62 (風險顯著降低)

        #### 2. CAPM (資本資產定價模式) & 投資決策
        * **公式**：$E(R_i) = R_f + \\beta(R_m - R_f)$
            * $R_f$：無風險利率
            * $MRP$ ($R_m - R_f$)：市場風險溢酬
        * **應用**：計算 **Ke (權益資金成本)**，即投資人要求的報酬率。
        * **💰 投資決策 (WACC)**：計算加權資金成本，將未來現金流折現算出現值 (NPV)。
        * **🏦 融資決策**：
            * 若 銀行借款利率 (4%) < 預期報酬率 (6%) ➜ **傾向舉債** (成本較低)。
            * *註：本系統 V9.1 採現貨策略，不建議個人過度槓桿。*

        #### 3. Gordon Model (股價評價)
        * **公式**：$P = Div / (K - g)$
        * **範例**：每年發放股利 3 元，預期報酬率 (K) 6%，成長率假設 0%。
            * 合理股價 = $3 / 0.06 = 50$ 元。

        #### 4. APT (套利定價模式)
        * **Ross (1976)**：主張預期報酬與多個系統因子有關 (如通膨、利差)。

        #### 5. Fama-French 三因子模型 (FF3)
        * 修正 CAPM $\\beta$ 解釋力不足的問題，加入規模與價值因子。
        * $E(R_i) = \\beta_0 + \\beta_1(MRP) + \\beta_2(SMB) + \\beta_3(HML)$
            * **MRP**：市場風險溢酬。
            * **SMB (規模溢酬)**：小型股報酬通常優於大型股。
            * **HML (價值溢酬)**：高淨值市價比 (價值股) 優於成長股。

        #### 6. 市場多因子 (八大因子)
        * **價值**：益本比、淨值市價比 (B/P)。
        * **規模**：市值大小。
        * **動能**：過去一年表現好，預期續強。
        * **反轉**：短期 (1個月) 或長期 (3-4年) 表現差，預期反轉。
        * **波動率**：低波動通常被認為風險較低。
        * **現金股利率**：高股息。
        """)
        
    with tab3:
        st.markdown("""
        ### 🚀 Smart Beta & CGO 策略 (Miniko 精選)

        #### 什麼是 Smart Beta?
        * 介於主動與被動之間，針對特定因子 (價值、品質、低波動、動能) 進行曝險，獲取超額報酬 (Alpha)。
        * **多因子策略**：結合多個因子 (如：低波動 + 高 CGO) 來降低單一因子失效的風險。

        #### 核心策略：CGO (未實現資本利得)
        * **定義**：衡量市場上的「潛在賣壓」或「惜售心理」。
        * **邏輯**：當 CGO 高 (大家都在賺錢)，持股者傾向惜售，支撐強；當 CGO 低 (大家賠錢)，解套賣壓重。

        #### 🔥 Miniko 實戰策略：序貫排序 (Sequential Sort)
        本系統採用 **cgo_low_tv** 策略：
        1.  **第一步 (篩選)**：先選歷史波動度 (TV100) 最低的 10% 股票 (籌碼穩定)。
        2.  **第二步 (擇優)**：從中選出 CGO 值最高的 50 檔。

        #### 📊 歷史回測績效 (2005 - 2025)
        
        | 績效指標 | 純 CGO 策略 | **cgo_low_tv (本系統採用)** | Benchmark (大盤) |
        | :--- | :--- | :--- | :--- |
        | **年化報酬率** | 14.89% | **14.04%** | 10.74% |
        | **年化波動度** | 16.45% | **8.46% (風險減半)** | 18.38% |
        | **夏普比率 (Sharpe)** | 0.927 | **1.596 (最高)** | 0.647 |
        | **最大回撤 (MDD)** | -57.29% | **-32.91%** | -56.02% |
        | **Alpha** | 0.080 | **0.096** | - |
        | **Beta** | 0.631 | **0.361** | 1.0 |

        **結論**：
        * **低波動 + CGO** 能顯著提升夏普比率 (1.60)，代表在承受較低風險下獲得更高的報酬。
        * Beta 僅 0.361，代表受大盤震盪影響小，適合穩健現貨投資。
        """)

# --- 主程式區 ---
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
            column_order=["代號", "名稱", "現價", "AI決策", "AI綜合評分", "建議買點", "買點說明", "合理價", "策略標籤", "CGO指標", "波動率", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(help="股票代碼"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI決策": st.column_config.TextColumn(help="AI根據評分與乖離率給出的即時操作建議"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100, help="綜合基本面與技術面的AI評分"),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="與決策同步：建議買進時為現價；觀望時為支撐價"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="Gordon Model 計算之合理股價"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%", help="正值代表多數人獲利(支撐強)"),
                "波動率": st.column_config.NumberColumn(format="%.2f", help="越低代表籌碼越穩定"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
