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
    【Miniko V9.6 機構法人旗艦版 - 大戶多因子模型】
    新增核心：
    1. ROIC (投入資本回報率)：識破財務槓桿，尋找高效率公司。
    2. FCF Yield (自由現金流收益率)：大戶的真實估值指標。
    3. Earnings Quality (獲利品質)：檢視現金流與淨利的比例。
    """
    try:
        stock_name = name_map.get(ticker_symbol, ticker_symbol)
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 基礎數據下載
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # 取得詳細財務數據 (用於計算 ROIC, FCF 等)
        # 注意：yfinance info 請求較慢，但在單線程或少量多線程下尚可接受
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # --- 0. 基礎趨勢與意圖因子 ---
        days = 60
        close_series = data['Close']
        volume_series = data['Volume']
        
        # S: 60天報酬率 & V: 變動率
        price_60_ago = close_series.iloc[-days]
        s_return = (current_price / price_60_ago) - 1
        v_variability = close_series.pct_change().abs().tail(days).sum()
        avg_volume = volume_series.tail(days).mean()
        
        # 意圖因子計算 (保留您的原始邏輯)
        intent_factor = 0
        score_intent = 0
        is_intent_candidate = False
        
        if v_variability > 0 and avg_volume > 1000:
            raw_intent = s_return / v_variability
            if 0 < s_return < 0.25: 
                intent_factor = raw_intent
                is_intent_candidate = True
                score_intent = 20 # 權重微調，讓位給基本面
            elif s_return < -0.1:
                score_intent = 5 

        # --- 1. 機構大戶深層數據 (Institutional Data) ---
        
        # A. ROIC 計算 (簡易估算版)
        # NOPAT (稅後淨營業利潤) ≈ EBITDA * (1 - 稅率20%) 
        # Invested Capital (投入資本) ≈ 總負債 + 股東權益 - 現金
        ebitda = info.get('ebitda')
        total_debt = info.get('totalDebt')
        total_cash = info.get('totalCash')
        equity = info.get('stockholdersEquity')
        
        roic = 0
        if ebitda and total_debt and equity and total_cash:
            invested_capital = total_debt + equity - total_cash
            nopat = ebitda * 0.8 
            if invested_capital > 0:
                roic = nopat / invested_capital

        # B. FCF Yield 計算 (真實估值)
        fcf = info.get('freeCashflow')
        mkt_cap = info.get('marketCap')
        fcf_yield = 0
        if fcf and mkt_cap and mkt_cap > 0:
            fcf_yield = fcf / mkt_cap

        # C. 獲利品質 (Quality of Income)
        # 營業現金流 / 淨利 (若無淨利數據則忽略)
        op_cash = info.get('operatingCashflow')
        net_income = info.get('netIncomeToCommon')
        earnings_quality = 0
        if op_cash and net_income and net_income > 0:
            earnings_quality = op_cash / net_income

        # D. PEG 與 估值
        peg_ratio = info.get('pegRatio', None)
        pb = info.get('priceToBook', 0)

        # --- 2. CAPM & WACC (風險控管) ---
        stock_returns = close_series.pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        if len(aligned) < 60: beta = 1.0
        else:
            cov = aligned.cov().iloc[0, 1]
            mkt_var = aligned['Market'].var()
            beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        ke = RF + beta * MRP # 權益資金成本
        
        # Gordon 合理價 (作為參考)
        div_rate = info.get('dividendRate', 0)
        if not div_rate:
            yield_val = info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val
        
        fair_value = np.nan
        k_minus_g = max(ke - G_GROWTH, 0.015) 
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        # --- 3. 大戶多因子評分系統 (Scoring) ---
        score = score_intent # 初始分 (0-20)
        factors = []
        
        if is_intent_candidate: factors.append("💎主力畫線")

        # [新增] 資本效率因子 (ROIC) - 大戶最愛
        # ROIC > 15% 代表極佳護城河 (如台積電)
        if roic > 0.15:
            score += 20
            factors.append(f"高資本效率(ROIC {roic:.1%})")
        elif roic > 0.10:
            score += 10
            factors.append("ROIC優")

        # [新增] 現金流因子 (FCF Yield) - 價值防禦
        # FCF Yield > 4% 代表即使不成長，現金回報也很可觀
        if fcf_yield > 0.05:
            score += 20
            factors.append(f"現金牛(FCF殖利率{fcf_yield:.1%})")
        elif fcf_yield > 0.03:
            score += 10

        # [新增] 獲利品質 (Earnings Quality) - 避雷針
        # 現金流比淨利大，代表賺的是真錢
        if earnings_quality > 1.2:
            score += 10
            factors.append("獲利含金量高")
        elif earnings_quality < 0.5 and net_income > 0:
            score -= 10 # 扣分：賺的錢都是應收帳款(虛的)

        # 成長估值 (PEG)
        if peg_ratio and 0 < peg_ratio < 1.0:
            score += 15
            factors.append("PEG低估(成長>估值)")

        # 傳統價值 (PB)
        if 0 < pb < 1.2: 
            score += 10
            factors.append("低PB")

        # 波動率 (Smart Beta)
        volatility = stock_returns.std() * (252**0.5)
        if volatility < 0.30:
            score += 10
            if volatility < 0.25: factors.append("籌碼安定")
            
        # 技術面 (站上月線)
        ma20 = close_series.rolling(20).mean().iloc[-1]
        if current_price > ma20: score += 5
        
        # CGO (籌碼獲利)
        ma100 = close_series.rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        if cgo_val > 0.05: score += 5

        # --- 4. 生成大戶視角診斷 ---
        
        # 診斷文字
        inst_view = ""
        if roic > ke:
            inst_view += "✅ **價值創造**：ROIC > 資金成本(Ke)，公司正在為股東創造真實價值。"
        else:
            inst_view += "⚠️ **價值毀滅**：ROIC < 資金成本(Ke)，需留意資本使用效率。"
            
        if fcf_yield > 0.04:
            inst_view += f" 現金流強勁 (FCF Yield {fcf_yield:.1%})，下檔具支撐。"
        elif fcf < 0:
            inst_view += " 自由現金流為負，留意燒錢狀況。"

        path_diagnosis = f"趨勢向上 ({s_return:.1%})" if s_return > 0 else f"趨勢修正 ({s_return:.1%})"

        final_advice = (
            f"📊 **大戶因子解析**：\n"
            f"1. **品質**：ROIC {roic:.1%} | {inst_view}\n"
            f"2. **估值**：FCF Yield {fcf_yield:.1%} | PEG {peg_ratio if peg_ratio else 'N/A'}\n"
            f"3. **技術**：{path_diagnosis} | Beta {beta:.2f}"
        )

        # 回傳門檻
        if score >= 30: 
            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "AI綜合建議": final_advice,
                "ROIC": f"{roic:.1%}", # 新增顯示
                "FCF Yield": f"{fcf_yield:.1%}", # 新增顯示
                "合理價": fair_value if not np.isnan(fair_value) else 0,
                "亮點": " | ".join(factors)
            }
    except Exception as e:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.6", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.6 (機構法人旗艦版)")
st.markdown("""
本系統整合 **CAPM、Fama-French** 與 **大戶品質因子 (Quality)**。
**V9.6 核心升級：** 引入 **ROIC、FCF Yield、PEG**，透過機構法人視角，識破財務槓桿與虛胖成長。
""")

# --- 知識庫 Expander ---
with st.expander("📚 點此查看：機構法人選股邏輯 (ROIC & FCF)"):
    tab_intent, tab_theory, tab_chips = st.tabs(["💎 核心：ROIC與品質", "CAPM與三因子", "籌碼與CGO"])
    
    with tab_intent:
        st.markdown("""
        ### 💎 大戶核心：ROIC 與 FCF
        
        **1. ROIC (投入資本回報率)**：
        * **定義**：公司用本錢 (股東權益+負債) 賺取本業獲利的效率。
        * **門檻**：至少要 > WACC (約 5~8%)。若 > 15% 則為頂級護城河公司。
        
        **2. FCF Yield (自由現金流收益率)**：
        * **定義**：`自由現金流 / 市值`。
        * **意義**：這是您買下整間公司後，每年能拿到的真實現金回報。比本益比 (PE) 更真實，因為現金流騙不了人。
        
        **3. 價格意圖因子**：
        * 輔助判斷：在基本面優異的前提下，尋找走勢穩定 (直線上漲) 的標的。
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
    st.info("💡 系統執行：ROIC 資本效率篩選 + FCF 真實估值 + 意圖因子輔助。")
    if st.button("🚀 啟動 AI 智能運算 (Top 100)", type="primary"):
        with st.spinner("Step 1: 載入大盤數據..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 全市場掃描 (財務結構運算)..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定 {len(tickers)} 檔標的，開始深度財務分析...")
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
        
        # 排序：優先展示 AI 評分高，且 ROIC 表現好的
        df = df.sort_values(by=['AI綜合評分', '意圖因子'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 Top 100 (機構法人觀點)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "AI綜合建議", "ROIC", "FCF Yield", "合理價", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含股價路徑與大戶財務診斷"),
                "ROIC": st.column_config.TextColumn(help="投入資本回報率 (越高等級越高)"),
                "FCF Yield": st.column_config.TextColumn(help="自由現金流收益率 (真實的殖利率)"),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
