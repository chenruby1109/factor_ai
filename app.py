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

# --- 全局參數 ---
RF = 0.015  # 無風險利率
MRP = 0.055 # 市場風險溢酬
G_GROWTH = 0.02 

# --- 核心功能函數 ---

def send_telegram_message(message):
    if TELEGRAM_BOT_TOKEN == '您的_BOT_TOKEN': return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

@st.cache_data(ttl=3600) 
def get_market_data():
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

def get_financial_metrics_deep(ticker_obj):
    """
    【V9.9 大戶法人旗艦版】
    如果不依賴 unstable 的 .info，直接從三大報表 (financials, balance_sheet, cashflow) 
    挖掘原始數據並手動計算 ROIC 和 FCF。
    """
    metrics = {
        'roic': None,
        'fcf_yield': None,
        'peg': None,
        'pb': None,
        'div_rate': None,
        'ebitda': None
    }
    
    try:
        # 1. 嘗試從 info 抓取 (最快，但常失敗)
        info = ticker_obj.info
        metrics['pb'] = info.get('priceToBook')
        metrics['peg'] = info.get('pegRatio')
        metrics['div_rate'] = info.get('dividendRate')
        
        # 2. 深層挖掘：抓取三大報表
        # 使用 try-except 避免某些表不存在導致 crash
        fin = ticker_obj.financials
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        mkt_cap = info.get('marketCap')

        # --- 手動計算 ROIC ---
        # 公式: NOPAT / Invested Capital
        # NOPAT ≈ EBIT * (1-Tax) 
        # Invested Capital = Total Debt + Equity - Cash
        try:
            # 尋找 EBIT 或 Operating Income
            ebit = None
            if 'EBIT' in fin.index: ebit = fin.loc['EBIT'].iloc[0]
            elif 'Operating Income' in fin.index: ebit = fin.loc['Operating Income'].iloc[0]
            elif 'OperatingIncome' in fin.index: ebit = fin.loc['OperatingIncome'].iloc[0]
            
            # 尋找資本結構
            total_debt = 0
            if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
            elif 'TotalDebt' in bs.index: total_debt = bs.loc['TotalDebt'].iloc[0]
            
            stockholders_equity = 0
            if 'Stockholders Equity' in bs.index: stockholders_equity = bs.loc['Stockholders Equity'].iloc[0]
            elif 'StockholdersEquity' in bs.index: stockholders_equity = bs.loc['StockholdersEquity'].iloc[0]
            
            cash = 0
            if 'Cash And Cash Equivalents' in bs.index: cash = bs.loc['Cash And Cash Equivalents'].iloc[0]
            
            if ebit and stockholders_equity:
                invested_capital = total_debt + stockholders_equity - cash
                if invested_capital > 0:
                    # 假設稅率 20%
                    metrics['roic'] = (ebit * 0.8) / invested_capital
        except:
            pass

        # --- 手動計算 FCF ---
        # 公式: Operating Cash Flow + CapEx (CapEx通常為負值)
        try:
            ocf = None
            if 'Operating Cash Flow' in cf.index: ocf = cf.loc['Operating Cash Flow'].iloc[0]
            elif 'Total Cash From Operating Activities' in cf.index: ocf = cf.loc['Total Cash From Operating Activities'].iloc[0]
            
            capex = 0
            if 'Capital Expenditure' in cf.index: capex = cf.loc['Capital Expenditure'].iloc[0]
            
            fcf_val = None
            # 有些 API 會直接給 Free Cash Flow
            if 'Free Cash Flow' in cf.index: 
                fcf_val = cf.loc['Free Cash Flow'].iloc[0]
            elif ocf is not None:
                fcf_val = ocf + capex
            
            if fcf_val and mkt_cap:
                metrics['fcf_yield'] = fcf_val / mkt_cap
        except:
            pass
            
    except:
        pass
        
    return metrics

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.9 深層數據版】
    改用 get_financial_metrics_deep 函數強行計算指標，解決 N/A 問題。
    """
    try:
        stock_name = name_map.get(ticker_symbol, ticker_symbol)
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # 下載數據
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 60: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # --- 啟動深層挖掘 ---
        ticker = yf.Ticker(ticker_symbol)
        deep_metrics = get_financial_metrics_deep(ticker)
        
        roic = deep_metrics['roic']
        fcf_yield = deep_metrics['fcf_yield']
        pb = deep_metrics['pb']
        peg_ratio = deep_metrics['peg']
        div_rate = deep_metrics['div_rate']

        # --- 0. 基礎趨勢與意圖因子 ---
        days = 60
        close_series = data['Close']
        volume_series = data['Volume']
        
        price_60_ago = close_series.iloc[-days]
        s_return = (current_price / price_60_ago) - 1
        v_variability = close_series.pct_change().abs().tail(days).sum()
        avg_volume = volume_series.tail(days).mean()
        
        # 意圖因子
        intent_factor = 0
        score_intent = 0
        is_intent_candidate = False
        
        if v_variability > 0 and avg_volume > 500: 
            raw_intent = s_return / v_variability
            if 0 < s_return < 0.3: 
                intent_factor = raw_intent
                is_intent_candidate = True
                score_intent = 15
            elif s_return < -0.05:
                score_intent = 5 

        # --- 2. CAPM ---
        stock_returns = close_series.pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        beta = 1.0
        if len(aligned) > 30:
            cov = aligned.cov().iloc[0, 1]
            mkt_var = aligned['Market'].var()
            beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        ke = RF + beta * MRP 

        # --- 3. 評分系統 ---
        score = 0
        factors = []
        
        # A. 技術面保底
        ma20 = close_series.rolling(20).mean().iloc[-1]
        ma60 = close_series.rolling(60).mean().iloc[-1]
        
        if current_price > ma20: score += 20 
        if current_price > ma60: score += 10
            
        if is_intent_candidate: 
            score += score_intent
            factors.append("💎主力軌跡")

        # B. 財務面 (深層數據)
        if roic is not None:
            if roic > 0.15: 
                score += 25
                factors.append(f"高資本效率(ROIC {roic:.1%})")
            elif roic > 0.08:
                score += 15
        else:
            if pb and 0 < pb < 1.5:
                score += 15
                factors.append("低PB價值")
        
        # C. 現金流
        if fcf_yield is not None:
            if fcf_yield > 0.04:
                score += 20
                factors.append(f"現金牛({fcf_yield:.1%})")
        
        # D. 波動率
        volatility = stock_returns.std() * (252**0.5)
        if volatility < 0.35: score += 10
        
        # E. 估值保護
        fair_value = np.nan
        if div_rate:
            k_minus_g = max(ke - G_GROWTH, 0.015)
            fair_value = div_rate / k_minus_g

        # --- 4. 生成詳細診斷文本 ---
        if score >= 15: 
            
            # 數據格式化
            roic_str = f"{roic:.1%}" if roic is not None else "N/A"
            fcf_str = f"{fcf_yield:.1%}" if fcf_yield is not None else "N/A"
            peg_str = f"{peg_ratio}" if peg_ratio else "N/A"
            
            # 1. 品質觀點
            inst_view = ""
            if roic and roic > ke: inst_view += "✅價值創造(ROIC>Ke)"
            elif roic: inst_view += "⚠️資本效率待提升"
            else: inst_view += "財報暫缺，改採PB評價"

            # 2. 技術觀點
            path_diagnosis = f"趨勢向上 (+{s_return:.1%})" if s_return > 0 else f"趨勢修正 ({s_return:.1%})"
            
            final_advice = (
                f"📊 **AI 深度解析**：\n"
                f"1. **品質**：ROIC {roic_str} | {inst_view}\n"
                f"2. **估值**：FCF Yield {fcf_str} | PEG {peg_str}\n"
                f"3. **技術**：{path_diagnosis} | Beta {beta:.2f}"
            )

            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "AI綜合評分": round(score, 1),
                "AI綜合建議": final_advice, 
                "意圖因子": round(intent_factor, 2), 
                "ROIC": roic_str, 
                "FCF Yield": fcf_str,
                "合理價": round(fair_value, 2) if not np.isnan(fair_value) else 0,
                "亮點": " | ".join(factors)
            }
    except Exception as e:
        return None
    return None

# --- Streamlit 介面 ---

st.set_page_config(page_title="Miniko 投資戰情室 V9.9", layout="wide")

st.title("📊 Miniko  - 大戶悄悄話茶室 V9.9 (大戶法人旗艦版)")
st.markdown("""
本系統整合 **CAPM、Fama-French** 與 **大戶品質因子 (Quality)**。
**V9.9 最終修復：** 啟用「深層挖掘 (Deep Mining)」技術，直接讀取財報原始數據並手動運算，解決資料庫缺漏問題，讓 ROIC 與 FCF 數據重見天日。
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
    st.info("💡 系統執行：啟動深層報表挖掘 (Financials Mining)...")
    if st.button("🚀 啟動 AI 智能運算", type="primary"):
        with st.spinner("Step 1: 載入大盤數據..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 全市場掃描 (這可能會花一點時間挖掘財報)..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定 {len(tickers)} 檔標的，開始深度挖掘...")
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
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序
        df = df.sort_values(by=['AI綜合評分', '意圖因子'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 (Top 100)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI綜合評分", "AI綜合建議", "ROIC", "FCF Yield", "合理價", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含大戶視角的三面向診斷"),
                "ROIC": st.column_config.TextColumn(help="投入資本回報率 (深層挖掘版)"),
                "FCF Yield": st.column_config.TextColumn(),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
