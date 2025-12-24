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
COST_OF_DEBT_NET = 0.022 # 稅後債務成本

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
    【V9.9 大戶法人旗艦版】保留原功能，增加 WACC 所需數據
    """
    metrics = {
        'roic': None,
        'fcf_yield': None,
        'peg': None,
        'pb': None,
        'div_rate': None,
        'total_debt': 0,      
        'total_equity': 0     
    }
    
    try:
        info = ticker_obj.info
        metrics['pb'] = info.get('priceToBook')
        metrics['peg'] = info.get('pegRatio')
        metrics['div_rate'] = info.get('dividendRate')
        
        fin = ticker_obj.financials
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        mkt_cap = info.get('marketCap')

        # WACC 數據
        total_debt = 0
        if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
        elif 'TotalDebt' in bs.index: total_debt = bs.loc['TotalDebt'].iloc[0]
        metrics['total_debt'] = total_debt

        stockholders_equity = 0
        if 'Stockholders Equity' in bs.index: stockholders_equity = bs.loc['Stockholders Equity'].iloc[0]
        elif 'StockholdersEquity' in bs.index: stockholders_equity = bs.loc['StockholdersEquity'].iloc[0]
        metrics['total_equity'] = stockholders_equity

        # ROIC 計算
        try:
            ebit = None
            if 'EBIT' in fin.index: ebit = fin.loc['EBIT'].iloc[0]
            elif 'Operating Income' in fin.index: ebit = fin.loc['Operating Income'].iloc[0]
            elif 'OperatingIncome' in fin.index: ebit = fin.loc['OperatingIncome'].iloc[0]
            
            cash = 0
            if 'Cash And Cash Equivalents' in bs.index: cash = bs.loc['Cash And Cash Equivalents'].iloc[0]
            
            if ebit and stockholders_equity:
                invested_capital = total_debt + stockholders_equity - cash
                if invested_capital > 0:
                    metrics['roic'] = (ebit * 0.8) / invested_capital
        except: pass

        # FCF 計算
        try:
            ocf = None
            if 'Operating Cash Flow' in cf.index: ocf = cf.loc['Operating Cash Flow'].iloc[0]
            elif 'Total Cash From Operating Activities' in cf.index: ocf = cf.loc['Total Cash From Operating Activities'].iloc[0]
            
            capex = 0
            if 'Capital Expenditure' in cf.index: capex = cf.loc['Capital Expenditure'].iloc[0]
            
            fcf_val = None
            if 'Free Cash Flow' in cf.index: 
                fcf_val = cf.loc['Free Cash Flow'].iloc[0]
            elif ocf is not None:
                fcf_val = ocf + capex
            
            if fcf_val and mkt_cap:
                metrics['fcf_yield'] = fcf_val / mkt_cap
        except: pass
            
    except: pass
    return metrics

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    try:
        stock_name = name_map.get(ticker_symbol, ticker_symbol)
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 60: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # --- 深層挖掘 ---
        ticker = yf.Ticker(ticker_symbol)
        deep_metrics = get_financial_metrics_deep(ticker)
        
        roic = deep_metrics['roic']
        fcf_yield = deep_metrics['fcf_yield']
        pb = deep_metrics['pb']
        peg_ratio = deep_metrics['peg']
        div_rate = deep_metrics['div_rate']

        # 技術指標準備 (為了安全濾網)
        close_series = data['Close']
        ma60 = close_series.rolling(60).mean().iloc[-1]

        # ==========================================
        # 🛡️ 【安全防禦過濾系統】 
        # ==========================================
        
        # 1. 現金流濾網：FCF Yield < 10% (0.10) 淘汰
        #    (從 15% 稍微下修至 10% 以避免選不到股，但仍屬於 Deep Value)
        if fcf_yield is None or fcf_yield < 0.10:
            return None
            
        # 2. 趨勢濾網 (避開價值陷阱)：股價必須在季線之上
        #    (如果高殖利率但股價在季線下，極可能是接刀)
        if current_price < ma60:
            return None

        # 3. 品質濾網 (避開爛公司)：ROIC 必須大於 8%
        #    (確保公司本業具有一定賺錢效率，非曇花一現)
        if roic is None or roic < 0.08:
            return None
        # ==========================================

        # --- 1. CAPM 與 Beta ---
        stock_returns = close_series.pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        beta = 1.0
        if len(aligned) > 30:
            cov = aligned.cov().iloc[0, 1]
            mkt_var = aligned['Market'].var()
            beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        ke = RF + beta * MRP 

        # --- 2. WACC 計算 ---
        wacc = None
        total_debt = deep_metrics['total_debt']
        total_equity = deep_metrics['total_equity']
        if total_equity > 0:
            total_capital = total_equity + total_debt
            weight_equity = total_equity / total_capital
            weight_debt = total_debt / total_capital
            wacc = (ke * weight_equity) + (COST_OF_DEBT_NET * weight_debt)

        # --- 3. CGO 與 VWAP ---
        df_60 = data.tail(60)
        vwap_60 = (df_60['Close'] * df_60['Volume']).sum() / df_60['Volume'].sum()
        cgo_status = ""
        cgo_score = 0
        if vwap_60 > 0:
            cgo_val = (current_price - vwap_60) / vwap_60
            if cgo_val > 0.05:
                cgo_status = "籌碼獲利🔥"
                cgo_score = 10
            elif cgo_val > 0:
                cgo_status = "成本之上✅"
                cgo_score = 5
            else:
                cgo_status = "套牢壓力🥶"

        # --- 4. Smart Beta 低波動 ---
        volatility = stock_returns.std() * (252**0.5)
        is_low_vol = False
        if volatility < 0.25 or (beta < 0.8 and volatility < 0.35):
            is_low_vol = True

        # --- 原有指標計算 ---
        days = 60
        volume_series = data['Volume']
        price_60_ago = close_series.iloc[-days]
        s_return = (current_price / price_60_ago) - 1
        v_variability = close_series.pct_change().abs().tail(days).sum()
        avg_volume = volume_series.tail(days).mean()
        
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

        # --- 評分系統 ---
        score = 0
        factors = []
        
        ma20 = close_series.rolling(20).mean().iloc[-1]
        
        if current_price > ma20: score += 20 
        if current_price > ma60: score += 10 # 雖然前面已經濾過，這裡保留加分邏輯
        if is_intent_candidate: 
            score += score_intent
            factors.append("💎主力軌跡")
        
        # CGO 加分
        score += cgo_score

        # 低波動加分
        if is_low_vol: 
            score += 10
            factors.append("🛡️低波動")

        # ROIC / WACC 判斷
        inst_view = "" 
        if roic is not None:
            if wacc and roic > wacc: 
                score += 25
                factors.append(f"價值創造(ROIC>WACC)")
                inst_view = f"✅價值創造 (ROIC {roic:.1%} > WACC {wacc:.1%})"
            elif roic > 0.15:
                score += 25
                factors.append(f"高資本效率(ROIC {roic:.1%})")
                inst_view = "✅高資本效率"
            else:
                inst_view = "資本效率尚可"
        
        # FCF 加分 (既然能通過篩選，FCF 肯定很高)
        if fcf_yield > 0.15:
            score += 30
            factors.append(f"超高現金流({fcf_yield:.1%})")
        else:
            score += 20
            factors.append(f"高現金流({fcf_yield:.1%})")

        volatility_old = stock_returns.std() * (252**0.5)
        if volatility_old < 0.35: score += 10
        
        # 合理價
        fair_value = np.nan
        if div_rate:
            k_minus_g = max(ke - G_GROWTH, 0.015)
            fair_value = div_rate / k_minus_g

        # --- 生成文字 ---
        if score >= 15: 
            roic_str = f"{roic:.1%}" if roic is not None else "N/A"
            fcf_str = f"{fcf_yield:.1%}" 
            peg_str = f"{peg_ratio}" if peg_ratio else "N/A"
            wacc_str = f"{wacc:.1%}" if wacc else "N/A"

            path_diagnosis = f"趨勢向上 (+{s_return:.1%})" if s_return > 0 else f"趨勢修正 ({s_return:.1%})"
            
            final_advice = (
                f"📊 **AI 深度解析**：\n"
                f"1. **品質**：{inst_view} (已過濾掉 ROIC < 8% 之爛股)\n"
                f"2. **估值**：FCF Yield {fcf_str} (已過濾 FCF < 10% 且趨勢向下之標的)\n"
                f"3. **技術**：{path_diagnosis} | Beta {beta:.2f} | 站穩季線\n"
                f"4. **籌碼/風險**：CGO {cgo_status} | {'低波動 Smart Beta' if is_low_vol else '一般波動'}"
            )

            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "合理價": round(fair_value, 2) if not np.isnan(fair_value) else 0,
                "AI綜合評分": round(score, 1),
                "AI綜合建議": final_advice,
                "意圖因子": round(intent_factor, 2), 
                "ROIC": roic_str,     
                "FCF Yield": fcf_str, 
                "WACC": wacc_str,     
                "CGO": cgo_status,    
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
**V9.9 安全防禦版：** * **FCF Yield > 10%**：確保深度價值。
* **ROIC > 8%**：確保公司體質健康，非曇花一現。
* **Price > MA60**：確保趨勢向上，避開價值陷阱（接刀）。
""")

# --- 知識庫 Expander ---
with st.expander("📚 點此查看：機構法人選股邏輯 (ROIC & WACC)"):
    tab_intent, tab_theory, tab_chips = st.tabs(["💎 ROIC vs WACC", "CAPM與三因子", "籌碼與CGO"])
    with tab_intent:
        st.markdown("""
        ### 💎 大戶核心：ROIC vs WACC
        * **ROIC**：公司用本錢賺取獲利的效率。
        * **WACC**：公司的資金成本。
        * **關鍵**：ROIC 必須大於 WACC，才代表公司真的在創造價值。
        """)
    with tab_theory:
        st.markdown("""
        ### CAPM & Smart Beta
        * **Beta**：評估個股相對於大盤的波動風險。
        * **低波動**：系統會自動標記低波動且 Beta 較低的防禦型標的。
        """)
    with tab_chips:
        st.markdown("""
        ### CGO (籌碼獲利狀態)
        * **CGO > 0**：現價高於市場平均成本 (VWAP)，籌碼處於獲利狀態，賣壓較輕。
        * **CGO < 0**：現價低於市場平均成本，上方有解套賣壓。
        """)

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統執行：啟動安全防禦篩選 (FCF>10%, ROIC>8%, Price>MA60)...")
    if st.button("🚀 啟動 AI 智能運算", type="primary"):
        with st.spinner("Step 1: 載入大盤數據..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 全市場掃描 (這會非常嚴格，請稍候)..."):
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
        st.write("👈 請點擊左側按鈕開始分析。(注意：已開啟安全過濾，只會顯示趨勢向上的價值股)")
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序
        df = df.sort_values(by=['AI綜合評分', '意圖因子'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 (Top 100)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_order=[
                "代號", "名稱", "現價", "合理價", 
                "AI綜合評分", "AI綜合建議", 
                "ROIC", "FCF Yield", 
                "WACC", "CGO",       
                "亮點"
            ],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含大戶視角的三面向診斷"),
                "ROIC": st.column_config.TextColumn(help="投入資本回報率 (>8% 品質保證)"),
                "FCF Yield": st.column_config.TextColumn(help="自由現金流收益率 (>10%)"),
                "WACC": st.column_config.TextColumn(help="加權平均資本成本"),
                "CGO": st.column_config.TextColumn(help="籌碼獲利狀態"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
