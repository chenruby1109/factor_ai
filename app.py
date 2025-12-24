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
RF = 0.015  # 無風險利率 (Risk-Free Rate, e.g., 10Y Bond)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 永續成長率
COST_OF_DEBT_NET = 0.022 # 假設稅後債務成本 (約2.2%)，用於WACC估算

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
    【V9.9 大戶法人旗艦版 - 深層挖掘】
    新增: 提取債務與權益結構，用於後續 WACC 計算
    """
    metrics = {
        'roic': None,
        'fcf_yield': None,
        'peg': None,
        'pb': None,
        'div_rate': None,
        'total_debt': 0,      # 新增
        'total_equity': 0,    # 新增
        'invested_capital': 0 # 新增
    }
    
    try:
        # 1. 嘗試從 info 抓取
        info = ticker_obj.info
        metrics['pb'] = info.get('priceToBook')
        metrics['peg'] = info.get('pegRatio')
        metrics['div_rate'] = info.get('dividendRate')
        
        # 2. 深層挖掘：抓取三大報表
        fin = ticker_obj.financials
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
        mkt_cap = info.get('marketCap')

        # --- 結構數據 (用於 WACC) ---
        total_debt = 0
        if 'Total Debt' in bs.index: total_debt = bs.loc['Total Debt'].iloc[0]
        elif 'TotalDebt' in bs.index: total_debt = bs.loc['TotalDebt'].iloc[0]
        
        stockholders_equity = 0
        if 'Stockholders Equity' in bs.index: stockholders_equity = bs.loc['Stockholders Equity'].iloc[0]
        elif 'StockholdersEquity' in bs.index: stockholders_equity = bs.loc['StockholdersEquity'].iloc[0]
        
        metrics['total_debt'] = total_debt
        metrics['total_equity'] = stockholders_equity

        # --- 手動計算 ROIC ---
        try:
            # 尋找 EBIT
            ebit = None
            if 'EBIT' in fin.index: ebit = fin.loc['EBIT'].iloc[0]
            elif 'Operating Income' in fin.index: ebit = fin.loc['Operating Income'].iloc[0]
            elif 'OperatingIncome' in fin.index: ebit = fin.loc['OperatingIncome'].iloc[0]
            
            cash = 0
            if 'Cash And Cash Equivalents' in bs.index: cash = bs.loc['Cash And Cash Equivalents'].iloc[0]
            
            if ebit and stockholders_equity:
                invested_capital = total_debt + stockholders_equity - cash
                metrics['invested_capital'] = invested_capital # 存起來備用
                if invested_capital > 0:
                    metrics['roic'] = (ebit * 0.8) / invested_capital # 稅後 EBIT / 投入資本
        except:
            pass

        # --- 手動計算 FCF ---
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
        except:
            pass
            
    except:
        pass
        
    return metrics

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V9.9 全能版】
    整合 CAPM, WACC, CGO, Low Volatility 四大新指標
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
        
        # --- 1. CAPM 計算 (Beta & Ke) ---
        close_series = data['Close']
        stock_returns = close_series.pct_change().dropna()
        aligned = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned.columns = ['Stock', 'Market']
        
        beta = 1.0
        if len(aligned) > 30:
            cov = aligned.cov().iloc[0, 1]
            mkt_var = aligned['Market'].var()
            beta = cov / mkt_var if mkt_var != 0 else 1.0
        
        ke = RF + beta * MRP # Cost of Equity (股權成本)

        # --- 2. WACC 估算 (權重平均資本成本) ---
        # WACC = Ke * (E/V) + Kd*(1-t) * (D/V)
        wacc = None
        total_debt = deep_metrics['total_debt']
        total_equity = deep_metrics['total_equity']
        
        if total_equity > 0:
            total_capital = total_equity + total_debt
            weight_equity = total_equity / total_capital
            weight_debt = total_debt / total_capital
            # 使用全局設定的稅後債務成本 (COST_OF_DEBT_NET)
            wacc = (ke * weight_equity) + (COST_OF_DEBT_NET * weight_debt)

        # --- 3. CGO (未實現獲利) & VWAP ---
        # 計算 60日 VWAP (成交量加權平均價) 作為市場平均成本
        df_60 = data.tail(60)
        vwap_60 = (df_60['Close'] * df_60['Volume']).sum() / df_60['Volume'].sum()
        
        cgo_status = "N/A"
        cgo_score = 0
        if vwap_60 > 0:
            # CGO > 0 代表現價高於成本，籌碼獲利中 (Overhang of Profit)
            # CGO < 0 代表現價低於成本，有解套賣壓
            cgo_val = (current_price - vwap_60) / vwap_60
            if cgo_val > 0.05:
                cgo_status = "籌碼獲利🔥"
                cgo_score = 10
            elif cgo_val > 0:
                cgo_status = "成本之上✅"
                cgo_score = 5
            else:
                cgo_status = "套牢壓力🥶"

        # --- 4. 低波動 (Low Volatility / Smart Beta) ---
        volatility = stock_returns.std() * (252**0.5)
        is_low_vol = False
        if volatility < 0.25 or (beta < 0.8 and volatility < 0.35):
            is_low_vol = True

        # --- 5. 意圖因子 ---
        days = 60
        volume_series = data['Volume']
        price_60_ago = close_series.iloc[-days]
        s_return = (current_price / price_60_ago) - 1
        v_variability = close_series.pct_change().abs().tail(days).sum()
        avg_volume = volume_series.tail(days).mean()
        
        intent_factor = 0
        score_intent = 0
        
        if v_variability > 0 and avg_volume > 500: 
            raw_intent = s_return / v_variability
            if 0 < s_return < 0.3: 
                intent_factor = raw_intent
                score_intent = 15

        # --- 6. 綜合評分系統 ---
        score = 0
        factors = []
        
        # A. 技術面
        ma20 = close_series.rolling(20).mean().iloc[-1]
        if current_price > ma20: score += 20 
        if score_intent > 0: score += score_intent

        # B. 籌碼面 (CGO)
        score += cgo_score
        if cgo_score > 0: factors.append(f"{cgo_status}")

        # C. 風險面 (CAPM & Low Vol)
        if is_low_vol: 
            score += 10
            factors.append("🛡️低波動")
        
        # D. 品質面 (ROIC vs WACC)
        roic_view = ""
        if roic is not None:
            if wacc and roic > wacc:
                score += 25
                factors.append("💎價值創造(ROIC>WACC)")
                roic_view = f"ROIC {roic:.1%} > WACC {wacc:.1%}"
            elif roic > 0.10:
                score += 15
                roic_view = f"ROIC {roic:.1%} (Good)"
            else:
                roic_view = f"ROIC {roic:.1%} (Low)"
        else:
            if pb and 0 < pb < 1.5:
                score += 15
                factors.append("低PB價值")
                roic_view = "ROIC N/A"
        
        # E. 現金流
        if fcf_yield is not None and fcf_yield > 0.04:
            score += 20
            factors.append(f"現金牛({fcf_yield:.1%})")

        # F. 估值保護 (合理價)
        fair_value = np.nan
        if div_rate:
            k_minus_g = max(ke - G_GROWTH, 0.015)
            fair_value = div_rate / k_minus_g

        # --- 7. 生成詳細診斷文本 (整合 CAPM, WACC, CGO) ---
        if score >= 15: 
            
            # 數據格式化
            fcf_str = f"{fcf_yield:.1%}" if fcf_yield is not None else "N/A"
            wacc_str = f"{wacc:.1%}" if wacc else "N/A"
            ke_str = f"{ke:.1%}"
            
            # 1. 品質觀點 (Integrate WACC)
            quality_check = "✅" if (roic and wacc and roic > wacc) else "⚠️"
            
            # 2. 技術/籌碼觀點 (Integrate CGO)
            trend_view = f"多頭 ({s_return:.1%})" if s_return > 0 else "修正"
            
            final_advice = (
                f"📊 **AI 深度解析 (Miniko V9.9)**：\n"
                f"1. **品質對決**：{quality_check} {roic_view}\n"
                f"   (資金成本 WACC: {wacc_str} | 股權成本 Ke: {ke_str})\n"
                f"2. **籌碼CGO**：{cgo_status} | 現價 vs 市場成本(VWAP)\n"
                f"3. **風險屬性**：Beta {beta:.2f} | {'低波動 Smart Beta 🛡️' if is_low_vol else '一般波動'}\n"
                f"4. **估值**：FCF Yield {fcf_str}"
            )

            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""),
                "名稱": stock_name,
                "現價": float(current_price),
                "合理價": round(fair_value, 2) if not np.isnan(fair_value) else 0, # 移動到現價旁
                "AI綜合評分": round(score, 1),
                "AI綜合建議": final_advice, 
                "意圖因子": round(intent_factor, 2), 
                "ROIC": f"{roic:.1%}" if roic else "N/A", 
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
本系統整合 **CAPM、WACC、CGO (籌碼成本)** 與 **大戶品質因子 (Quality)**。
**V9.9 最終修復：** 啟用「深層挖掘 (Deep Mining)」與「風險定價模型」，全方位診斷價值創造能力。
""")

# --- 知識庫 Expander ---
with st.expander("📚 點此查看：新增指標說明 (WACC, CGO, CAPM)"):
    tab_quality, tab_risk, tab_chips = st.tabs(["💎 品質：ROIC vs WACC", "⚖️ 風險：CAPM & Smart Beta", "💰 籌碼：CGO"])
    
    with tab_quality:
        st.markdown("""
        ### 💎 終極檢驗：價值創造
        * **ROIC (投入資本回報率)**：公司運用資本賺錢的能力。
        * **WACC (加權平均資本成本)**：公司取得資金的成本 (包含付給股東的 Ke 與付給銀行的 Kd)。
        * **黃金法則**：只有當 **ROIC > WACC** 時，公司成長才是有意義的「價值創造」；反之則是在「毀滅價值」。
        """)

    with tab_risk:
        st.markdown("""
        ### ⚖️ CAPM 與 Smart Beta
        * **CAPM (Ke)**：根據市場風險 (Beta) 計算出的股東最低要求回報率。
        * **Smart Beta (低波動)**：系統會自動標記 Beta < 0.8 且波動率低的股票，這類股票在長期往往能提供更穩定的複利效果。
        """)
        
    with tab_chips:
        st.markdown("""
        ### 💰 CGO (Capital Gain Overhang)
        * **定義**：計算過去 60 天市場的「平均持倉成本 (VWAP)」。
        * **判讀**：
            * **籌碼獲利 (CGO > 0)**：現價在平均成本之上，主力與散戶皆賺錢，上方無解套賣壓，易漲難跌。
            * **套牢壓力 (CGO < 0)**：現價在平均成本之下，反彈容易遇到解套賣壓。
        """)

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統執行：啟動 CAPM 模型與深層報表挖掘...")
    if st.button("🚀 啟動 AI 智能運算", type="primary"):
        with st.spinner("Step 1: 載入大盤數據 (計算 Beta 用)..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 全市場掃描 (計算 WACC & CGO)..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"鎖定 {len(tickers)} 檔標的，開始深度運算...")
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
        
        # 排序：優先看 AI 評分高且籌碼面好的
        df = df.sort_values(by=['AI綜合評分', '意圖因子'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 嚴選現貨清單 (Top 100)")
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            # 修改順序：將合理價移到現價旁邊，並加入新指標
            column_order=["代號", "名稱", "現價", "合理價", "AI綜合評分", "AI綜合建議", "亮點", "WACC", "ROIC", "CGO"],
            column_config={
                "代號": st.column_config.TextColumn(width="small"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "合理價": st.column_config.NumberColumn(format="$%.2f", help="基於 CAPM Ke 與股利折現模型推算"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "AI綜合建議": st.column_config.TextColumn(width="large", help="包含 WACC, CGO, Beta 綜合診斷"),
                "WACC": st.column_config.TextColumn(help="加權平均資本成本"),
                "ROIC": st.column_config.TextColumn(help="投入資本回報率"),
                "CGO": st.column_config.TextColumn(help="市場持倉盈虧狀態"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
