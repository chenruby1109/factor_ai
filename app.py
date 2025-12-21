import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock
import time
import gc  # V9.8 新增：垃圾回收機制

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 ---
RF = 0.015  
MRP = 0.055 
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
    """下載大盤指數"""
    try:
        # 關鍵：threads=False 避免與主程式衝突
        market = yf.download("^TWII", period="1y", interval="1d", progress=False, threads=False)
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
    """【V9.8】增強版價格抓取"""
    price = None
    
    # 策略 1: yfinance
    try:
        ticker = yf.Ticker(stock_code)
        # 縮短 timeout 避免卡死
        hist = ticker.history(period="5d", timeout=5) 
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    # 策略 2: twstock
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
    """【V9.8】核心運算：資源控管 + 決策同步"""
    try:
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None

        # V9.8 關鍵：timeout 控制與單線程下載
        try:
            data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False, timeout=5, threads=False)
        except:
            return None

        if len(data) < 60: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if current_price < 10: return None 

        # --- 因子計算 ---
        stock_returns = data['Close'].pct_change().dropna()
        volatility = stock_returns.std() * (252**0.5)
        
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: div_rate = current_price * yield_val

        beta = 1.0
        ke = RF + beta * MRP
        k_minus_g = max(ke - G_GROWTH, 0.015)
        fair_value = np.nan
        if div_rate and div_rate > 0:
            fair_value = round(div_rate / k_minus_g, 2)

        market_cap = ticker_info.get('marketCap', 0)
        pb = ticker_info.get('priceToBook', 0)
        is_small_cap = 0 < market_cap < 50000000000
        is_value_stock = 0 < pb < 1.5
        
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        strategy_tags = []
        if cgo_val > 0.1 and volatility < 0.3:
            strategy_tags.append("🔥CGO低波")

        # --- AI 評分 ---
        score = 0.0
        factors = []
        
        ma5 = data['Close'].rolling(5).mean().iloc[-1]
        ma10 = data['Close'].rolling(10).mean().iloc[-1]
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]

        if is_value_stock: score += 15; factors.append("💎價值型")
        if not np.isnan(fair_value) and fair_value > current_price: score += 20; factors.append("💰低估")
        if is_small_cap: score += 10; factors.append("🐟中小股")
        if current_price > ma20: score += 10
        else: score -= 5
        
        roe = ticker_info.get('returnOnEquity', 0)
        if roe > 0.15: score += 15; factors.append("👑高ROE")
        
        if volatility < 0.25: score += 15; factors.append("🛡️低波")
        elif volatility > 0.5: score -= 10

        # --- 買點與決策同步邏輯 (消除落差感) ---
        bias_ma20 = (current_price - ma20) / ma20 
        anchor_price = ma20
        anchor_note = "MA20"

        if current_price > ma20:
            if bias_ma20 > 0.1: anchor_price = ma10; anchor_note = "MA10"
            elif bias_ma20 > 0.04: anchor_price = ma20; anchor_note = "MA20"
        else:
            if current_price > ma60: anchor_price = ma60; anchor_note = "MA60"
            else: anchor_price = current_price * 0.95; anchor_note = "超跌區"

        gap_percent = (current_price - anchor_price) / current_price
        ai_advice = "⏳ 觀望"
        final_buy_price = anchor_price
        final_buy_note = f"{anchor_note}支撐"

        # 這裡的門檻設為 40 以確保有結果顯示
        if score >= 40: 
            if gap_percent <= 0.03: 
                # 【同步邏輯優化】
                # 當 AI 判斷可買，建議價格直接顯示為現價，避免使用者混淆
                if score >= 80: ai_advice = "🚀 強力買進"
                else: ai_advice = "✅ 建議買進"
                final_buy_price = current_price 
                final_buy_note = f"現價進場 (防守{anchor_note})"
            else:
                # 當 AI 判斷要等，建議價格顯示下方的支撐價
                wait_percent = round(gap_percent * 100, 1)
                ai_advice = f"📉 等回檔 ({wait_percent}%)"
                final_buy_price = anchor_price
                final_buy_note = f"乖離過大，等待{anchor_note}"

            return {
                "代號": ticker_symbol.replace(".TW", "").replace(".TWO", ""), 
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "現價": float(current_price),
                "AI決策": ai_advice, 
                "AI綜合評分": round(score, 1), 
                "建議買點": float(round(final_buy_price, 2)),
                "買點說明": final_buy_note,
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

st.set_page_config(page_title="Miniko 投資戰情室 V9.8", layout="wide")

st.title("📊 Miniko & 曜鼎豐 - 投資戰情室 V9.8 (全台股穩定掃描版)")
st.markdown("""
**【V9.8 企業級更新】** 1. **穩定性核心**：導入 `gc` 記憶體回收與小批次處理，解決 1800 檔掃描時的崩潰問題。
2. **決策直觀化**：完全消除「建議買進」與「建議價格」的視覺落差，操作更直覺。
**⚠️ 提示：** 全市場掃描資料量巨大，請耐心等候進度條跑完 (約 10-15 分鐘)。
""")

# --- 主程式區 ---
if 'results' not in st.session_state:
    st.session_state['results'] = []
if 'scan_performed' not in st.session_state:
    st.session_state['scan_performed'] = False

col1, col2 = st.columns([1, 4])

with col1:
    st.info("💡 系統將執行全市場 AI 綜合評估。")
    if st.button("🚀 啟動 AI 全台股掃描", type="primary"):
        st.session_state['scan_performed'] = True
        
        with st.spinner("Step 1: 計算市場風險參數..."):
            market_returns = get_market_data()
        
        with st.spinner("Step 2: 載入全台股清單..."):
            tickers, name_map = get_all_tw_tickers()
            
        st.success(f"清單載入完成！準備分析 {len(tickers)} 檔股票...")
        st.session_state['results'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # --- V9.8 關鍵技術：批次處理 + 記憶體回收 ---
        # 將批次縮小為 20 檔，確保穩定
        BATCH_SIZE = 20 
        total_tickers = len(tickers)
        
        # 使用 max_workers=4 
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i in range(0, total_tickers, BATCH_SIZE):
                batch_tickers = tickers[i : i + BATCH_SIZE]
                
                # 送出這一批次
                future_to_ticker = {executor.submit(calculate_theoretical_factors, t, name_map, market_returns): t for t in batch_tickers}
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    data = future.result()
                    if data:
                        st.session_state['results'].append(data)
                
                # 強制執行垃圾回收，防止記憶體溢出
                gc.collect()

                # 更新進度
                current_count = min(i + BATCH_SIZE, total_tickers)
                progress_percent = current_count / total_tickers
                progress_bar.progress(progress_percent)
                status_text.text(f"🔥 AI 全力運算中... 進度: {current_count} / {total_tickers} 檔")
                
                # 稍微休息，保護 IP
                time.sleep(0.1) 

        status_text.text("✅ 全市場掃描完成！")

with col2:
    if not st.session_state['scan_performed']:
        st.info("👈 請點擊左側按鈕開始分析 (全掃描需時較長)。")
    
    elif st.session_state['scan_performed'] and len(st.session_state['results']) == 0:
        st.error("⚠️ 掃描完成，但沒有找到符合條件的股票。")
        st.markdown("**可能原因：** 網路不穩導致資料抓取失敗，請重新整理頁面再試。")
        
    else:
        df = pd.DataFrame(st.session_state['results'])
        
        # 排序邏輯
        df['SortKey'] = df['策略標籤'].apply(lambda x: 100 if "CGO" in x else 0)
        df['TotalScore'] = df['AI綜合評分'] + df['SortKey']
        
        # 取前 100 名
        df_top100 = df.sort_values(by=['TotalScore', 'AI綜合評分'], ascending=[False, False]).head(100)
        
        st.subheader(f"🏆 AI 全台股精選 Top 100 (從 {len(st.session_state['results'])} 檔有效股中選出)")
        
        st.dataframe(
            df_top100,
            use_container_width=True,
            hide_index=True,
            column_order=["代號", "名稱", "現價", "AI決策", "AI綜合評分", "建議買點", "買點說明", "合理價", "策略標籤", "CGO指標", "亮點"],
            column_config={
                "代號": st.column_config.TextColumn(help="股票代碼"),
                "現價": st.column_config.NumberColumn(format="$%.2f"),
                "AI決策": st.column_config.TextColumn(help="AI操作建議"),
                "AI綜合評分": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
                "建議買點": st.column_config.NumberColumn(format="$%.2f", help="若建議買進，此價格即為現價"),
                "合理價": st.column_config.NumberColumn(format="$%.2f"),
                "CGO指標": st.column_config.NumberColumn(format="%.1f%%"),
                "亮點": st.column_config.TextColumn(width="medium"),
            }
        )
