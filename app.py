import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import requests
from datetime import datetime
import concurrent.futures
import twstock

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 ---
RF = 0.015  # 無風險利率 (Risk-Free Rate)
MRP = 0.055 # 市場風險溢酬 (Market Risk Premium)
G_GROWTH = 0.02 # 股利長期成長率 (Gordon Growth Rate)

# --- 核心功能 ---

def get_realtime_price_robust(stock_code):
    """
    【V8.3 終極價格修復版】
    解決週末/盤後價格為 0 或異常的問題。
    策略：
    1. 優先抓取 yfinance 最近 5 日的 'Close' (最穩定的收盤價)。
    2. 如果是平日盤中，才嘗試 twstock 即時報價。
    """
    price = None
    
    # --- 策略 1: yfinance 歷史數據 (最穩定，適合週末/盤後) ---
    try:
        # 抓 5 天是為了避開連假，取最後一筆非 NaN 的 Close
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except:
        pass

    # --- 策略 2: twstock (僅在平日盤中或 yf 失敗時做為輔助) ---
    # 如果策略 1 失敗，或者我們懷疑 yf 資料延遲(盤中)，再用這個
    if price is None:
        try:
            code = stock_code.split('.')[0]
            realtime = twstock.realtime.get(code)
            if realtime['success']:
                rt_price = realtime['realtime']['latest_trade_price']
                # 處理 twstock 回傳 '-' 的情況
                if rt_price and rt_price != '-' and float(rt_price) > 0:
                    price = float(rt_price)
                else:
                    # 如果沒有成交價(比如剛開盤)，抓開盤價或最佳買價
                    best_bid = realtime['realtime']['best_bid_price'][0]
                    if best_bid and best_bid != '-' and float(best_bid) > 0:
                        price = float(best_bid)
        except:
            pass

    return price

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """
    【Miniko V8.3 運算核心】
    """
    try:
        # 1. 獲取絕對正確的價格 (V8.3)
        current_price = get_realtime_price_robust(ticker_symbol)
        
        # 如果價格還是抓不到或是 0，直接跳過這檔股票
        if current_price is None or current_price <= 0: 
            return None

        # 2. 下載歷史數據 (用於計算技術指標與 Beta)
        # 注意：這邊不用再抓一次 current_price，避免覆蓋掉上面抓準的價格
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # 過濾雞蛋水餃股
        if current_price < 10: return None

        # --- A. CAPM 模型 (資本資產定價模型) ---
        stock_returns = data['Close'].pct_change().dropna()
        # 確保索引對齊
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner').dropna()
        aligned_data.columns = ['Stock', 'Market']
        
        if len(aligned_data) < 60: return None # 樣本數太少不計算

        covariance = aligned_data.cov().iloc[0, 1]
        market_variance = aligned_data['Market'].var()
        
        # Beta (風險係數)
        beta = covariance / market_variance if market_variance != 0 else 1.0
        
        # 預期報酬率 (Expected Return)
        expected_return = RF + beta * MRP

        # --- B. Gordon 模型 (股利折現模型) ---
        ticker_info = yf.Ticker(ticker_symbol).info
        dividend_rate = ticker_info.get('dividendRate', 0)
        
        # 補強：如果 Yahoo 缺股利資料，改用殖利率推算
        if dividend_rate is None or dividend_rate == 0:
            yield_val = ticker_info.get('dividendYield', 0)
            if yield_val: dividend_rate = current_price * yield_val

        fair_value = np.nan
        upside_potential = np.nan
        
        # 公式：合理價 = 股利 / (預期報酬率 - 成長率)
        # 保護機制：避免分母過小導致價格無限大
        k_minus_g = max(expected_return - G_GROWTH, 0.015) 
        
        if dividend_rate and dividend_rate > 0:
            theoretical_price = dividend_rate / k_minus_g
            fair_value = round(theoretical_price, 2)
            # 計算獲利空間
            upside_potential = (fair_value - current_price) / current_price

        # --- C. 數據準備 ---
        rev_growth = ticker_info.get('revenueGrowth', 0)
        roe = ticker_info.get('returnOnEquity', 0)
        pb_ratio = ticker_info.get('priceToBook', 0)
        
        # --- D. 評分系統 ---
        score = 0.0
        factors = []
        
        # 1. 價值 (Value)
        if not np.isnan(fair_value) and fair_value > current_price:
            val_score = min(upside_potential * 100, 30)
            score += val_score
            factors.append(f"💰低於合理價")
        
        # 2. 成長 (Growth)
        if rev_growth and rev_growth > 0:
            g_score = min(rev_growth * 100, 25)
            score += g_score
            if g_score > 15: factors.append(f"📈營收高成長")

        # 3. 品質 (Quality - ROE)
        if roe and roe > 0:
            q_score = min(roe * 100, 20)
            score += q_score
            if roe > 0.15: factors.append(f"👑高股東權益報酬")

        # 4. 價值 (PB)
        if pb_ratio and 0 < pb_ratio < 1.5:
            score += 15
            factors.append(f"💎低股價淨值比")
            
        # 5. 技術 (Momentum) - 確保 data['Close'] 有值
        if len(data) > 60:
            ma60 = data['Close'].rolling(60).mean().iloc[-1]
            bias = (current_price - ma60) / ma60
            if 0 < bias < 0.08:
                score += 20
                factors.append("🎯剛站上季線")
            elif bias > 0.2:
                score -= 10
        
        # 6. 風險 (Volatility)
        volatility = stock_returns.std() * (252**0.5)
        if volatility > 0.6: score -= 15
        
        if score >= 50:
            return {
                "代號": ticker_symbol,
                "名稱": name_map.get(ticker_symbol, ticker_symbol),
                "最新收盤價": float(current_price), # 確保是 float
                "綜合評分": round(score, 1),
                "理論合理價": fair_value if not np.isnan(fair_value) else None,
                "預估獲利空間": upside_potential if not np.isnan(upside_potential) else None,
                "資金成本": expected_return,
                "風險係數": float(beta),
                "亮點因子": " | ".join(factors)
            }

    except Exception as e:
        # print(f"Error analyzing {ticker_symbol}: {e}") # Debug用
        return None
    return None
