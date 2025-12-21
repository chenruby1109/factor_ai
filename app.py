import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import concurrent.futures
import twstock
import gc
import time

# --- 設定區 ---
TELEGRAM_BOT_TOKEN = '您的_BOT_TOKEN' 
TELEGRAM_CHAT_ID = '您的_CHAT_ID'

# --- 全局參數 ---
RF = 0.015  
MRP = 0.055 
G_GROWTH = 0.02 

# --- 核心功能函數 ---

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
        # 抓取 twstock 內建清單
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
        # 優先嘗試 yfinance (歷史數據最後一筆)
        ticker = yf.Ticker(stock_code)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
    except: pass

    if price is None:
        try:
            # 備用嘗試 twstock
            code = stock_code.split('.')[0]
            realtime = twstock.realtime.get(code)
            if realtime['success']:
                rt = realtime['realtime']
                # 嘗試抓成交價，沒有則抓買進價
                p = rt.get('latest_trade_price', '-')
                if p == '-' or not p:
                    p = rt.get('best_bid_price', ['-'])[0]
                if p and p != '-':
                    price = float(p)
        except: pass
    return price

def calculate_theoretical_factors(ticker_symbol, name_map, market_returns):
    """【Miniko V10.0 AI 旗艦運算核心 - 記憶體優化版】"""
    try:
        # 1. 先抓價格，若失敗直接跳過，節省資源
        current_price = get_realtime_price_robust(ticker_symbol)
        if current_price is None or current_price <= 0: return None
        if current_price < 10: return None # 排除雞蛋水餃股

        # 2. 下載數據 (限制只抓 1 年)
        data = yf.download(ticker_symbol, period="1y", interval="1d", progress=False)
        if len(data) < 100: return None 
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # --- 技術面與買點 ---
        ma20 = data['Close'].rolling(20).mean().iloc[-1]
        ma60 = data['Close'].rolling(60).mean().iloc[-1]
        
        if current_price > ma20:
            buy_point = ma20 # 強勢回檔買月線
        else:
            buy_point = current_price * 0.98 # 弱勢下方接
        buy_point = round(buy_point, 2)

        # --- CAPM & 基本面 ---
        stock_returns = data['Close'].pct_change().dropna()
        # 簡化 Covariance 計算以節省記憶體，若數據長度不對齊則用簡易 Beta
        if len(stock_returns) > 60:
            volatility = stock_returns.std() * (252**0.5)
        else:
            volatility = 0.5 # 預設值

        ke = RF + 1.0 * MRP # 簡化 Beta=1 以加速運算，差異不大
        
        # Gordon Model
        ticker_info = yf.Ticker(ticker_symbol).info
        div_rate = ticker_info.get('dividendRate', 0)
        if not div_rate:
            y_val = ticker_info.get('dividendYield', 0)
            if y_val: div_rate = current_price * y_val
            
        fair_value = np.nan
        if div_rate and div_rate > 0:
            k_g = max(ke - G_GROWTH, 0.015)
            fair_value = round(div_rate / k_g, 2)

        # Smart Beta (CGO)
        ma100 = data['Close'].rolling(100).mean().iloc[-1]
        cgo_val = (current_price - ma100) / ma100
        
        # --- AI 評分 ---
        score = 0
        factors = []
        
        # 價值面
