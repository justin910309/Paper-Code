"""
此程式實現一個動態再平衡的投資策略，核心功能如下：
- 根據預先設定的投資權重與股價資料模擬投資組合績效
- 執行定期再平衡（每年）與動態再平衡（當偏離門檻）
- 記錄每次交易紀錄並儲存為 CSV
- 輸出累積報酬、Sharpe Ratio、最大跌幅等績效指標
- 印出曾觸發再平衡的股票清單
"""

import os
import pandas as pd
import numpy as np
import glob

# ----------------------------
# (A) 可調整參數集中放在這裡
# ----------------------------
CONFIG = {
    "weights_csv_path": "C:/Users/User/Desktop/123/新權重矩陣改過/Optimal_Weights_2008_2015年負0.9.csv",  # 投資組合權重
    "data_folder_path": "C:/Users/User/Desktop/台股資料2004-2023/test",  # 股價資料資料夾
    "start_date": "2016-01-01",
    "end_date": "2023-12-31",
    "initial_investment": 100000,
    "commission_rate": 0.001425,
    "tax_rate": 0.003,
    "threshold": 0.06,                # ✅ 動態再平衡偏離門檻
    "rebalance_freq": 'Y',           # ✅ 定期再平衡頻率（'M'=月、'Q'=季、'Y'=年）
    "risk_free_rate": 0.01,
    "output_file": "rebalance_trades.csv"
}

# ----------------------------
# (B) 工具函式（資料讀取與成本計算）
# ----------------------------
def read_weights_from_csv(csv_file_path):
    weights_df = pd.read_csv(csv_file_path)
    return dict(zip(weights_df['Ticker'], weights_df['Weight']))

def calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=False):
    commission = trade_value * commission_rate
    tax = trade_value * tax_rate if is_sell else 0
    return commission, tax

def validate_trade(cash, trade_value, commission, tax):
    return cash >= (trade_value + commission + tax)

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        if 'Close' not in df.columns:
            continue
        df = df[['Close']]
        df.columns = [stock_name]
        df = df[~df.index.duplicated(keep='first')]
        full_date_range = pd.date_range(start=start_date, end=end_date)
        df = df.reindex(full_date_range).interpolate(method='linear')
        df = df.loc[start_date:end_date]
        if not df.empty:
            df_list.append(df)
    return pd.concat(df_list, axis=1).dropna(how='all') if df_list else pd.DataFrame()

# ----------------------------
# (C) 主再平衡邏輯（定期 + 動態）
# ----------------------------
def rebalance_portfolio_dynamic(df_prices, weights, config):
    # 參數初始化
    initial_investment = config["initial_investment"]
    commission_rate = config["commission_rate"]
    tax_rate = config["tax_rate"]
    threshold = config["threshold"]
    rebalance_freq = config["rebalance_freq"]  # ✅ 這裡控制定期再平衡頻率
    risk_free_rate = config["risk_free_rate"]
    output_file = config["output_file"]

    cash = initial_investment
    shares = {}
    all_trades = []
    daily_values = []
    triggered_by_threshold = {}
    all_triggered_stock_names = set()

    trading_days = df_prices.index
    regular_rebalance_days = pd.date_range(start=trading_days.min(), end=trading_days.max(), freq=rebalance_freq)
    regular_rebalance_days = set(regular_rebalance_days.intersection(trading_days))

    # ✅（內部函式）執行一次再平衡（定期或動態皆可共用）
    def do_rebalance(rebalance_date, full_rebalance):
        nonlocal cash
        prices_today = df_prices.loc[rebalance_date]
        portfolio_value = sum(shares.get(stock, 0) * prices_today[stock] for stock in shares if not pd.isna(prices_today[stock])) + cash
        new_shares = {}

        for stock, w in weights.items():
            if pd.isna(prices_today[stock]) or prices_today[stock] == 0:
                new_shares[stock] = shares.get(stock, 0)
                continue

            current_holding = shares.get(stock, 0)
            current_weight = (current_holding * prices_today[stock]) / portfolio_value if portfolio_value > 0 else 0

            # 若非定期再平衡，且偏離未超過門檻，則跳過
            if not full_rebalance and abs(current_weight - w) <= threshold:
                new_shares[stock] = current_holding
                continue

            # 計算應買/賣股數
            target_value = portfolio_value * w
            target_shares = target_value / prices_today[stock]
            share_diff = target_shares - current_holding
            trade_value = abs(share_diff) * prices_today[stock]

            # 賣出
            if share_diff < 0:
                commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=True)
                cash += (trade_value - commission - tax)
                all_trades.append([rebalance_date, stock, "Sell", -share_diff, prices_today[stock], commission, tax])
            # 買入
            elif share_diff > 0:
                commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate)
                if validate_trade(cash, trade_value, commission, tax):
                    cash -= (trade_value + commission + tax)
                    all_trades.append([rebalance_date, stock, "Buy", share_diff, prices_today[stock], commission, tax])
                else:
                    target_shares = current_holding  # 現金不足則不買

            new_shares[stock] = target_shares

        # 出清已不在投資組合中的股票
        for old_stock in list(shares.keys()):
            if old_stock not in weights:
                old_qty = shares[old_stock]
                if old_qty > 0:
                    trade_value = old_qty * prices_today[old_stock]
                    commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=True)
                    cash += (trade_value - commission - tax)
                    all_trades.append([rebalance_date, old_stock, "Sell", old_qty, prices_today[old_stock], commission, tax])
                new_shares.pop(old_stock, None)

        return new_shares

    # 初始配置（第一天）
    if len(trading_days) > 0:
        shares = do_rebalance(trading_days[0], full_rebalance=True)

    # 模擬每一天
    for i, day in enumerate(trading_days):
        prices_today = df_prices.loc[day]
        portfolio_value = sum(shares.get(stock, 0) * prices_today[stock] for stock in shares if not pd.isna(prices_today[stock])) + cash
        daily_values.append(portfolio_value)

        # 定期 or 動態再平衡
        if (day in regular_rebalance_days) or (triggered_by_threshold.get(day, False)):
            shares = do_rebalance(day, full_rebalance=(day in regular_rebalance_days))

        # 判斷明天是否要觸發動態再平衡
        if i < len(trading_days) - 1:
            exceed_threshold = False
            triggered_stocks = []

            if portfolio_value > 0:
                for stock in weights:
                    if pd.isna(prices_today[stock]) or prices_today[stock] == 0:
                        continue
                    current_holding = shares.get(stock, 0)
                    actual_weight = (current_holding * prices_today[stock]) / portfolio_value
                    target_weight = weights.get(stock, 0)
                    deviation = abs(actual_weight - target_weight)

                    if deviation > threshold:
                        exceed_threshold = True
                        triggered_stocks.append((stock, actual_weight, target_weight, deviation))

            if exceed_threshold:
                next_day = trading_days[i + 1]
                triggered_by_threshold[next_day] = True
                all_triggered_stock_names.update(s[0] for s in triggered_stocks)

    # 出清所有持股
    if len(trading_days) > 0:
        final_day = trading_days[-1]
        prices_final = df_prices.loc[final_day]
        for stock in list(shares.keys()):
            if shares[stock] > 0:
                qty = shares[stock]
                trade_value = qty * prices_final[stock]
                commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=True)
                cash += (trade_value - commission - tax)
                all_trades.append([final_day, stock, "Sell", qty, prices_final[stock], commission, tax])

    # 儲存交易紀錄
    trades_df = pd.DataFrame(all_trades, columns=['Date', 'Stock', 'Action', 'Shares', 'Price', 'Commission', 'Tax'])
    trades_df.to_csv(output_file, index=False)

    # 計算績效指標
    final_value = cash
    total_return = final_value - initial_investment
    return_percentage = (total_return / initial_investment) * 100
    daily_returns = pd.Series(daily_values).pct_change().dropna()
    n_years = (df_prices.index[-1] - df_prices.index[0]).days / 365.25
    annualized_return = (final_value / initial_investment) ** (1 / n_years) - 1
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else np.nan
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    # 印出績效
    print("\n📊 Performance Metrics:")
    for k, v in {
        'Total Final Value': final_value,
        'Total Return': total_return,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Final Cash': cash
    }.items():
        print(f"{k}: {v:.4f}")

    if all_triggered_stock_names:
        print("\n📌 曾觸發過動態再平衡的股票：")
        print(", ".join(sorted(all_triggered_stock_names)))
    else:
        print("\n✅ 無任何股票觸發過動態再平衡")

    return {
        'Final Value': final_value,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }

# ----------------------------
# (D) 主程式入口
# ----------------------------
if __name__ == "__main__":
    weights = read_weights_from_csv(CONFIG["weights_csv_path"])
    csv_files = glob.glob(os.path.join(CONFIG["data_folder_path"], '*.csv'))
    df_prices = process_data(csv_files, CONFIG["start_date"], CONFIG["end_date"])

    if not df_prices.empty:
        rebalance_portfolio_dynamic(df_prices, weights, CONFIG)
    else:
        print("❌ 無法處理任何資料，請確認路徑與資料格式正確。")
