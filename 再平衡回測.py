"""
📁 rebalance_simulation.py

功能：
根據歷史股價與權重資料，模擬投資組合的定期再平衡，計算交易成本與績效指標（總報酬、Sharpe 比率、最大回撤等）。
支援自訂再平衡頻率（年、半年、季、月）。

作者：使用者提供，由 ChatGPT 協助整理註解與模組化
"""

import os
import pandas as pd
import numpy as np
import glob

# ------------------------
# 📌 參數設定區
# ------------------------

weights_csv_path = 'C:/Users/User/Desktop/123/新權重矩陣改過/年負0.9.csv'
data_folder_path = 'C:/Users/User/Desktop/台股資料2004-2023/test'
start_date = "2016-01-01"
end_date = "2023-12-31"
initial_investment = 100000
output_file = 'rebalance_trades.csv'

# 🔁 再平衡頻率參數（可選：'Y'=年, '6M'=半年, '3M'=季, 'M'=月）
REBALANCE_FREQ = 'Y'

# ------------------------
# 📦 工具函式
# ------------------------

def read_weights_from_csv(csv_file_path):
    weights_df = pd.read_csv(csv_file_path)
    weights = dict(zip(weights_df['Ticker'], weights_df['Weight']))
    return weights

def calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=False):
    commission = trade_value * commission_rate
    tax = trade_value * tax_rate if is_sell else 0
    return commission, tax

def validate_trade(cash, trade_value, commission, tax):
    total_cost = trade_value + commission + tax
    return cash >= total_cost

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)[['Close']]
        df.columns = [stock_name]
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        full_date_range = pd.date_range(start=start_date, end=end_date)
        df = df.reindex(full_date_range).interpolate(method='linear')
        df = df.loc[start_date:end_date]
        if not df.empty:
            df_list.append(df)
    if df_list:
        df_prices = pd.concat(df_list, axis=1)
        return df_prices.dropna()
    else:
        return pd.DataFrame()

# ------------------------
# 🔁 再平衡模擬主函式
# ------------------------

def rebalance_portfolio(df_prices, weights, initial_investment=100000,
                        commission_rate=0.001425, tax_rate=0.003,
                        output_file='rebalance_trades.csv',
                        rebalance_freq='Y'):
    cash = initial_investment
    shares = {}
    all_trades = []
    daily_values = [initial_investment]

    rebalance_dates = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq=rebalance_freq)
    if df_prices.index[0] not in rebalance_dates:
        rebalance_dates = [df_prices.index[0]] + list(rebalance_dates)

    for i in range(len(rebalance_dates) - 1):
        period_start = rebalance_dates[i]
        period_end = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else df_prices.index[-1]
        if period_start not in df_prices.index:
            period_start = df_prices.loc[df_prices.index.year == period_start.year].last_valid_index()
        if period_end not in df_prices.index:
            period_end = df_prices.loc[df_prices.index.year == period_end.year].last_valid_index()
        prices_at_start = df_prices.loc[period_start]
        period_prices = df_prices.loc[period_start:period_end]

        if not shares:
            for stock, weight in weights.items():
                target_value = cash * weight
                share_count = target_value / prices_at_start[stock]
                commission, tax = calculate_trade_costs(target_value, commission_rate, tax_rate)
                if validate_trade(cash, target_value, commission, tax):
                    shares[stock] = share_count
                    cash -= (target_value + commission + tax)
                    all_trades.append([period_start, stock, "Buy", share_count,
                                       prices_at_start[stock], commission, tax])
        else:
            current_portfolio_value = sum(shares[s] * prices_at_start[s] for s in shares) + cash
            new_shares = {}
            for stock in weights:
                target_value = current_portfolio_value * weights[stock]
                target_shares = target_value / prices_at_start[stock]
                if stock in shares:
                    share_diff = target_shares - shares[stock]
                    if abs(share_diff) > 0.0001:
                        trade_value = abs(share_diff) * prices_at_start[stock]
                        commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=(share_diff < 0))
                        if share_diff < 0:
                            cash += trade_value - commission - tax
                            all_trades.append([period_start, stock, "Sell", -share_diff,
                                               prices_at_start[stock], commission, tax])
                        else:
                            if validate_trade(cash, trade_value, commission, tax):
                                cash -= (trade_value + commission + tax)
                                all_trades.append([period_start, stock, "Buy", share_diff,
                                                   prices_at_start[stock], commission, tax])
                            else:
                                target_shares = shares[stock]
                else:
                    trade_value = target_shares * prices_at_start[stock]
                    commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate)
                    if validate_trade(cash, trade_value, commission, tax):
                        cash -= (trade_value + commission + tax)
                        all_trades.append([period_start, stock, "Buy", target_shares,
                                           prices_at_start[stock], commission, tax])
                    else:
                        target_shares = 0
                new_shares[stock] = target_shares
            for old_stock in list(shares.keys()):
                if old_stock not in weights:
                    if shares[old_stock] > 0:
                        sell_value = shares[old_stock] * prices_at_start[old_stock]
                        commission, tax = calculate_trade_costs(sell_value, commission_rate, tax_rate, is_sell=True)
                        cash += sell_value - commission - tax
                        all_trades.append([period_start, old_stock, "Sell", shares[old_stock],
                                           prices_at_start[old_stock], commission, tax])
                    new_shares.pop(old_stock, None)
            shares = new_shares

        for date, prices in period_prices.iterrows():
            portfolio_value = sum(shares[s] * prices[s] for s in shares) + cash
            daily_values.append(portfolio_value)

    final_date = df_prices.index[-1]
    prices_at_end = df_prices.loc[final_date]
    for stock in list(shares.keys()):
        if shares[stock] > 0:
            sell_value = shares[stock] * prices_at_end[stock]
            commission, tax = calculate_trade_costs(sell_value, commission_rate, tax_rate, is_sell=True)
            cash += sell_value - commission - tax
            all_trades.append([final_date, stock, "Sell", shares[stock],
                               prices_at_end[stock], commission, tax])

    trades_df = pd.DataFrame(all_trades, columns=['Date', 'Stock', 'Action', 'Shares', 'Price', 'Commission', 'Tax'])
    trades_df.to_csv(output_file, index=False)

    final_value = cash
    total_return = final_value - initial_investment
    return_percentage = (total_return / initial_investment) * 100
    annualized_return = ((final_value / initial_investment) ** (365 / len(df_prices))) - 1
    risk_free_rate = 0.01
    daily_returns = pd.Series(daily_values).pct_change().dropna()
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    performance_metrics = {
        'Total Final Value': final_value,
        'Total Return': total_return,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Final Cash': cash
    }

    print("\n📊 績效指標：")
    for key, value in performance_metrics.items():
        print(f"{key}: {value:.4f}")

    return performance_metrics

# ------------------------
# ▶ 主程式執行區
# ------------------------

if __name__ == "__main__":
    weights = read_weights_from_csv(weights_csv_path)
    csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
    df_prices = process_data(csv_files, start_date, end_date)
    if not df_prices.empty:
        rebalance_portfolio(df_prices, weights,
                            initial_investment=initial_investment,
                            output_file=output_file,
                            rebalance_freq=REBALANCE_FREQ)
    else:
        print("❌ 無有效價格資料")