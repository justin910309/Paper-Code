# ===============================================
# 📈 程式名稱：dynamic_rebalance_with_threshold_comparison.py
# 🧠 功能簡介：
#   - 對固定權重的投資組合進行「定期再平衡（每月）」與「動態再平衡（超過 threshold）」
#   - 測試多組 threshold 值（例如 0.01～0.06）對策略績效的影響
#   - 輸出績效比較表格（CSV）與累積報酬圖（PNG）
# ===============================================

import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------- (A) 可調整參數 ----------------------------
CONFIG = {
    "weights_csv_path": "C:/Users/User/Desktop/123/新權重矩陣改過/Optimal_Weights_2008_2015月負0.2.csv",
    "data_folder_path": "C:/Users/User/Desktop/台股資料2004-2023/test",
    "start_date": "2016-01-01",
    "end_date": "2023-12-31",
    "initial_investment": 100000,
    "commission_rate": 0.001425,
    "tax_rate": 0.003,
    "risk_free_rate": 0.01,
    "rebalance_frequency": 'M'  # 每月定期再平衡
}
thresholds_to_test = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

# ---------------------------- (B) 工具函式 ----------------------------
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
        df = pd.read_csv(file, index_col='Date', parse_dates=True)[['Close']]
        df.columns = [stock_name]
        df = df[~df.index.duplicated(keep='first')]
        full_date_range = pd.date_range(start=start_date, end=end_date)
        df = df.reindex(full_date_range).interpolate(method='linear')
        df = df.loc[start_date:end_date]
        if not df.empty:
            df_list.append(df)
    return pd.concat(df_list, axis=1).dropna(how='all') if df_list else pd.DataFrame()

# ---------------------------- (C) 再平衡邏輯 ----------------------------
def rebalance_portfolio_dynamic(df_prices, weights, config):
    initial_investment = config["initial_investment"]
    commission_rate = config["commission_rate"]
    tax_rate = config["tax_rate"]
    threshold = config["threshold"]
    risk_free_rate = config["risk_free_rate"]
    rebalance_freq = config["rebalance_frequency"]

    cash = initial_investment
    shares = {}
    all_trades = []
    daily_values = []
    triggered_stocks = set()

    trading_days = df_prices.index
    rebalance_days = pd.date_range(start=trading_days.min(), end=trading_days.max(), freq=rebalance_freq)
    rebalance_days = set(rebalance_days.intersection(trading_days))

    def do_rebalance(rebalance_date, full_rebalance):
        nonlocal cash
        prices_today = df_prices.loc[rebalance_date]
        portfolio_value = sum(shares.get(stock, 0) * prices_today[stock]
                              for stock in shares if not pd.isna(prices_today[stock])) + cash
        new_shares = {}
        for stock, w in weights.items():
            if pd.isna(prices_today[stock]) or prices_today[stock] == 0:
                new_shares[stock] = shares.get(stock, 0)
                continue
            current_holding = shares.get(stock, 0)
            current_weight = (current_holding * prices_today[stock]) / portfolio_value if portfolio_value > 0 else 0
            if not full_rebalance and abs(current_weight - w) <= threshold:
                new_shares[stock] = current_holding
                continue
            target_value = portfolio_value * w
            target_shares = target_value / prices_today[stock]
            share_diff = target_shares - current_holding
            trade_value = abs(share_diff) * prices_today[stock]
            if share_diff < 0:
                commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=True)
                cash += (trade_value - commission - tax)
                all_trades.append([rebalance_date, stock, "Sell", -share_diff, prices_today[stock], commission, tax])
            elif share_diff > 0:
                commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate)
                if validate_trade(cash, trade_value, commission, tax):
                    cash -= (trade_value + commission + tax)
                    all_trades.append([rebalance_date, stock, "Buy", share_diff, prices_today[stock], commission, tax])
                else:
                    target_shares = current_holding
            new_shares[stock] = target_shares
        return new_shares

    if len(trading_days) > 0:
        shares = do_rebalance(trading_days[0], full_rebalance=True)

    for i, day in enumerate(trading_days):
        prices_today = df_prices.loc[day]
        portfolio_value = sum(shares.get(stock, 0) * prices_today[stock]
                              for stock in shares if not pd.isna(prices_today[stock])) + cash
        daily_values.append(portfolio_value)

        if day in rebalance_days:
            shares = do_rebalance(day, full_rebalance=True)

        if i < len(trading_days) - 1:
            for stock in weights:
                if pd.isna(prices_today[stock]) or prices_today[stock] == 0:
                    continue
                current_holding = shares.get(stock, 0)
                actual_weight = (current_holding * prices_today[stock]) / portfolio_value if portfolio_value > 0 else 0
                if abs(actual_weight - weights[stock]) > threshold:
                    triggered_stocks.add(stock)
                    shares = do_rebalance(trading_days[i + 1], full_rebalance=False)
                    break

    final_day = trading_days[-1]
    prices_final = df_prices.loc[final_day]
    for stock in list(shares.keys()):
        if shares[stock] > 0:
            qty = shares[stock]
            trade_value = qty * prices_final[stock]
            commission, tax = calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=True)
            cash += (trade_value - commission - tax)
            all_trades.append([final_day, stock, "Sell", qty, prices_final[stock], commission, tax])

    final_value = cash
    return_percentage = (final_value - initial_investment) / initial_investment * 100
    daily_returns = pd.Series(daily_values, index=df_prices.index[:len(daily_values)]).pct_change().dropna()
    n_years = (df_prices.index[-1] - df_prices.index[0]).days / 365.25
    annualized_return = (final_value / initial_investment) ** (1 / n_years) - 1
    sharpe_ratio = (daily_returns.mean() - (risk_free_rate / 252)) / daily_returns.std() * np.sqrt(252)
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    return {
        'Threshold': threshold,
        'Return Percentage': return_percentage,
        'Annualized Return': annualized_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Triggered Stocks': ", ".join(sorted(triggered_stocks)),
        'Daily Returns': daily_returns
    }

# ---------------------------- (D) 畫圖函式 ----------------------------
def plot_cumulative_return_curves(metrics_list, thresholds, save_path=None):
    plt.figure(figsize=(10, 6))
    for metrics, threshold in zip(metrics_list, thresholds):
        daily_returns = metrics['Daily Returns']
        cum_returns = (1 + daily_returns).cumprod() - 1
        plt.plot(daily_returns.index, cum_returns * 100, label=f"Threshold={threshold:.2f}")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Cumulative Return Curves for Different Thresholds")
    plt.xlabel("Time (Year)")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 圖片已儲存至：{save_path}")
    plt.show()

# ---------------------------- (E) 主程式 ----------------------------
if __name__ == "__main__":
    results = []

    weights = read_weights_from_csv(CONFIG["weights_csv_path"])
    csv_files = glob.glob(os.path.join(CONFIG["data_folder_path"], '*.csv'))
    df_prices = process_data(csv_files, CONFIG["start_date"], CONFIG["end_date"])

    if not df_prices.empty:
        for t in thresholds_to_test:
            current_config = CONFIG.copy()
            current_config["threshold"] = t
            metrics = rebalance_portfolio_dynamic(df_prices, weights, current_config)
            results.append(metrics)

        result_df = pd.DataFrame(results)
        print("\n📊 不同閾值下的績效比較：")
        print(result_df)

        output_dir = "C:/Users/User/Desktop/123/動態再平衡結果/累積報酬圖"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(CONFIG["weights_csv_path"]))[0]
        result_csv_path = os.path.join(output_dir, f"{base_name}_dynamic_threshold_comparison.csv")
        result_df.to_csv(result_csv_path, index=False)

        image_path = os.path.join(output_dir, f"{base_name}_cumulative_returns.png")
        plot_cumulative_return_curves(results, thresholds_to_test, save_path=image_path)
    else:
        print("⚠️ 無可用資料，請檢查資料來源或日期區間。")
