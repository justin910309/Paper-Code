import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ====== ✅ 參數設定 ======
initial_investment = 100000
commission_rate = 0.001425
tax_rate = 0.003
start_date = "2016-01-01"
end_date = "2023-12-31"
data_folder_path = 'C:/Users/User/Desktop/台股資料2004-2023/test'
weights_csv_paths = [
    'C:/Users/User/Desktop/123/新權重矩陣改過/Optimal_Weights_2008_2015年負0.9.csv',
    'C:/Users/User/Desktop/123/新權重矩陣改過/Optimal_Weights_2008_2015年負0.8.csv',
    'C:/Users/User/Desktop/123/新權重矩陣改過/Optimal_Weights_2008_2015年負0.7.csv',
]
line_names = ['Cor. <-0.9', 'Cor. <-0.8', 'Cor. <-0.7']
rebalance_freq = 'Y'  # ✅ 再平衡頻率：'M'=月、'Q'=季、'2Q'=雙季、'6M'=半年、'Y'=年

# ====== 📘 函式定義 ======
def read_weights_from_csv(csv_file_path):
    weights_df = pd.read_csv(csv_file_path)
    return dict(zip(weights_df['Ticker'], weights_df['Weight']))

def calculate_trade_costs(trade_value, commission_rate, tax_rate, is_sell=False):
    commission = trade_value * commission_rate
    tax = trade_value * tax_rate if is_sell else 0
    return commission, tax

def validate_trade(cash, trade_value, commission, tax):
    return cash >= trade_value + commission + tax

def process_data(files, start_date, end_date):
    df_list = []
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        full_date_range = pd.date_range(start=start_date, end=end_date)
        df = df.reindex(full_date_range).interpolate(method='linear')
        df = df.loc[start_date:end_date]
        if not df.empty:
            df_list.append(df)
    return pd.concat(df_list, axis=1).dropna() if df_list else pd.DataFrame()

def rebalance_portfolio(df_prices, weights, rebalance_freq):
    cash = initial_investment
    shares = {}
    all_trades = []
    daily_values = []

    rebalance_dates = pd.date_range(start=df_prices.index.min(), end=df_prices.index.max(), freq=rebalance_freq)
    if df_prices.index[0] not in rebalance_dates:
        rebalance_dates = [df_prices.index[0]] + list(rebalance_dates)

    for i in range(len(rebalance_dates) - 1):
        period_start = rebalance_dates[i]
        period_end = rebalance_dates[i + 1] if i < len(rebalance_dates) - 2 else df_prices.index[-1]

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
                        else:
                            if validate_trade(cash, trade_value, commission, tax):
                                cash -= (trade_value + commission + tax)
                            else:
                                target_shares = shares[stock]
                        all_trades.append([period_start, stock, "Sell" if share_diff < 0 else "Buy",
                                           abs(share_diff), prices_at_start[stock], commission, tax])
                new_shares[stock] = target_shares
            shares = new_shares

        for date, prices in period_prices.iterrows():
            portfolio_value = sum(shares[s] * prices[s] for s in shares) + cash
            daily_values.append(portfolio_value)

    final_date = df_prices.index[-1]
    prices_at_end = df_prices.loc[final_date]
    for stock in list(shares.keys()):
        sell_value = shares[stock] * prices_at_end[stock]
        cash += sell_value
        all_trades.append([final_date, stock, "Sell", shares[stock], prices_at_end[stock], 0, 0])
        shares[stock] = 0

    final_value = cash
    trades_df = pd.DataFrame(all_trades, columns=['Date', 'Stock', 'Action', 'Shares', 'Price', 'Commission', 'Tax'])
    trades_df.to_csv('rebalance_trades.csv', index=False)

    min_length = min(len(df_prices.index), len(daily_values))
    cumulative_returns = (np.array(daily_values[:min_length]) / initial_investment) - 1
    return {
        'Final Value': final_value,
        'Cumulative Returns': cumulative_returns
    }

# ====== 📈 主程式：模擬與繪圖 ======
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
    df_prices = process_data(csv_files, start_date, end_date)

    if not df_prices.empty:
        all_cumulative_returns = {}
        for weights_csv_path, line_name in zip(weights_csv_paths, line_names):
            weights = read_weights_from_csv(weights_csv_path)
            result = rebalance_portfolio(df_prices, weights, rebalance_freq)
            all_cumulative_returns[line_name] = result['Cumulative Returns']

        font_prop = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
        plt.figure(figsize=(16, 8))

        for label, returns in all_cumulative_returns.items():
            dates = df_prices.index[:len(returns)]
            plt.plot(dates, returns * 100, linestyle='-', label=label)

        plt.axhline(y=0, color='r', linestyle='--', label='Initial Asset (%)')
        plt.xlabel("Time", fontproperties=font_prop)
        plt.ylabel("Cumulative Returns (%)", fontproperties=font_prop)
        plt.title("Cumulative Returns Comparison", fontproperties=font_prop)
        plt.legend(prop=font_prop)
        plt.grid(True)
        plt.ylim(-50, 200)
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No data available for the specified period.")
