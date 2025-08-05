"""
📁 optimize_hrp_weights.py

功能：
1. 根據選定股票資料計算報酬率
2. 使用 HRP（Hierarchical Risk Parity）進行資產配置
3. 輸出投資組合權重到 CSV
4. 顯示投資組合預期績效（年報酬率、波動率、Sharpe 比率）
"""

import os
import pandas as pd
import numpy as np
import glob
from pypfopt import HRPOpt

# --------------------------
# 🔧 使用者參數設定
# --------------------------
INPUT_FOLDER = 'C:/Users/User/Desktop/123/符合條件股票/年負0.7'
OUTPUT_CSV_PATH = 'C:/Users/User/Desktop/123/新權重矩陣改過/年負0.7.csv'
START_DATE = '2008-01-01'
END_DATE = '2015-12-31'

# --------------------------
# 📦 資料處理函式
# --------------------------
def process_data(files, start_date, end_date):
    df_list = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        df = df[['Close']]
        df.columns = [stock_name]

        df = df.sort_index()
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        if df.index.min() > start_date:
            continue

        df = df.loc[start_date:end_date]

        if not df.empty:
            df_list.append(df)

    if not df_list:
        print("[❌錯誤] 沒有可用資料")
        return pd.DataFrame()

    df_prices = pd.concat(df_list, axis=1)
    df_prices.interpolate(method='linear', inplace=True)
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prices.dropna(inplace=True)

    returns = df_prices.pct_change().dropna()
    return returns

# --------------------------
# 🚀 主流程
# --------------------------
if __name__ == "__main__":
    print("▶ 開始使用 HRP 進行權重優化...")
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    returns = process_data(csv_files, START_DATE, END_DATE)

    if not returns.empty:
        # 使用 HRP 進行配置
        hrp = HRPOpt(returns)
        hrp.optimize()
        weights = hrp.clean_weights()

        # 儲存權重
        weights_df = pd.DataFrame(list(weights.items()), columns=['Ticker', 'Weight'])
        weights_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"✅ 權重已儲存至：{OUTPUT_CSV_PATH}")

        # 顯示投資組合績效
        perf = hrp.portfolio_performance(verbose=True)

    else:
        print("[⚠️提醒] 報酬率資料為空，請確認股票資料與日期範圍。")
