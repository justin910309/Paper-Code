"""
📈 return_corr_matrix_by_freq.py
根據指定頻率（年、半年、季、月）計算股票報酬率的相關係數矩陣，並輸出為 CSV。

適用頻率範例：
    - 年: '1Y'
    - 半年: '6M'
    - 季: '3M'
    - 月: '1M'
"""

import os
import glob
import pandas as pd

# --------------------------
# 🔧 使用者設定參數
# --------------------------
FOLDER_PATH = 'C:/Users/User/Desktop/台股資料2004-2023/test'
OUTPUT_DIR = 'C:/Users/User/Desktop/123/matrix'
START_DATE = '2008-01-01'
END_DATE = '2015-12-31'
FREQUENCY = '1Y'  # 可選 '1Y'、'6M'、'3M'、'1M'

# --------------------------
# 📦 讀取與處理股價資料
# --------------------------
def process_data(files, start_date, end_date):
    df_list = []
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    for file in files:
        stock_name = os.path.basename(file).split('.')[0]
        df = pd.read_csv(file, index_col='Date', parse_dates=True)

        if 'Close' not in df.columns:
            print(f"[⚠️警告] {file} 缺少 Close 欄位，已略過")
            continue

        df = df[['Close']]
        df.columns = [stock_name]

        df = df[~df.index.duplicated(keep='first')]
        full_date_range = pd.date_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(full_date_range).interpolate(method='linear')

        df = df.loc[start_date:end_date]

        if not df.empty:
            df_list.append(df)
        else:
            print(f"[ℹ️訊息] {stock_name} 在指定期間內無資料")

    if not df_list:
        print("[❌錯誤] 無有效資料")
        return pd.DataFrame()

    df_prices = pd.concat(df_list, axis=1)
    return df_prices

# --------------------------
# 📊 計算報酬率相關係數矩陣（依指定頻率）
# --------------------------
def calculate_correlation_matrix(df_prices, freq):
    prices = df_prices.resample(freq).last()
    returns = prices.pct_change().dropna()

    if returns.empty:
        print("[❌錯誤] 無法計算報酬率：資料為空")
        return pd.DataFrame()

    corr_matrix = returns.corr()
    return corr_matrix

# --------------------------
# 🚀 主程式流程
# --------------------------
if __name__ == "__main__":
    print(f"▶ 開始計算 {FREQUENCY} 頻率下的報酬率相關係數矩陣...")
    csv_files = glob.glob(f'{FOLDER_PATH}/*.csv')
    processed_data = process_data(csv_files, START_DATE, END_DATE)

    if not processed_data.empty:
        print(f"[✅成功] 整理後資料形狀：{processed_data.shape}")
        corr_matrix = calculate_correlation_matrix(processed_data, FREQUENCY)

        if not corr_matrix.empty:
            filename = f'matrix_{FREQUENCY}.csv'.replace('/', '')
            output_path = os.path.join(OUTPUT_DIR, filename)
            corr_matrix.to_csv(output_path)
            print(f"[💾已儲存] 矩陣儲存於：{output_path}")
        else:
            print("[⚠️提醒] 無法輸出：相關係數矩陣為空")
    else:
        print("[❌錯誤] 無可用股價資料")
