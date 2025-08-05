"""
📁 select_negative_corr_stocks.py

功能：
根據已計算的相關係數矩陣（如半年報酬率），找出彼此高度負相關（如 < -0.5）的股票對，
並將這些股票的資料檔案從來源資料夾複製到指定目標資料夾中，以供後續投資組合建構使用。
"""

import pandas as pd
import os
import shutil

# --------------------------
# 🔧 使用者設定參數
# --------------------------
# 輸入的相關係數矩陣 CSV
CORR_MATRIX_PATH = 'C:/Users/User/Desktop/123/matrix/matrix台2008-2015年報酬率.csv'

# 股票資料來源資料夾
SOURCE_FOLDER = 'C:/Users/User/Desktop/台股資料2004-2023/test'

# 篩選後的資料存放目標資料夾
TARGET_FOLDER = 'C:/Users/User/Desktop/123/符合條件股票/年負0.9'

# 設定負相關閾值（例如 -0.5）
NEG_CORR_THRESHOLD = -0.9

# --------------------------
# 📥 讀取相關係數矩陣
# --------------------------
df_corr = pd.read_csv(CORR_MATRIX_PATH, index_col=0)
print(f"✅ 已讀取相關係數矩陣，股票數量：{len(df_corr)}")

# --------------------------
# 🎯 找出符合條件的負相關股票對
# --------------------------
selected_stocks = set()

for i in range(len(df_corr.index)):
    for j in range(i + 1, len(df_corr.columns)):
        corr_value = df_corr.iat[i, j]
        if corr_value < NEG_CORR_THRESHOLD:
            selected_stocks.add(df_corr.index[i])
            selected_stocks.add(df_corr.columns[j])

print(f"📌 符合條件的股票共 {len(selected_stocks)} 檔")

# --------------------------
# 📁 複製符合條件的股票 CSV 到目標資料夾
# --------------------------
os.makedirs(TARGET_FOLDER, exist_ok=True)
copied = 0
missing = []

for stock in selected_stocks:
    source_file = os.path.join(SOURCE_FOLDER, f"{stock}.csv")
    target_file = os.path.join(TARGET_FOLDER, f"{stock}.csv")

    if os.path.exists(source_file):
        shutil.copy(source_file, target_file)
        copied += 1
    else:
        missing.append(stock)

print(f"✅ 已複製 {copied} 檔股票資料至：{TARGET_FOLDER}")

if missing:
    print(f"⚠️ 找不到以下檔案，共 {len(missing)} 檔：")
    for m in missing:
        print(f"  - {m}.csv")

print("🎉 所有選股檔案處理完畢！")
