'''
本程式目的：
===========
用來依據不同「時間區間」與「再平衡頻率」，針對台股歷史股價資料進行：
1. 計算各頻率下的報酬率相關係數矩陣。
2. 根據設定的負相關門檻（如 correlation < -0.9），選出負相關股票對。
3. 使用 PyPortfolioOpt 套件的 HRP（Hierarchical Risk Parity）方法，根據所選股票計算投資權重。
4. 輸出以下結果：
   - 負相關股票的相關係數矩陣（CSV）
   - HRP 計算後的投資組合權重（CSV）
   - 該投資組合的原始股價 CSV（供後續回測使用）

'''

import os, glob, shutil, numpy as np, pandas as pd
from pypfopt import HRPOpt

# --------------------------------------------------
# 0. 全域參數
# --------------------------------------------------
price_folder = r'C:/Users/User/Desktop/台股資料2004-2023/test'      # 原始股價 CSV
neg_thresh   = -0.9                                                # corr < -0.1
base_out_dir = r'C:/Users/User/Desktop/123/更換成分股_多頻率/1'         # 輸出根目錄
float_fmt    = '%.9f'                                               # 相關矩陣 CSV 精度

# 三段訓練期
configs = [
    dict(label='A', start='2008-01-01', end='2015-12-31'),
    dict(label='B', start='2011-01-01', end='2018-12-31'),
    dict(label='C', start='2014-01-01', end='2021-12-31')
]

# 再平衡／相關矩陣頻率
freqs = {
    'Y' : 'Y',     # 年
    '6M': '2Q',    # 半年
    'Q' : 'Q',     # 季
    'M' : 'M'      # 月
}
# --------------------------------------------------


# ----------------- 共用函式 -----------------
def load_prices(folder: str, start: str, end: str) -> pd.DataFrame:
    """一次讀入所有 CSV，補日期缺值並回傳收盤價表"""
    dfs = []
    for f in glob.glob(os.path.join(folder, '*.csv')):
        tic = os.path.basename(f).split('.')[0]
        df  = pd.read_csv(f, usecols=['Date', 'Close'],
                          index_col='Date', parse_dates=True)
        df  = df[~df.index.duplicated(keep='first')].loc[start:end]
        if df.empty:
            continue
        df  = df.reindex(pd.date_range(start, end)).interpolate('linear')
        df.columns = [tic]
        dfs.append(df)
    if not dfs:
        raise ValueError("❌ 找不到任何股價資料")
    return pd.concat(dfs, axis=1).dropna(how='all')


def calc_corr(prices: pd.DataFrame, rule: str):
    """依頻率 rule 聚合 → 報酬率 → 相關矩陣，並回傳報酬率表"""
    returns = prices.resample(rule).last().pct_change().dropna()
    return returns.corr(), returns


def neg_set(corr: pd.DataFrame, th: float):
    """找出 corr < th 的股票集合（上三角）"""
    mask = np.triu(np.ones_like(corr, bool), 1)
    r, c = np.where((corr.values < th) & mask)
    return set(corr.index[r]).union(corr.columns[c])


def copy_csv(stocks, src, dst):
    os.makedirs(dst, exist_ok=True)
    for s in stocks:
        f = os.path.join(src, f'{s}.csv')
        if os.path.exists(f):
            shutil.copy(f, os.path.join(dst, f'{s}.csv'))
# --------------------------------------------------


# ----------------- 主流程 -----------------
if __name__ == '__main__':

    for cfg in configs:
        lab, s_date, e_date = cfg['label'], cfg['start'], cfg['end']
        print(f"\n=== 區段 {lab}  {s_date} ~ {e_date} ===")

        prices = load_prices(price_folder, s_date, e_date)
        print("• 價格表維度 :", prices.shape)

        for tag, rule in freqs.items():
            print(f"  ── 處理頻率 {tag} ({rule})")
            out_dir = os.path.join(base_out_dir, f"{tag}_負{abs(neg_thresh):.1f}{lab}")
            os.makedirs(out_dir, exist_ok=True)

            # 1) 相關矩陣 (依 rule 聚合)
            corr, _ = calc_corr(prices, rule)
            corr.to_csv(os.path.join(out_dir,
                                     f"matrix_{s_date[:4]}_{e_date[:4]}.csv"),
                        float_format=float_fmt)

            # 2) 挑負相關股票
            picks = neg_set(corr, neg_thresh)
            print(f"     • 負相關股票數: {len(picks)}")
            if not picks:
                continue

            # 3) ★ HRP 權重：
            daily_ret = prices[sorted(picks)].pct_change().dropna()
            hrp = HRPOpt(daily_ret)
            hrp.optimize()

            pd.DataFrame(hrp.clean_weights().items(),
                         columns=['Ticker', 'Weight']
                         ).to_csv(os.path.join(out_dir,
                                               f"weights_{s_date[:4]}_{e_date[:4]}.csv"),
                                  index=False)

            # 4) 複製原始 CSV 方便後續回測
            copy_csv(picks, price_folder, out_dir)
            print("     • 已輸出至", out_dir)

    print("\n✓ 所有頻率與區段已完成")