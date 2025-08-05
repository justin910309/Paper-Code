# 本程式用於回測結合「負相關股票 + HRP 權重 + 再平衡機制」的投資策略，
# 支援不同資料頻率（年、半年、季、月）下的績效分析。
# 核心功能為：
# 1. 載入各時期股票價格資料。
# 2. 根據指定的負相關門檻（如 -0.9, -0.8, -0.7）與權重檔進行多期回測。
# 3. 每隔固定再平衡頻率（如：每月、每季）調整投資組合至目標權重。
# 4. 計算最終資產值、總報酬率、年化報酬、Sharpe比率與最大跌幅。
# 5. 將結果以表格與圖表形式輸出，協助比較不同條件下的績效表現。

import os, glob, pandas as pd, numpy as np
import matplotlib.pyplot as plt

# ---------- 使用者設定 ----------
price_dir = r'C:/Users/User/Desktop/台股資料2004-2023/test'

# ⚠️ 調整這裡的參數以變更「再平衡頻率」
reb_freq = 'Y'                       # 'Y' 年、'6M' 半年、'Q' 季、'M' 月
out_dir  = rf'C:/Users/User/Desktop/123/回測結果_多頻率/報酬圖'
os.makedirs(out_dir, exist_ok=True)

# 對應不同負相關門檻的訓練資料路徑（後綴為 A/B/C 對應不同年份）
coef_roots  = {
    '0.9': r'C:/Users/User/Desktop/123/更換成分股_多頻率/Y_負0.9',
    '0.8': r'C:/Users/User/Desktop/123/更換成分股_多頻率/Y_負0.8',
    '0.7': r'C:/Users/User/Desktop/123/更換成分股_多頻率/Y_負0.7',
}

# 權重檔名稱對應
name_map = {'A': 'weights_2008_2015.csv',
            'B': 'weights_2011_2018.csv',
            'C': 'weights_2014_2021.csv'}

# 測試區間分段設計（用於三段回測）
segments = {'A': ('2016-01-01', '2018-12-31'),
            'B': ('2019-01-01', '2021-12-31'),
            'C': ('2022-01-01', '2023-12-31')}

# 回測參數設定
initial_capital = 100_000
commission_rate = 0.001425
tax_rate        = 0.003
risk_free       = 0.01
# ---------------------------------

# ---------- 共用函式 ----------
def load_prices(folder, start, end):
    """載入所有股票在指定時間區間的收盤價，並補齊缺失資料"""
    dfs = []
    for fp in glob.glob(os.path.join(folder, '*.csv')):
        tic = os.path.basename(fp).split('.')[0]
        df  = pd.read_csv(fp, usecols=['Date', 'Close'],
                          index_col='Date', parse_dates=True)
        df  = df[~df.index.duplicated(keep='first')].loc[start:end]
        if df.empty: continue
        df  = df.reindex(pd.date_range(start, end)).interpolate('linear')
        df.columns = [tic]
        dfs.append(df)
    if not dfs:
        raise ValueError('❌ 無股價檔')
    return pd.concat(dfs, axis=1).dropna(how='all')

def read_weights(csv_path):
    """讀取每檔股票的權重並正規化"""
    w = pd.read_csv(csv_path).set_index('Ticker')['Weight'].to_dict()
    s = sum(w.values())
    return {k: v/s for k, v in w.items()}

def trade_cost(value, sell=False):
    """計算單筆交易的手續費與稅費"""
    comm = value * commission_rate
    tax  = value * tax_rate if sell else 0
    return comm, tax
# ---------------------------------

# ---------- 回測主邏輯：回傳資產曲線與績效指標 ----------
def rebalance_full(df, sched, freq='M', ini=initial_capital):
    cash, sh = ini, {}
    daily    = [ini]              # 資產值
    date_idx = [df.index[0]]      # 日期索引 ——★ 同步維護

    # 產生再平衡日期
    reb = list(pd.date_range(df.index.min(), df.index.max(), freq=freq))
    for s, _, _ in sched:
        if df.index.min() <= s <= df.index.max():
            adj = s if s in df.index else df[df.index >= s].index[0]
            reb.append(adj)
    reb = sorted(set(reb))

    for i in range(len(reb)-1):
        ts = reb[i]
        te = reb[i+1] if i < len(reb)-2 else df.index[-1]

        # 若調整日缺值 → 用該年最後交易日
        if ts not in df.index:
            ts = df[df.index.year == ts.year].last_valid_index()
        if te not in df.index:
            te = df[df.index.year == te.year].last_valid_index()

        px  = df.loc[ts]
        wt  = next(w for s, e, w in sched if s <= ts <= e)

        # (a) 賣掉剔除股票
        for old in list(sh):
            if old not in wt:
                v = sh[old] * px[old]
                c, t = trade_cost(v, sell=True)
                cash += v - c - t
                sh.pop(old)

        # (b) 調整至目標權重
        port_val = sum(sh[s] * px[s] for s in sh) + cash
        new_sh   = {}
        for stk, w in wt.items():
            tgt = port_val * w / px[stk]
            diff = tgt - sh.get(stk, 0)
            if abs(diff) < 1e-4:
                new_sh[stk] = sh.get(stk, 0)
                continue
            v = abs(diff) * px[stk]
            c, t = trade_cost(v, sell=(diff < 0))
            if diff < 0:
                cash += v - c - t
            else:
                if cash >= v + c + t:
                    cash -= v + c + t
                else:
                    tgt = sh.get(stk, 0)          # 現金不足
            new_sh[stk] = tgt
        sh = new_sh

        # (c) 每日計算資產值
        for d, row in df.loc[ts:te].iterrows():
            daily.append(sum(sh[s] * row[s] for s in sh) + cash)
            date_idx.append(d)

    # 期末清倉
    px_end = df.iloc[-1]
    for stk, qty in sh.items():
        if qty > 0:
            v = qty * px_end[stk]
            c, t = trade_cost(v, sell=True)
            cash += v - c - t
    daily[-1] = cash

    curve = pd.Series(daily, index=date_idx)

    # 計算績效指標
    dr     = curve.pct_change().dropna()
    days   = (curve.index[-1] - curve.index[0]).days
    tot_rt = cash - ini
    ann_rt = (cash / ini) ** (365 / days) - 1
    sharpe = np.sqrt(252) * (dr.mean() - risk_free / 252) / dr.std()
    mdd    = (1 + dr).cumprod()
    mdd    = (mdd / mdd.cummax() - 1).min()

    metrics = dict(Final_Value=cash,
                   Total_Return=tot_rt,
                   Return_Pct=tot_rt/ini,
                   Annual_Return=ann_rt,
                   Sharpe_Ratio=sharpe,
                   Max_Drawdown=mdd)
    return curve, metrics
# ---------------------------------

# -------------- 主流程 --------------
prices_all = load_prices(price_dir, '2016-01-01', '2023-12-31')
rows, curves = [], {}

for coef, root in coef_roots.items():
    sched = []
    for seg, (s, e) in segments.items():
        w_csv = os.path.join(f"{root}{seg}", name_map[seg])
        sched.append((pd.Timestamp(s), pd.Timestamp(e), read_weights(w_csv)))

    curve, met = rebalance_full(prices_all, sched, reb_freq)
    curves[coef] = curve / curve.iloc[0]          # 標準化
    met['Coefficient'] = coef
    rows.append(met)

    print(f"[coef {coef}] Final {met['Final_Value']:.0f}  "
          f"TotRet {met['Total_Return']:.0f}({met['Return_Pct']:.2%})  "
          f"AnnRet {met['Annual_Return']:.2%}  "
          f"Sharpe {met['Sharpe_Ratio']:.2f}  "
          f"MDD {met['Max_Drawdown']:.2%}")

# ---- 輸出 CSV ----
summary_csv = os.path.join(out_dir, f'summary_{reb_freq}_multi_coef.csv')
pd.DataFrame(rows).to_csv(summary_csv, index=False)
print("✓ Summary →", summary_csv)

# ---- 畫累積報酬 ----
df_eq = pd.DataFrame(curves)
df_eq_pct = (df_eq - 1) * 100          # ← 轉成百分比

plt.figure(figsize=(11, 5))
for col in df_eq_pct.columns:
    plt.plot(df_eq_pct.index, df_eq_pct[col],
             label=f'cor < - {col}', linewidth=1.3)

plt.axhline(0, color='red', linestyle='--', linewidth=1,
            label='Initial Asset(%)')

plt.title('Cumulative Returns', fontsize=14, pad=10)
plt.xlabel('Time')
plt.ylabel('Cumulative Returns (%)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

img_path = os.path.join(out_dir,
                        f'equity_{reb_freq}_multi_coef.png')
plt.savefig(img_path, dpi=150)
plt.show()
print('✓ Equity curve →', img_path)
