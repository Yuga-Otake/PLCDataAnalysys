"""
analyzer.py - サイクル検出・遅れ時間計算ロジック（最適化版）

主な最適化:
  - Timestamp を int64 ナノ秒で保持し、浮動小数点除算 1 回に削減
  - 全データ列を numpy 行列 (N×C) に事前変換してサイクル毎のスライスをビュー参照
  - エッジ検出を np.diff() ベクトル演算に統一（DataFrame.loc/normalize_bool_series 廃止）
  - get_cycle_waveforms で ts_ns を事前計算して time_offset_ms 計算を高速化
"""
import pandas as pd
import numpy as np
from math import log10


# ── タイムスタンプ解析 ──────────────────────────────────────────────
def load_csv(file) -> pd.DataFrame:
    """CSVを読み込み、タイムスタンプを変換して返す。

    対応フォーマット:
      - APB Variable ロガー形式: 列名 "Date Time"  2026-02-06 13:26:00.088848
      - 旧形式:                  列名 "Timestamp"   2024/03/01 08:00:00.000
    内部では "Timestamp" 列として統一する。
    """
    # ── 高速パス: APB 標準形式 ──────────────────────────────────
    try:
        df = pd.read_csv(
            file,
            parse_dates=["Date Time"],
            date_format="%Y-%m-%d %H:%M:%S.%f",
        )
        df = df.rename(columns={"Date Time": "Timestamp"})
        # pandas 2.0 以降は datetime64[us] になる場合があるため ns に統一
        df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
        return df
    except Exception:
        pass

    # ── フォールバック ────────────────────────────────────────────
    if hasattr(file, "seek"):
        file.seek(0)
    df = pd.read_csv(file)

    if "Date Time" in df.columns and "Timestamp" not in df.columns:
        df = df.rename(columns={"Date Time": "Timestamp"})
    if "Timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Timestamp"})

    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S.%f",
        None,
    ):
        try:
            df["Timestamp"] = (
                pd.to_datetime(df["Timestamp"], format=fmt)
                if fmt
                else pd.to_datetime(df["Timestamp"])
            )
            break
        except Exception:
            if fmt is None:
                raise ValueError(
                    "タイムスタンプのフォーマットが認識できません。"
                    "「Date Time」または「Timestamp」列が必要です。"
                )
    # pandas 2.0 以降は datetime64[us] になる場合があるため ns に統一
    df["Timestamp"] = df["Timestamp"].astype("datetime64[ns]")
    return df


# ── 列型判定 ───────────────────────────────────────────────────────
def detect_bool_columns(df: pd.DataFrame) -> dict:
    """各列が Bool 変数か数値変数かを自動判定する"""
    result = {}
    _ts_cols = {"Timestamp", "Date Time"}
    for col in df.columns:
        if col in _ts_cols:
            continue
        series = df[col].dropna()
        if series.dtype == object:
            unique_vals = set(series.astype(str).str.upper().unique())
            is_bool = unique_vals.issubset({"TRUE", "FALSE", "0", "1"})
        else:
            unique_vals = set(series.unique())
            is_bool = unique_vals.issubset({0, 1, 0.0, 1.0})
        result[col] = "bool" if is_bool else "numeric"
    return result


def normalize_bool_series(series: pd.Series) -> pd.Series:
    """Bool 列を 0/1 に正規化する"""
    if series.dtype == object:
        return (
            series.astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    return series.fillna(0).astype(int)


# ── サイクル検出 ───────────────────────────────────────────────────
def detect_cycles(df: pd.DataFrame, trigger_col: str, edge: str) -> pd.Index:
    """基準変数のエッジ検出でサイクル開始インデックスを返す"""
    trigger = normalize_bool_series(df[trigger_col]).reset_index(drop=True)
    prev = trigger.shift(1, fill_value=0)
    mask = ((prev == 0) & (trigger == 1)) if edge == "RISE" else ((prev == 1) & (trigger == 0))
    return df.index[mask.values]


# ── 内部ヘルパー（numpy 高速版）────────────────────────────────────
_NUMERIC_OPS = {
    "==": lambda a, b: a == b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
}


def _edge_time_ns(
    col_vals: np.ndarray,   # 1-D float/int スライス
    ts_slice:  np.ndarray,  # 1-D int64 ns スライス
    edge:      str,
    start_ns:  int,
) -> "float | None":
    """numpy 配列から指定エッジの最初の発生時刻 [ms] を返す（int64 ns ベース）。
    未検出は None。
    """
    v = col_vals.astype(np.int8)
    target = np.int8(1) if edge == "RISE" else np.int8(-1)
    diff = np.empty_like(v)
    diff[0] = 0
    diff[1:] = v[1:] - v[:-1]
    pos = np.nonzero(diff == target)[0]
    if len(pos) == 0:
        return None
    return round(float(ts_slice[pos[0]] - start_ns) / 1e6, 3)


def _numeric_time_ns(
    col_vals: np.ndarray,
    ts_slice:  np.ndarray,
    op:        str,
    value:     float,
    start_ns:  int,
) -> tuple:
    """数値条件が最初に True になる時刻 [ms] と継続時間 [ms] を返す。
    未検出は (None, None)。
    """
    fn = _NUMERIC_OPS.get(op)
    if fn is None or len(col_vals) == 0:
        return None, None

    cond = fn(col_vals, value)

    # False→True 遷移を探す
    entry_arr = np.nonzero(~cond[:-1] & cond[1:])[0] + 1
    if len(entry_arr) == 0:
        if cond[0]:
            entry_pos = 0
        else:
            return None, None
    else:
        entry_pos = int(entry_arr[0])

    entry_ms = round(float(ts_slice[entry_pos] - start_ns) / 1e6, 3)

    # True→False 遷移（継続時間）
    sub = cond[entry_pos:]
    exit_arr = np.nonzero(sub & ~np.concatenate([sub[1:], [False]]))[0]
    if len(exit_arr) > 0:
        exit_pos = entry_pos + int(exit_arr[0])
        dur_ms = round(float(ts_slice[exit_pos] - ts_slice[entry_pos]) / 1e6, 3)
    else:
        dur_ms = round(float(ts_slice[-1] - ts_slice[entry_pos]) / 1e6, 3)

    return entry_ms, dur_ms


# ── 後方互換ラッパー（app.py から直接呼ばれる場合に対応）────────────
def find_edge_time(
    cycle_df: pd.DataFrame, var: str, edge: str, start_time
) -> "float | None":
    """指定変数・エッジの最初の発生時刻 [ms] を返す（後方互換）。"""
    if var not in cycle_df.columns:
        return None
    ts_ns = cycle_df["Timestamp"].values.astype(np.int64)
    start_ns = int(pd.Timestamp(start_time).value)
    return _edge_time_ns(cycle_df[var].values, ts_ns, edge, start_ns)


def find_numeric_condition_time(
    cycle_df: pd.DataFrame, var: str, op: str, value: float, start_time
) -> tuple:
    """数値条件が最初に True になる時刻 [ms] と継続時間 [ms]（後方互換）。"""
    if var not in cycle_df.columns:
        return None, None
    ts_ns = cycle_df["Timestamp"].values.astype(np.int64)
    start_ns = int(pd.Timestamp(start_time).value)
    return _numeric_time_ns(cycle_df[var].values, ts_ns, op, value, start_ns)


# ── メイン解析（最適化版）─────────────────────────────────────────
def calculate_delays_v2(
    df: pd.DataFrame, cycle_starts: pd.Index, steps: list
) -> pd.DataFrame:
    """新ステップ形式（single/range/numeric）に対応した遅れ時間計算。

    最適化:
      - Timestamp を int64 ns 配列に事前変換（Timestamp 演算を排除）
      - 全データ列を float64 行列 (N×C) に事前変換
      - サイクル毎の処理は numpy スライス参照のみ（DataFrame.loc を排除）
    """
    # ── 事前計算（全サイクル共通）────────────────────────────────
    ts_ns    = df["Timestamp"].values.astype(np.int64)
    non_ts   = [c for c in df.columns if c != "Timestamp"]
    data_arr = df[non_ts].to_numpy(dtype=np.float64)   # shape (N, C)
    col_idx  = {c: i for i, c in enumerate(non_ts)}

    cycle_arr = np.asarray(cycle_starts, dtype=np.intp)
    n_cycles  = len(cycle_arr)
    N         = len(ts_ns)

    records = []
    for i in range(n_cycles):
        s        = int(cycle_arr[i])
        e        = int(cycle_arr[i + 1]) if i + 1 < n_cycles else N
        start_ns = int(ts_ns[s])

        ts_sl  = ts_ns[s:e]          # 1-D int64 スライス（コピーなし）
        da_sl  = data_arr[s:e]       # 2-D float64 スライス（コピーなし）

        cycle_len_ms = float(ts_ns[e - 1] - start_ns) / 1e6

        record: dict = {
            "サイクル#":    i + 1,
            "開始時刻":     pd.Timestamp(start_ns, unit="ns"),
            "サイクル長[ms]": round(cycle_len_ms, 3),
        }

        for step in steps:
            name = step.get("name", "")
            mode = step.get("mode", "single")

            if mode == "single":
                ci   = col_idx.get(step.get("variable", ""))
                edge = step.get("edge", "RISE")
                t = (
                    _edge_time_ns(da_sl[:, ci], ts_sl, edge, start_ns)
                    if ci is not None else None
                )
                record[f"{name}_遅れ[ms]"] = t

            elif mode in ("range", "on_period"):
                if mode == "range":
                    svar, se = step.get("start_var", ""), step.get("start_edge", "RISE")
                    evar, ee = step.get("end_var",   ""), step.get("end_edge",   "FALL")
                else:
                    svar = evar = step.get("variable", "")
                    se, ee = "RISE", "FALL"

                si = col_idx.get(svar)
                ei = col_idx.get(evar)
                t_s = _edge_time_ns(da_sl[:, si], ts_sl, se, start_ns) if si is not None else None
                t_e = _edge_time_ns(da_sl[:, ei], ts_sl, ee, start_ns) if ei is not None else None
                record[f"{name}_start[ms]"] = t_s
                record[f"{name}_end[ms]"]   = t_e
                record[f"{name}_dur[ms]"]   = (
                    round(t_e - t_s, 3) if t_s is not None and t_e is not None else None
                )

            else:  # numeric
                ci    = col_idx.get(step.get("variable", ""))
                op    = step.get("op", "==")
                value = float(step.get("value", 0))
                if ci is None:
                    record[f"{name}_start[ms]"] = None
                    record[f"{name}_dur[ms]"]   = None
                else:
                    t_entry, t_dur = _numeric_time_ns(
                        da_sl[:, ci], ts_sl, op, value, start_ns
                    )
                    record[f"{name}_start[ms]"] = t_entry
                    record[f"{name}_dur[ms]"]   = t_dur

        records.append(record)

    return pd.DataFrame(records)


def analyze_cycles_v2(
    df: pd.DataFrame, trigger_col: str, edge: str, steps: list
) -> pd.DataFrame:
    """新バージョン：single/range 混在ステップ対応のメイン解析関数"""
    cycle_starts = detect_cycles(df, trigger_col, edge)
    if len(cycle_starts) == 0:
        return pd.DataFrame()
    return calculate_delays_v2(df, cycle_starts, steps)


# ── 波形データ取得（最適化版）─────────────────────────────────────
def get_cycle_waveforms(
    df: pd.DataFrame, cycle_starts: pd.Index, target_cols: list
) -> list:
    """各サイクルの波形データ（時間ゼロ基準）をリストで返す"""
    ts_ns     = df["Timestamp"].values.astype(np.int64)   # 事前変換
    cycle_arr = np.asarray(cycle_starts, dtype=np.intp)
    n         = len(cycle_arr)
    N         = len(ts_ns)

    waveforms = []
    for i in range(n):
        s = int(cycle_arr[i])
        e = int(cycle_arr[i + 1]) - 1 if i + 1 < n else N - 1
        slc = df.iloc[s : e + 1][["Timestamp"] + target_cols].copy()
        slc["time_offset_ms"] = (ts_ns[s : e + 1] - ts_ns[s]) / 1e6
        waveforms.append(slc)
    return waveforms


# ── 旧 API 互換（calculate_delays）────────────────────────────────
def calculate_delays(
    df: pd.DataFrame, cycle_starts: pd.Index, target_cols: list
) -> pd.DataFrame:
    """旧 API 互換ラッパー（single モードのみ）"""
    steps = [
        {"name": col, "mode": "single", "variable": col, "edge": "RISE"}
        for col in target_cols
    ]
    result = calculate_delays_v2(df, cycle_starts, steps)
    # 旧カラム名に合わせてリネーム
    rename = {f"{col}_遅れ[ms]": f"{col}_遅れ[ms]" for col in target_cols}
    return result


def analyze_cycles(
    df: pd.DataFrame, trigger_col: str, edge: str, target_cols: list
) -> pd.DataFrame:
    """旧 API 互換：メイン解析関数"""
    cycle_starts = detect_cycles(df, trigger_col, edge)
    if len(cycle_starts) == 0:
        return pd.DataFrame()
    return calculate_delays(df, cycle_starts, target_cols)


# ── ユーティリティ ─────────────────────────────────────────────────
def calc_sturges_bins(n: int) -> int:
    """スタージェスの公式でビン数を算出"""
    if n <= 1:
        return 5
    return max(5, int(1 + 3.32 * log10(n)))


def calc_statistics(delays) -> dict:
    """基本統計量を計算"""
    arr = np.array([d for d in delays if d is not None and not np.isnan(float(d))])
    if len(arr) == 0:
        return {}
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    return {
        "サンプル数":   int(len(arr)),
        "最小[ms]":    round(float(np.min(arr)), 3),
        "最大[ms]":    round(float(np.max(arr)), 3),
        "平均[ms]":    round(mean, 3),
        "標準偏差[ms]": round(std, 3),
        "3σ上限[ms]":  round(mean + 3 * std, 3),
    }


def calc_variable_periods(df: pd.DataFrame, bool_cols: list) -> dict:
    """各 Bool 変数の自然周期 [ms] を計算（RISE 間の平均インターバル）。"""
    periods = {}
    for col in bool_cols:
        try:
            series = normalize_bool_series(df[col])
            vals   = series.values
            prev   = np.concatenate([[vals[0]], vals[:-1]])
            rises  = np.where((prev == 0) & (vals == 1))[0]
            if len(rises) >= 2:
                times    = df["Timestamp"].iloc[rises]
                diffs_ms = times.diff().dropna().dt.total_seconds() * 1000
                periods[col] = float(diffs_ms.mean())
            else:
                periods[col] = None
        except Exception:
            periods[col] = None
    return periods
