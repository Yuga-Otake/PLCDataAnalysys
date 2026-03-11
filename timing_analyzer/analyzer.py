"""
analyzer.py - サイクル検出・遅れ時間計算ロジック
"""
import pandas as pd
import numpy as np
from math import log10


def load_csv(file) -> pd.DataFrame:
    """CSVを読み込み、タイムスタンプを変換して返す。

    対応フォーマット:
      - APB Variableロガー形式: 列名 "Date Time"、フォーマット "2026-02-06 13:26:00.088848"
      - 旧形式:                 列名 "Timestamp"、フォーマット "2024/03/01 08:00:00.000"
    いずれも内部では "Timestamp" 列として統一して扱う。
    """
    df = pd.read_csv(file)

    # 列名正規化: "Date Time" → "Timestamp"
    if "Date Time" in df.columns and "Timestamp" not in df.columns:
        df = df.rename(columns={"Date Time": "Timestamp"})

    if "Timestamp" not in df.columns:
        # 最初の列がタイムスタンプであると推定
        df = df.rename(columns={df.columns[0]: "Timestamp"})

    # フォーマット順に試行
    for fmt in ("%Y-%m-%d %H:%M:%S.%f",   # APB形式  2026-02-06 13:26:00.088848
                "%Y/%m/%d %H:%M:%S.%f",   # 旧形式   2024/03/01 08:00:00.000
                None):                     # pandas 自動推定
        try:
            if fmt:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=fmt)
            else:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            break
        except Exception:
            if fmt is None:
                raise ValueError(
                    "タイムスタンプのフォーマットが認識できません。"
                    "「Date Time」または「Timestamp」列が必要です。"
                )
    return df


def detect_bool_columns(df: pd.DataFrame) -> dict:
    """各列がBool変数か数値変数かを自動判定する"""
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
    """Bool列を0/1に正規化する"""
    if series.dtype == object:
        return (
            series.astype(str)
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )
    return series.fillna(0).astype(int)


def detect_cycles(df: pd.DataFrame, trigger_col: str, edge: str) -> pd.Index:
    """基準変数のエッジ検出でサイクル開始インデックスを返す"""
    trigger = normalize_bool_series(df[trigger_col]).reset_index(drop=True)
    prev = trigger.shift(1, fill_value=0)

    if edge == "RISE":
        mask = (prev == 0) & (trigger == 1)
    else:  # FALL
        mask = (prev == 1) & (trigger == 0)

    # 元のDataFrameのインデックスに対応させる
    positional = df.index[mask.values]
    return positional


def calculate_delays(
    df: pd.DataFrame, cycle_starts: pd.Index, target_cols: list
) -> pd.DataFrame:
    """各サイクルの遅れ時間[ms]をDataFrameで返す"""
    records = []

    for i in range(len(cycle_starts)):
        start_idx = cycle_starts[i]
        end_idx = cycle_starts[i + 1] if i + 1 < len(cycle_starts) else df.index[-1]

        cycle_df = df.loc[start_idx:end_idx]
        start_time = df.loc[start_idx, "Timestamp"]

        if i + 1 < len(cycle_starts):
            end_time = df.loc[end_idx, "Timestamp"]
        else:
            end_time = df["Timestamp"].iloc[-1]

        cycle_len_ms = (end_time - start_time).total_seconds() * 1000

        record = {
            "サイクル#": i + 1,
            "開始時刻": start_time,
            "サイクル長[ms]": round(cycle_len_ms, 3),
        }

        for col in target_cols:
            col_series = normalize_bool_series(cycle_df[col])
            on_rows = cycle_df[col_series.values == 1]
            if len(on_rows) > 0:
                delay_ms = (
                    on_rows.iloc[0]["Timestamp"] - start_time
                ).total_seconds() * 1000
                record[f"{col}_遅れ[ms]"] = round(delay_ms, 3)
            else:
                record[f"{col}_遅れ[ms]"] = None

        records.append(record)

    return pd.DataFrame(records)


def analyze_cycles(
    df: pd.DataFrame, trigger_col: str, edge: str, target_cols: list
) -> pd.DataFrame:
    """メイン解析関数：サイクル検出＋遅れ時間計算"""
    cycle_starts = detect_cycles(df, trigger_col, edge)
    if len(cycle_starts) == 0:
        return pd.DataFrame()
    return calculate_delays(df, cycle_starts, target_cols)


def get_cycle_waveforms(
    df: pd.DataFrame, cycle_starts: pd.Index, target_cols: list
) -> list:
    """各サイクルの波形データ（時間ゼロ基準）をリストで返す"""
    waveforms = []
    for i in range(len(cycle_starts)):
        start_idx = cycle_starts[i]
        if i + 1 < len(cycle_starts):
            end_idx = cycle_starts[i + 1]
            cycle_df = df.loc[start_idx : end_idx - 1].copy()
        else:
            cycle_df = df.loc[start_idx:].copy()

        start_time = df.loc[start_idx, "Timestamp"]
        cycle_df = cycle_df.copy()
        cycle_df["time_offset_ms"] = (
            cycle_df["Timestamp"] - start_time
        ).dt.total_seconds() * 1000
        waveforms.append(cycle_df)
    return waveforms


def calc_sturges_bins(n: int) -> int:
    """スタージェスの公式でビン数を算出"""
    if n <= 1:
        return 5
    return max(5, int(1 + 3.32 * log10(n)))


def find_edge_time(cycle_df: pd.DataFrame, var: str, edge: str, start_time) -> float:
    """指定変数・エッジの最初の発生時刻[ms]を返す（サイクル開始基準）。未検出はNone。"""
    if var not in cycle_df.columns:
        return None
    vals = normalize_bool_series(cycle_df[var]).values
    if len(vals) == 0:
        return None
    prev_vals = np.concatenate([[vals[0]], vals[:-1]])
    if edge == "RISE":
        mask = (prev_vals == 0) & (vals == 1)
    else:
        mask = (prev_vals == 1) & (vals == 0)
    positions = np.where(mask)[0]
    if len(positions) > 0:
        edge_idx = cycle_df.index[positions[0]]
        t = cycle_df.loc[edge_idx, "Timestamp"]
        return round((t - start_time).total_seconds() * 1000, 3)
    return None


def calc_variable_periods(df: pd.DataFrame, bool_cols: list) -> dict:
    """各Bool変数の自然周期[ms]を計算（RISE間の平均インターバル）。検出不能はNone。"""
    periods = {}
    for col in bool_cols:
        try:
            series = normalize_bool_series(df[col])
            vals = series.values
            prev_vals = np.concatenate([[vals[0]], vals[:-1]])
            rise_pos = np.where((prev_vals == 0) & (vals == 1))[0]
            if len(rise_pos) >= 2:
                times = df["Timestamp"].iloc[rise_pos]
                diffs_ms = times.diff().dropna().dt.total_seconds() * 1000
                periods[col] = float(diffs_ms.mean())
            else:
                periods[col] = None
        except Exception:
            periods[col] = None
    return periods


def calculate_delays_v2(
    df: pd.DataFrame, cycle_starts: pd.Index, steps: list
) -> pd.DataFrame:
    """新ステップ形式（single/range）に対応した遅れ時間計算。

    steps の各要素:
      単一変数モード: {"name":str, "mode":"single", "variable":str, "edge":"RISE"|"FALL"}
      範囲モード:     {"name":str, "mode":"range",
                       "start_var":str, "start_edge":"RISE"|"FALL",
                       "end_var":str,   "end_edge":"RISE"|"FALL"}
    """
    records = []
    for i in range(len(cycle_starts)):
        start_idx = cycle_starts[i]
        is_last   = (i + 1 >= len(cycle_starts))
        end_idx   = df.index[-1] if is_last else cycle_starts[i + 1]

        # .loc は両端を含むため、次サイクルの先頭行（= end_idx）を除外する
        # これをしないと次サイクルのトリガー立ち上がりを現サイクルで拾ってしまう
        cycle_df   = df.loc[start_idx:end_idx].iloc[:-1] if not is_last else df.loc[start_idx:]
        start_time = df.loc[start_idx, "Timestamp"]

        end_time = (df.loc[end_idx, "Timestamp"]
                    if not is_last else df["Timestamp"].iloc[-1])
        cycle_len_ms = (end_time - start_time).total_seconds() * 1000

        record = {
            "サイクル#": i + 1,
            "開始時刻": start_time,
            "サイクル長[ms]": round(cycle_len_ms, 3),
        }

        for step in steps:
            name = step.get("name", "")
            mode = step.get("mode", "single")

            if mode == "single":
                var  = step.get("variable", "")
                edge = step.get("edge", "RISE")
                t = find_edge_time(cycle_df, var, edge, start_time)
                record[f"{name}_遅れ[ms]"] = t

            else:  # range
                sv = step.get("start_var", "")
                se = step.get("start_edge", "RISE")
                ev = step.get("end_var", "")
                ee = step.get("end_edge", "FALL")
                t_s = find_edge_time(cycle_df, sv, se, start_time)
                t_e = find_edge_time(cycle_df, ev, ee, start_time)
                record[f"{name}_start[ms]"] = t_s
                record[f"{name}_end[ms]"]   = t_e
                if t_s is not None and t_e is not None:
                    record[f"{name}_dur[ms]"] = round(t_e - t_s, 3)
                else:
                    record[f"{name}_dur[ms]"] = None

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


def calc_statistics(delays) -> dict:
    """基本統計量を計算"""
    arr = np.array([d for d in delays if d is not None and not np.isnan(float(d))])
    if len(arr) == 0:
        return {}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "サンプル数": int(len(arr)),
        "最小[ms]": round(float(np.min(arr)), 3),
        "最大[ms]": round(float(np.max(arr)), 3),
        "平均[ms]": round(mean, 3),
        "標準偏差[ms]": round(std, 3),
        "3σ上限[ms]": round(mean + 3 * std, 3),
    }
