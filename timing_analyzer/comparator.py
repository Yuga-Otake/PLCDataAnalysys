"""
comparator.py - 正常 vs 異常 比較解析ロジック
"""
import numpy as np
import pandas as pd
from analyzer import analyze_cycles


def compare_normal_abnormal(
    normal_df: pd.DataFrame,
    abnormal_df: pd.DataFrame,
    trigger_col: str,
    edge: str,
    target_cols: list,
) -> tuple:
    """正常/異常それぞれの解析結果DataFrameを返す"""
    normal_result = analyze_cycles(normal_df, trigger_col, edge, target_cols)
    abnormal_result = analyze_cycles(abnormal_df, trigger_col, edge, target_cols)
    return normal_result, abnormal_result


def calc_diff_ranking(
    normal_result: pd.DataFrame,
    abnormal_result: pd.DataFrame,
    target_cols: list,
) -> pd.DataFrame:
    """変数ごとの遅れ差分（異常平均 - 正常平均）ランキングを返す"""
    diff_data = []
    for col in target_cols:
        delay_col = f"{col}_遅れ[ms]"
        if delay_col not in normal_result.columns or delay_col not in abnormal_result.columns:
            continue

        normal_delays = normal_result[delay_col].dropna().values
        abnormal_delays = abnormal_result[delay_col].dropna().values

        if len(normal_delays) == 0 or len(abnormal_delays) == 0:
            continue

        normal_mean = float(np.mean(normal_delays))
        normal_std = float(np.std(normal_delays))
        abnormal_mean = float(np.mean(abnormal_delays))
        delta = abnormal_mean - normal_mean
        z_score = delta / normal_std if normal_std > 0 else 0.0

        diff_data.append(
            {
                "変数名": col,
                "遅れ差分[ms]": round(delta, 3),
                "正規化スコア(Zスコア)": round(z_score, 3),
                "正常 平均[ms]": round(normal_mean, 3),
                "異常 平均[ms]": round(abnormal_mean, 3),
                "正常 標準偏差[ms]": round(normal_std, 3),
            }
        )

    df_diff = pd.DataFrame(diff_data)
    if len(df_diff) > 0:
        df_diff = df_diff.sort_values("遅れ差分[ms]", ascending=False).reset_index(
            drop=True
        )
    return df_diff


def detect_anomalous_variables(
    normal_result: pd.DataFrame,
    abnormal_result: pd.DataFrame,
    target_cols: list,
    exceed_threshold: float = 0.3,
) -> list:
    """正常の3σを超える異常サイクルが30%以上の変数を異常候補として返す"""
    anomalies = []
    for col in target_cols:
        delay_col = f"{col}_遅れ[ms]"
        if delay_col not in normal_result.columns or delay_col not in abnormal_result.columns:
            continue

        normal_delays = normal_result[delay_col].dropna().values
        abnormal_delays = abnormal_result[delay_col].dropna().values

        if len(normal_delays) == 0 or len(abnormal_delays) == 0:
            continue

        normal_mean = float(np.mean(normal_delays))
        normal_std = float(np.std(normal_delays))
        threshold_3sigma = normal_mean + 3 * normal_std

        exceed_rate = float(np.mean(np.array(abnormal_delays) > threshold_3sigma))

        if exceed_rate > exceed_threshold:
            anomalies.append(
                {
                    "variable": col,
                    "exceed_rate": exceed_rate,
                    "normal_mean": normal_mean,
                    "normal_std": normal_std,
                    "threshold_3sigma": threshold_3sigma,
                }
            )

    return sorted(anomalies, key=lambda x: x["exceed_rate"], reverse=True)
