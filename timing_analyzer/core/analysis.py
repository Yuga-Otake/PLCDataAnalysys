"""
core/analysis.py — 工程・ステップ解析の共通ロジック（新画面用）

サイクル検出・ステップ解析（analyzer.analyze_cycles_v2）を呼び、
ステップ統計・ステップ候補・基準値をまとめて計算する。描画はガントのみ持つ。
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import analyze_cycles_v2, detect_cycles, normalize_bool_series

STEP_COLORS = [
    "#4472C4", "#ED7D31", "#70AD47", "#FF0000",
    "#7030A0", "#00B0F0", "#FFD966", "#FF6B6B",
    "#C00000", "#375623", "#833C00", "#1F3864",
]

MODE_LABELS = {
    "single":    "時刻",
    "on_period": "ON期間",
    "range":     "開始〜終了",
    "numeric":   "数値条件",
}


def default_color(idx: int) -> str:
    return STEP_COLORS[idx % len(STEP_COLORS)]


def describe_step(step: dict) -> str:
    """ステップの計測内容を 1 行で説明する"""
    mode = step.get("mode", "single")
    if mode == "single":
        edge = "立ち上がり" if step.get("edge", "RISE") == "RISE" else "立ち下がり"
        return f"{step.get('variable', '')} の{edge}"
    if mode == "on_period":
        return f"{step.get('variable', '')} が ON の間"
    if mode == "range":
        se = "↑" if step.get("start_edge", "RISE") == "RISE" else "↓"
        ee = "↑" if step.get("end_edge", "FALL") == "RISE" else "↓"
        return f"{step.get('start_var', '')}{se} 〜 {step.get('end_var', '')}{ee}"
    return f"{step.get('variable', '')} {step.get('op', '==')} {step.get('value', 0):g} の間"


# ═══════════════════════════════════════════════════════════════
# サイクル・ステップ解析
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def cycle_starts(df: pd.DataFrame, trigger_col: str, edge: str) -> list:
    try:
        return list(detect_cycles(df, trigger_col, edge))
    except Exception:
        return []


def mean_cycle_ms(df: pd.DataFrame, starts: list) -> float:
    if len(starts) < 2:
        return 0.0
    t0, t1 = df.loc[starts[0], "Timestamp"], df.loc[starts[-1], "Timestamp"]
    return (t1 - t0).total_seconds() * 1000 / (len(starts) - 1)


@st.cache_data(show_spinner=False)
def _analyze(df: pd.DataFrame, trigger_col: str, edge: str, steps_json: str) -> pd.DataFrame:
    steps = json.loads(steps_json)
    if not steps:
        return pd.DataFrame()
    return analyze_cycles_v2(df, trigger_col, edge, steps)


def analyze(df: pd.DataFrame, trigger_col: str, edge: str, steps: list) -> pd.DataFrame:
    try:
        return _analyze(df, trigger_col, edge, json.dumps(steps, ensure_ascii=False, sort_keys=True))
    except Exception:
        return pd.DataFrame()


def step_values(result_df: pd.DataFrame, step: dict) -> tuple:
    """(開始時刻の配列, 長さの配列) を返す。時刻ステップは開始=発生時刻・長さなし。"""
    name = step.get("name", "")
    if step.get("mode", "single") == "single":
        col = f"{name}_遅れ[ms]"
        v = result_df[col].dropna().to_numpy() if col in result_df.columns else np.array([])
        return v, np.array([])
    sc, dc = f"{name}_start[ms]", f"{name}_dur[ms]"
    s = result_df[sc].dropna().to_numpy() if sc in result_df.columns else np.array([])
    d = result_df[dc].dropna().to_numpy() if dc in result_df.columns else np.array([])
    return s, d


def step_stats(result_df: pd.DataFrame, steps: list) -> list:
    """ステップごとの統計。end_* はサイクル開始から終わりまでの時刻（時刻ステップは発生時刻）。"""
    out = []
    for i, step in enumerate(steps):
        starts, durs = step_values(result_df, step)
        mode = step.get("mode", "single")
        base = dict(name=step.get("name", ""), color=step.get("color", default_color(i)), mode=mode)
        if mode == "single":
            if len(starts) == 0:
                continue
            out.append({**base, "n": len(starts),
                        "start_mean": float(np.mean(starts)), "start_std": float(np.std(starts)),
                        "start_min": float(np.min(starts)), "start_max": float(np.max(starts)),
                        "dur_mean": 0.0, "dur_std": 0.0, "dur_min": 0.0, "dur_max": 0.0,
                        "end_mean": float(np.mean(starts))})
        else:
            if len(durs) == 0:
                continue
            s_mean = float(np.mean(starts)) if len(starts) else 0.0
            out.append({**base, "n": len(durs),
                        "start_mean": s_mean, "start_std": float(np.std(starts)) if len(starts) else 0.0,
                        "start_min": float(np.min(starts)) if len(starts) else 0.0,
                        "start_max": float(np.max(starts)) if len(starts) else 0.0,
                        "dur_mean": float(np.mean(durs)), "dur_std": float(np.std(durs)),
                        "dur_min": float(np.min(durs)), "dur_max": float(np.max(durs)),
                        "end_mean": s_mean + float(np.mean(durs))})
    return out


def gantt_figure(stats: list, takt_ms: int = 0, height_per_row: int = 34):
    """ステップのタイミング図。時刻ステップは点（ひげ＝最小〜最大）、期間ステップは棒。"""
    fig = go.Figure()
    names = [s["name"] for s in stats]
    for s in stats:
        if s["mode"] == "single":
            fig.add_trace(go.Scatter(
                x=[s["start_mean"]], y=[s["name"]], mode="markers",
                marker=dict(size=11, color=s["color"], symbol="diamond"),
                error_x=dict(type="data", symmetric=False, color=s["color"], thickness=1.5, width=5,
                             array=[s["start_max"] - s["start_mean"]],
                             arrayminus=[s["start_mean"] - s["start_min"]]),
                hovertemplate=(f"<b>{s['name']}</b><br>発生 平均 {s['start_mean']:.1f} ms"
                               f"（{s['start_min']:.1f}〜{s['start_max']:.1f}）<extra></extra>"),
                showlegend=False,
            ))
        else:
            fig.add_trace(go.Bar(
                y=[s["name"]], x=[s["dur_max"] - s["dur_min"]], base=[s["start_mean"] + s["dur_min"]],
                orientation="h", marker_color=s["color"], opacity=0.22, width=0.6,
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Bar(
                y=[s["name"]], x=[s["dur_mean"]], base=[s["start_mean"]],
                orientation="h", marker_color=s["color"], width=0.6, showlegend=False,
                hovertemplate=(f"<b>{s['name']}</b><br>開始 平均 {s['start_mean']:.1f} ms<br>"
                               f"長さ 平均 {s['dur_mean']:.1f} ms（{s['dur_min']:.1f}〜{s['dur_max']:.1f}）"
                               "<extra></extra>"),
            ))
    if takt_ms > 0:
        fig.add_vline(x=takt_ms, line_dash="dash", line_color="#cf222e",
                      annotation_text=f"タクト目標 {takt_ms} ms", annotation_position="top")
    fig.update_layout(
        barmode="overlay", height=max(220, len(stats) * height_per_row + 80),
        margin=dict(l=0, r=16, t=28, b=40),
        xaxis=dict(title="サイクル開始からの時間 [ms]", rangemode="tozero"),
        yaxis=dict(categoryorder="array", categoryarray=names, autorange="reversed"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# ステップ候補の自動検出
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def step_candidates(df: pd.DataFrame, bool_cols: tuple, trigger_col: str, edge: str) -> list:
    """全サイクルを横断し、各 Bool 変数の最初の状態変化を中央値で返す。

    返り値: [{"var", "kind" ("RISE"|"ON"|"FALL"), "t_ms", "cycles"}]（RISE→ON→FALL の順、各々時刻順）
    kind=ON はサイクル開始時点で既に ON の変数で、t_ms は ON の継続時間。
    """
    starts = cycle_starts(df, trigger_col, edge)
    rise, on, fall = {}, {}, {}
    for ci, csi in enumerate(starts):
        cei = starts[ci + 1] if ci + 1 < len(starts) else df.index[-1]
        cdf = df.loc[csi:cei]
        if len(cdf) == 0:
            continue
        t0 = int(pd.Timestamp(df.loc[csi, "Timestamp"]).value)
        ts = cdf["Timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        for var in bool_cols:
            if var == trigger_col or var not in cdf.columns:
                continue
            try:
                bv = normalize_bool_series(cdf[var]).to_numpy().astype(np.int8)
            except Exception:
                continue
            diff = np.zeros_like(bv)
            diff[1:] = bv[1:] - bv[:-1]
            if bv[0] == 1:
                fp = np.nonzero(diff == -1)[0]
                on.setdefault(var, []).append(((ts[fp[0]] if len(fp) else ts[-1]) - t0) / 1e6)
            else:
                rp = np.nonzero(diff == 1)[0]
                if len(rp):
                    rise.setdefault(var, []).append((ts[rp[0]] - t0) / 1e6)
                else:
                    fp = np.nonzero(diff == -1)[0]
                    if len(fp):
                        fall.setdefault(var, []).append((ts[fp[0]] - t0) / 1e6)

    def rows(acc, kind):
        return sorted(({"var": v, "kind": kind, "t_ms": float(np.median(t)), "cycles": len(t)}
                       for v, t in acc.items()), key=lambda r: r["t_ms"])

    return rows(rise, "RISE") + rows(on, "ON") + rows(fall, "FALL")


@st.cache_data(show_spinner=False)
def trigger_candidates(df: pd.DataFrame, bool_cols: tuple) -> list:
    """サイクル開始信号の候補（立ち上がり回数の多い順）: [(var, 回数)]"""
    out = []
    for v in bool_cols:
        try:
            b = normalize_bool_series(df[v]).to_numpy().astype(np.int8)
            # detect_cycles と同じく、先頭が ON なら 1 回目の立ち上がりとして数える
            out.append((v, int(np.sum((b[1:] - b[:-1]) == 1)) + int(len(b) > 0 and b[0] == 1)))
        except Exception:
            out.append((v, 0))
    return sorted(out, key=lambda x: -x[1])


# ═══════════════════════════════════════════════════════════════
# 基準値
# ═══════════════════════════════════════════════════════════════

def compute_baseline(dfs: list, trigger_col: str, edge: str, steps: list, method: str = "mean",
                     exclude: list | None = None):
    """複数データからステップごとの基準値を算出する。

    method: "mean"（平均）/ "min"（最小）
    exclude: dfs と同じ並びで、算出から外すサイクル番号（サイクル#）の集合のリスト
    返り値: (baseline dict, サイクル数)。baseline は設定ファイルと同じ形式
      時刻ステップ:  {"mode", "ref_ms", "std_ms"}
      期間ステップ:  {"mode", "ref_start_ms", "ref_dur_ms", "std_start_ms", "std_dur_ms"}
    """
    agg = np.mean if method == "mean" else np.min
    acc = {s["name"]: ([], []) for s in steps}
    n_cyc = 0
    for i, df in enumerate(dfs):
        rd = analyze(df, trigger_col, edge, steps)
        if rd is None or len(rd) == 0:
            continue
        if exclude and exclude[i]:
            rd = rd[~rd["サイクル#"].isin(exclude[i])]
        n_cyc += len(rd)
        for s in steps:
            st_, du = step_values(rd, s)
            acc[s["name"]][0].extend(st_.tolist())
            acc[s["name"]][1].extend(du.tolist())
    baseline = {}
    for s in steps:
        starts, durs = (np.array(x) for x in acc[s["name"]])
        mode = s.get("mode", "single")
        if mode == "single":
            if len(starts) == 0:
                continue
            baseline[s["name"]] = {"mode": mode, "ref_ms": round(float(agg(starts)), 3),
                                   "std_ms": round(float(np.std(starts)), 3)}
        else:
            if len(durs) == 0:
                continue
            baseline[s["name"]] = {
                "mode": mode,
                "ref_start_ms": round(float(agg(starts)), 3) if len(starts) else 0.0,
                "ref_dur_ms": round(float(agg(durs)), 3),
                "std_start_ms": round(float(np.std(starts)), 3) if len(starts) else 0.0,
                "std_dur_ms": round(float(np.std(durs)), 3),
            }
    return baseline, n_cyc


def baseline_meta(source: str, n_cycles: int) -> dict:
    return {"source": source, "n_cycles": n_cycles,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
