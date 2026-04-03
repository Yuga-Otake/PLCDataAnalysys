"""
app.py - APB タイミング解析ツール v6
・テキスト検索＋予測候補からステップ変数を追加
・単一変数モード / 開始-終了範囲モード
・ガントクリックでステップ詳細連動
・工程ごと異常比較インライン
"""
import os, json, hashlib, urllib.parse
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Xbar-R / Xbar-S 管理図定数テーブル ─────────────────────────────
# n: (A2, D3, D4, c4, B3, B4)
#   A2, D3, D4 → Xbar-R 用
#   c4, B3, B4 → Xbar-S 用（n>10 推奨）
_SPC_CONSTS = {
    2:  (1.880, 0,     3.267, 0.7979, 0,     3.267),
    3:  (1.023, 0,     2.574, 0.8862, 0,     2.568),
    4:  (0.729, 0,     2.282, 0.9213, 0,     2.266),
    5:  (0.577, 0,     2.114, 0.9400, 0,     2.089),
    6:  (0.483, 0,     2.004, 0.9515, 0.030, 1.970),
    7:  (0.419, 0.076, 1.924, 0.9594, 0.118, 1.882),
    8:  (0.373, 0.136, 1.864, 0.9650, 0.185, 1.815),
    9:  (0.337, 0.184, 1.816, 0.9693, 0.239, 1.761),
    10: (0.308, 0.223, 1.777, 0.9727, 0.284, 1.716),
    11: (0.285, 0.256, 1.744, 0.9754, 0.321, 1.679),
    12: (0.266, 0.284, 1.717, 0.9776, 0.354, 1.646),
    13: (0.249, 0.308, 1.692, 0.9794, 0.382, 1.618),
    14: (0.235, 0.329, 1.671, 0.9810, 0.406, 1.594),
    15: (0.223, 0.347, 1.653, 0.9823, 0.428, 1.572),
    16: (0.212, 0.363, 1.637, 0.9835, 0.448, 1.552),
    17: (0.203, 0.378, 1.622, 0.9845, 0.466, 1.534),
    18: (0.194, 0.391, 1.608, 0.9854, 0.482, 1.518),
    19: (0.187, 0.403, 1.597, 0.9862, 0.497, 1.503),
    20: (0.180, 0.415, 1.585, 0.9869, 0.510, 1.490),
    21: (0.173, 0.425, 1.575, 0.9876, 0.523, 1.477),
    22: (0.167, 0.434, 1.566, 0.9882, 0.534, 1.466),
    23: (0.162, 0.443, 1.557, 0.9887, 0.545, 1.455),
    24: (0.157, 0.452, 1.548, 0.9892, 0.555, 1.445),
    25: (0.153, 0.459, 1.541, 0.9896, 0.565, 1.435),
}

def _spc_consts(n: int):
    """サブグループサイズ n に最も近い SPC 定数 (A2,D3,D4,c4,B3,B4) を返す"""
    n = max(2, int(round(n)))
    if n in _SPC_CONSTS:
        return _SPC_CONSTS[n]
    keys = sorted(_SPC_CONSTS.keys())
    return _SPC_CONSTS[min(keys, key=lambda k: abs(k - n))]

from analyzer import (
    load_csv, detect_bool_columns, detect_cycles,
    analyze_cycles_v2, calc_variable_periods,
    get_cycle_waveforms, calc_sturges_bins, calc_statistics, normalize_bool_series,
)
from comparator import compare_normal_abnormal, calc_diff_ranking, detect_anomalous_variables

st.set_page_config(page_title="APB タイミング解析", page_icon="🏭", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ステップカラーパレット
STEP_COLORS = [
    "#4472C4", "#ED7D31", "#70AD47", "#FF0000",
    "#7030A0", "#00B0F0", "#FFD966", "#FF6B6B",
    "#C00000", "#375623", "#833C00", "#1F3864",
]


# ═══════════════════════════════════════════════════════════════
# ヘルパー関数
# ═══════════════════════════════════════════════════════════════

def pk(pname: str, suffix: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in pname)
    return f"P_{safe}__{suffix}"


def _default_color(idx: int) -> str:
    return STEP_COLORS[idx % len(STEP_COLORS)]


def init_proc_widgets(pname: str, pinfo: dict, bool_cols: list):
    def _set(key, val):
        if key not in st.session_state:
            st.session_state[key] = val

    trigger = pinfo.get("trigger_col", bool_cols[0] if bool_cols else "")
    if trigger not in bool_cols and bool_cols:
        trigger = bool_cols[0]
    _set(pk(pname, "trigger"), trigger)
    _set(pk(pname, "edge"),    pinfo.get("edge", "RISE"))
    _set(pk(pname, "takt"),    int(pinfo.get("takt_target_ms", 0)))

    # steps_list 初期化（旧形式 steps → 新形式に変換）
    if pk(pname, "steps_list") not in st.session_state:
        steps_list = []
        for i, s in enumerate(pinfo.get("steps", [])):
            v = s["variable"]
            steps_list.append({
                "name":     s.get("label", v),
                "color":    s.get("color", _default_color(i)),
                "mode":     "single",
                "variable": v,
                "edge":     "RISE",
            })
        st.session_state[pk(pname, "steps_list")] = steps_list


def rename_process(old_name: str, new_name: str):
    """工程名を変更し、関連 session_state キーをすべて移行する"""
    procs = st.session_state["processes"]
    new_procs = {(new_name if k == old_name else k): v for k, v in procs.items()}
    st.session_state["processes"] = new_procs

    old_safe = "".join(c if c.isalnum() else "_" for c in old_name)
    new_safe = "".join(c if c.isalnum() else "_" for c in new_name)
    old_pfx, new_pfx = f"P_{old_safe}__", f"P_{new_safe}__"
    for key in [k for k in list(st.session_state.keys()) if k.startswith(old_pfx)]:
        st.session_state[new_pfx + key[len(old_pfx):]] = st.session_state.pop(key)

    if st.session_state.get("_expand_new") == old_name:
        st.session_state["_expand_new"] = new_name
    ng_old, ng_new = f"_ng_df_{old_name}", f"_ng_df_{new_name}"
    if ng_old in st.session_state:
        st.session_state[ng_new] = st.session_state.pop(ng_old)


def mean_waveform(waves: list, var: str):
    if not waves:
        return [], []
    mt  = max(c["time_offset_ms"].max() for c in waves)
    ta  = np.linspace(0, mt, 300)
    mv  = [np.mean([normalize_bool_series(c[var]).values[
                        np.searchsorted(c["time_offset_ms"].values, t)]
                    for c in waves
                    if np.searchsorted(c["time_offset_ms"].values, t) < len(c)])
           for t in ta]
    return ta, mv


def steps_all_vars(steps_list: list, bool_cols: list) -> list:
    """steps_list から使用している変数名リストを返す（bool_cols の順序で）"""
    used = set()
    for s in steps_list:
        if s.get("mode", "single") == "single":
            if s.get("variable"):
                used.add(s["variable"])
        else:
            if s.get("start_var"): used.add(s["start_var"])
            if s.get("end_var"):   used.add(s["end_var"])
    return [v for v in bool_cols if v in used]


# ═══════════════════════════════════════════════════════════════
# キャッシュ付き解析
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def cached_load_sample(path: str):
    df = load_csv(path)
    return df, detect_bool_columns(df)

@st.cache_data
def cached_detect_cycles(df: pd.DataFrame, trigger_col: str, edge: str) -> list:
    return list(detect_cycles(df, trigger_col, edge))

@st.cache_data
def cached_analyze_v2(df: pd.DataFrame, trigger_col: str, edge: str,
                       steps_json: str) -> pd.DataFrame:
    steps = json.loads(steps_json)
    if not steps:
        return pd.DataFrame()
    return analyze_cycles_v2(df, trigger_col, edge, steps)

@st.cache_data
def cached_variable_periods(df: pd.DataFrame, bool_cols_t: tuple) -> dict:
    return calc_variable_periods(df, list(bool_cols_t))

@st.cache_data
def cached_waveforms(df: pd.DataFrame, trigger_col: str, edge: str,
                     target_vars: tuple) -> list:
    idx = pd.Index(cached_detect_cycles(df, trigger_col, edge))
    return get_cycle_waveforms(df, idx, list(target_vars))

@st.cache_data
def cached_build_gantt(result_df: pd.DataFrame, steps_json: str, takt_target: int):
    """build_gantt_v2 のキャッシュラッパー（result_df + steps_json が同一なら再計算しない）"""
    import json as _json
    steps_list = _json.loads(steps_json)
    return build_gantt_v2(result_df, steps_list, takt_target)


@st.cache_data
def make_mini_chart(df: pd.DataFrame, var: str, height: int = 65) -> go.Figure:
    sample = df.head(2000)
    try:
        vals    = normalize_bool_series(sample[var])
        y_range = [-0.1, 1.3]
        color   = "#4472C4"
    except Exception:
        vals    = sample[var]
        y_range = None
        color   = "#ED7D31"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(vals))), y=vals, fill="tozeroy",
                             mode="lines", line=dict(width=1, color=color)))
    lo = dict(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
              plot_bgcolor="#f5f7fa", paper_bgcolor="rgba(0,0,0,0)",
              xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    if y_range:
        lo["yaxis"]["range"] = y_range
    fig.update_layout(**lo)
    return fig


# ═══════════════════════════════════════════════════════════════
# ガントチャート（v2: single/range 混在対応）
# ═══════════════════════════════════════════════════════════════

def calc_nice_bins(data: np.ndarray, key: str = "") -> int:
    """Freedman-Diaconis ルールで bin 数を計算（見た目がきれいな刻み幅）"""
    # まずセッション上書きを確認
    override = st.session_state.get(f"_bins_{key}") if key else None
    if override and override > 0:
        return int(override)
    n = len(data)
    if n <= 4:
        return 5
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    data_range = float(np.max(data) - np.min(data))
    if iqr <= 0 or data_range <= 0:
        return max(5, min(30, int(1 + 3.32 * np.log10(max(n, 2)))))
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    # nice な幅に丸める
    mag = 10.0 ** np.floor(np.log10(bin_width))
    for f in [1, 2, 5, 10]:
        nw = f * mag
        if nw >= bin_width * 0.8:
            bin_width = nw
            break
    return max(5, min(50, int(np.ceil(data_range / bin_width))))


def build_gantt_v2(result_df: pd.DataFrame, steps_list: list, takt_target: int):
    step_stats, prev_mean = [], 0.0

    for step in steps_list:
        name  = step.get("name", "")
        color = step.get("color", "#4472C4")
        mode  = step.get("mode", "single")

        if mode == "single":
            dcol = f"{name}_遅れ[ms]"
            if dcol not in result_df.columns:
                continue
            delays = result_df[dcol].dropna().values
            if len(delays) == 0:
                continue
            mean_d = float(np.mean(delays))
            std_d  = float(np.std(delays))
            min_d, max_d = float(np.min(delays)), float(np.max(delays))
            step_stats.append(dict(
                name=name, color=color, mode="single",
                start=0.0, mean=mean_d,
                min=min_d, max=max_d,
                abs_mean=mean_d, abs_std=std_d, abs_min=min_d, abs_max=max_d,
            ))

        elif mode in ("range", "numeric", "on_period"):
            sc = f"{name}_start[ms]"
            dc = f"{name}_dur[ms]"
            if sc not in result_df.columns or dc not in result_df.columns:
                continue
            starts = result_df[sc].dropna().values
            durs   = result_df[dc].dropna().values
            if len(durs) == 0:
                continue
            mean_s = float(np.mean(starts)) if len(starts) > 0 else prev_mean
            mean_d = float(np.mean(durs))
            std_d  = float(np.std(durs))
            min_d, max_d = float(np.min(durs)), float(np.max(durs))
            step_stats.append(dict(
                name=name, color=color, mode=mode,
                start=mean_s, mean=mean_d,
                min=min_d, max=max_d,
                abs_mean=mean_s + mean_d, abs_std=std_d,
                abs_min=mean_s + min_d, abs_max=mean_s + max_d,
                abs_start=mean_s,
            ))
            prev_mean = mean_s + mean_d

    if not step_stats:
        return None, []

    total = max((s["abs_mean"] for s in step_stats), default=1.0) or 1.0
    fig   = go.Figure()

    for s in step_stats:
        pct      = round(s["abs_mean"] / total * 100, 1)
        smode    = s["mode"]  # "single" / "range" / "numeric" / "on_period"
        is_range = smode in ("range", "numeric", "on_period")
        pat      = {"range": "/", "numeric": "x", "on_period": "\\", "single": ""}.get(smode, "")
        if smode == "range":
            ht = (
                "<b>%{y}</b> [範囲]<br>"
                f"開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "長さ（平均）: %{customdata[1]:.1f} ms<br>"
                "長さばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        elif smode == "on_period":
            ht = (
                "<b>%{y}</b> [ON期間]<br>"
                f"ON開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "ON継続（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        elif smode == "numeric":
            ht = (
                "<b>%{y}</b> [数値条件]<br>"
                f"条件成立開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "継続時間（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        else:
            ht = (
                "<b>%{y}</b><br>"
                "サイクル開始から（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<extra></extra>"
            )

        fig.add_trace(go.Bar(
            name=s["name"], y=[s["name"]], x=[s["mean"]], base=[s["start"]],
            orientation="h", width=0.5, marker_color=s["color"],
            marker_pattern_shape=pat,
            customdata=[[s["start"], s["mean"], s["min"], s["max"], pct]],
            hovertemplate=ht,
            text=[f"{s['mean']:.0f}ms"] if not is_range else [f"{pct:.0f}%"],
            textposition="outside",
            textfont=dict(size=11, color=s["color"]),
            constraintext="none",
        ))
        bw = s["max"] - s["min"]
        if bw > 0:
            fig.add_trace(go.Bar(
                y=[s["name"]], x=[bw], base=[s["start"] + s["min"]],
                orientation="h", width=0.5, marker_color=s["color"],
                opacity=0.2, showlegend=False, hoverinfo="skip",
            ))

    if takt_target > 0:
        fig.add_vline(x=takt_target, line_dash="dash", line_color="red",
                      annotation_text=f"目標 {takt_target}ms")
    fig.update_layout(
        barmode="overlay",
        xaxis_title="サイクル開始からの経過時間 [ms]",
        yaxis=dict(autorange="reversed"),
        height=max(240, len(step_stats) * 60 + 100),
        margin=dict(l=0, r=50, t=12, b=36),
        showlegend=False, plot_bgcolor="white", dragmode="select",
    )
    return fig, step_stats


# ═══════════════════════════════════════════════════════════════
# B-1: トレンドチャート（サイクルごとの推移）
# ═══════════════════════════════════════════════════════════════

def build_trend_chart(result_df: pd.DataFrame, step_stats: list, takt_target: int):
    """サイクルごとの遅れ時間トレンドチャートを生成する"""
    if result_df is None or len(result_df) == 0 or not step_stats:
        return None
    cycle_col = "サイクル#"
    if cycle_col not in result_df.columns:
        return None

    x = result_df[cycle_col].values
    fig = go.Figure()

    for s in step_stats:
        name  = s["name"]
        color = s["color"]
        mode  = s.get("mode", "single")
        dcol  = f"{name}_遅れ[ms]" if mode == "single" else f"{name}_dur[ms]"
        if dcol not in result_df.columns:
            continue

        y = result_df[dcol].values.astype(float)
        mean_y = float(np.nanmean(y))
        std_y  = float(np.nanstd(y))

        # 散布点
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color=color, opacity=0.55),
            name=name, legendgroup=name,
        ))

        # 5サイクル移動平均
        ma = pd.Series(y).rolling(5, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=x, y=ma, mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} 移動平均", legendgroup=name, showlegend=False,
        ))

        # 外れ値マーカー（|val - mean| > 2σ）
        if std_y > 0:
            outlier_mask = np.abs(y - mean_y) > 2 * std_y
            if np.any(outlier_mask):
                fig.add_trace(go.Scatter(
                    x=x[outlier_mask], y=y[outlier_mask], mode="markers",
                    marker=dict(size=11, color="red", symbol="circle-open",
                                line=dict(width=2, color="red")),
                    name=f"{name} 外れ値", legendgroup=name, showlegend=False,
                ))

            # mean+3σ ライン
            fig.add_hline(
                y=mean_y + 3 * std_y,
                line_dash="dot", line_color=color, line_width=1,
                annotation_text=f"{name} 3σ上限",
                annotation_font_size=9,
            )

    if takt_target > 0:
        fig.add_hline(y=takt_target, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"タクト目標 {takt_target}ms")

    fig.update_layout(
        xaxis_title="サイクル番号",
        yaxis_title="時間 [ms]",
        height=400,
        margin=dict(t=16, b=36),
        showlegend=True,
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# B-2: IQR外れ値検出
# ═══════════════════════════════════════════════════════════════

def detect_outliers_iqr(result_df: pd.DataFrame, step_stats: list) -> list:
    """IQR法で各ステップの外れ値サイクルを検出する"""
    if result_df is None or len(result_df) == 0:
        return []
    cycle_col = "サイクル#"
    results = []
    for s in step_stats:
        name = s["name"]
        mode = s.get("mode", "single")
        dcol = f"{name}_遅れ[ms]" if mode == "single" else f"{name}_dur[ms]"
        if dcol not in result_df.columns:
            continue
        col = result_df[dcol].dropna()
        if len(col) < 4:
            continue
        Q1  = col.quantile(0.25)
        Q3  = col.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lo = Q1 - 1.5 * IQR
        hi = Q3 + 1.5 * IQR
        mask = (col < lo) | (col > hi)
        outlier_rows = result_df.loc[col[mask].index]
        if cycle_col in outlier_rows.columns:
            cycles = [int(c) for c in outlier_rows[cycle_col].tolist()]
        else:
            cycles = [int(i) for i in outlier_rows.index.tolist()]
        if cycles:
            results.append({
                "name":   name,
                "color":  s["color"],
                "cycles": cycles,
                "hi":     round(hi, 1),
                "lo":     round(lo, 1),
            })
    return results


# ═══════════════════════════════════════════════════════════════
# ステップ詳細
# ═══════════════════════════════════════════════════════════════

def render_step_detail(df: pd.DataFrame, trigger_col: str, edge: str,
                       step_stat: dict, step: dict, pname: str,
                       result_df: pd.DataFrame):
    mode = step.get("mode", "single")
    name = step_stat["name"]

    if mode == "single":
        _render_single_detail(df, trigger_col, edge, step_stat, step, pname, result_df)
    elif mode == "numeric":
        _render_numeric_detail(df, trigger_col, edge, step_stat, step, pname, result_df)
    else:  # range / on_period
        _render_range_detail(df, trigger_col, edge, step_stat, step, pname, result_df)


def _render_single_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    var       = step.get("variable", "")
    name      = step_stat["name"]
    delay_col = f"{name}_遅れ[ms]"
    delays    = (result_df[delay_col].dropna().values
                 if delay_col in result_df.columns else np.array([]))

    if len(delays) == 0:
        st.warning(f"{var}: データが取得できませんでした")
        return

    mean_d, std_d = float(np.mean(delays)), float(np.std(delays))
    sig3 = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}"

    # 基準値チェック
    _baseline    = st.session_state.get(pk(pname, "baseline"), {})
    _bl_entry    = _baseline.get(name, {})
    _bl_ref      = _bl_entry.get("ref_ms")
    _bl_std      = _bl_entry.get("std_ms", 0.0)
    _delta_mode  = _bl_ref is not None
    delays_plot  = (delays - _bl_ref) if _delta_mode else delays

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        if _delta_mode:
            st.caption(f"基準値: **{_bl_ref:.1f} ms**")
        else:
            st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        if _delta_mode:
            st.markdown("**差分ヒストグラム（基準値=0）**")
        else:
            st.markdown("**ヒストグラム**")
        _bkey  = f"{pname}_{name}_h"
        n_bins = calc_nice_bins(delays_plot, _bkey)
        fig_h  = go.Figure()
        if threshold > 0 and not _delta_mode:
            below = [d for d in delays_plot if d <= threshold]
            above = [d for d in delays_plot if d > threshold]
            if below:
                fig_h.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内",
                                             marker_color="royalblue", opacity=0.75))
            if above:
                fig_h.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"閾値 {threshold}ms")
        elif _delta_mode and _bl_std > 0:
            _t3 = 3 * _bl_std
            in_r  = delays_plot[np.abs(delays_plot) <= _t3]
            out_r = delays_plot[np.abs(delays_plot) > _t3]
            if len(in_r) > 0:
                fig_h.add_trace(go.Histogram(x=in_r, nbinsx=n_bins, name="±3σ以内",
                                             marker_color="royalblue", opacity=0.75))
            if len(out_r) > 0:
                fig_h.add_trace(go.Histogram(x=out_r, nbinsx=n_bins, name="±3σ超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=_t3,  line_dash="dash", line_color="orange",
                            annotation_text=f"+3σ({_t3:.1f}ms)")
            fig_h.add_vline(x=-_t3, line_dash="dash", line_color="orange",
                            annotation_text=f"-3σ({_t3:.1f}ms)")
        else:
            fig_h.add_trace(go.Histogram(x=delays_plot, nbinsx=n_bins,
                                         marker_color="steelblue", opacity=0.8))
        if _delta_mode:
            fig_h.add_vline(x=0, line_color="green", annotation_text=f"基準 {_bl_ref:.1f}ms")
            xaxis_title = "基準値からのずれ [ms]"
        else:
            fig_h.add_vline(x=sig3, line_dash="dot", line_color="gray",
                            annotation_text=f"3σ {sig3:.1f}ms")
            xaxis_title = "遅れ時間 [ms]"
        fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                             barmode="overlay", height=260, margin=dict(t=8, b=32),
                             showlegend=True)
        st.plotly_chart(fig_h, use_container_width=True, key=f"hist_{pname}_{name}")
        sc = st.columns(4)
        mean_plot = float(np.mean(delays_plot))
        std_plot  = float(np.std(delays_plot))
        sc[0].metric("N",      len(delays_plot))
        sc[1].metric("平均" if not _delta_mode else "差分平均", f"{mean_plot:.1f}ms")
        sc[2].metric("σ",     f"{std_plot:.1f}ms")
        sc[3].metric("3σ" if not _delta_mode else "基準σ", f"{(_bl_std if _delta_mode else sig3):.1f}ms")
        if threshold > 0 and not _delta_mode:
            rate = np.mean(delays <= threshold) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")
        st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                  help="ヒストグラムのビン数を手動調整")

    with w_col:
        st.markdown("**波形重ね（全サイクル＋平均）**")
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, (var,))
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        event_t    = step_stat.get("abs_mean", 30.0)
        half_win   = max(event_t * 0.3, 20.0)
        view_start = max(0.0, event_t - half_win)
        view_end   = event_t + half_win

        _MAX_WAVE = 60   # 表示サイクル上限（多いと描画が重くなるため）
        fig_w   = go.Figure()
        all_t, all_v = [], []
        _shown = 0
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = normalize_bool_series(sl[var]).values
            all_t.append(t); all_v.append(v)
            if _shown < _MAX_WAVE:
                fig_w.add_trace(go.Scatter(x=t, y=v, mode="lines",
                                           line=dict(color="rgba(100,100,200,0.10)", width=1),
                                           showlegend=False))
                _shown += 1
        if all_t:
            tmin = min(t[0]  for t in all_t if len(t) > 0)
            tmax = max(t[-1] for t in all_t if len(t) > 0)
            ct   = np.linspace(tmin, tmax, 300)
            mv   = [np.mean([v[np.searchsorted(t, tp)]
                             for t, v in zip(all_t, all_v)
                             if np.searchsorted(t, tp) < len(v)])
                    for tp in ct]
            fig_w.add_trace(go.Scatter(x=ct, y=mv, mode="lines",
                                       line=dict(color="royalblue", width=3), name="平均波形"))
        abs_mean = step_stat.get("abs_mean", mean_d)
        fig_w.add_vline(x=abs_mean, line_dash="dash", line_color="green",
                        annotation_text=f"平均 {abs_mean:.1f}ms")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title="変数値", height=260, margin=dict(t=8, b=32))
        st.plotly_chart(fig_w, use_container_width=True, key=f"wave_{pname}_{name}")
        st.caption(f"{len(all_t)} サイクル重ね　{view_start:.1f}〜{view_end:.1f} ms")


def _render_range_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    name = step_stat["name"]
    mode = step.get("mode", "range")
    if mode == "on_period":
        start_var = step.get("variable", "")
        end_var   = step.get("variable", "")
    else:
        start_var = step.get("start_var", "")
        end_var   = step.get("end_var", "")
    start_col = f"{name}_start[ms]"
    dur_col   = f"{name}_dur[ms]"

    starts = (result_df[start_col].dropna().values
              if start_col in result_df.columns else np.array([]))
    durs   = (result_df[dur_col].dropna().values
              if dur_col in result_df.columns else np.array([]))

    if len(durs) == 0:
        st.warning(f"データが取得できませんでした（変数: {start_var}）")
        return

    mean_s, mean_d = (float(np.mean(starts)) if len(starts) > 0 else 0), float(np.mean(durs))
    std_d = float(np.std(durs))
    sig3  = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}_dur"

    # 基準値チェック
    _baseline   = st.session_state.get(pk(pname, "baseline"), {})
    _bl_entry   = _baseline.get(name, {})
    _bl_ref_dur = _bl_entry.get("ref_dur_ms")
    _bl_std_dur = _bl_entry.get("std_dur_ms", 0.0)
    _delta_mode = _bl_ref_dur is not None
    durs_plot   = (durs - _bl_ref_dur) if _delta_mode else durs

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "所要時間閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="所要時間閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        if _delta_mode:
            st.caption(f"基準値: **{_bl_ref_dur:.1f} ms**")
        else:
            st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        if _delta_mode:
            st.markdown("**差分ヒストグラム（基準値=0）**")
        else:
            st.markdown("**所要時間ヒストグラム**")
        _bkey  = f"{pname}_{name}_r"
        n_bins = calc_nice_bins(durs_plot, _bkey)
        fig_h  = go.Figure()
        if threshold > 0 and not _delta_mode:
            below = [d for d in durs_plot if d <= threshold]
            above = [d for d in durs_plot if d > threshold]
            if below:
                fig_h.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内",
                                             marker_color="royalblue", opacity=0.75))
            if above:
                fig_h.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"閾値 {threshold}ms")
        elif _delta_mode and _bl_std_dur > 0:
            _t3 = 3 * _bl_std_dur
            in_r  = durs_plot[np.abs(durs_plot) <= _t3]
            out_r = durs_plot[np.abs(durs_plot) > _t3]
            if len(in_r) > 0:
                fig_h.add_trace(go.Histogram(x=in_r, nbinsx=n_bins, name="±3σ以内",
                                             marker_color="teal", opacity=0.75))
            if len(out_r) > 0:
                fig_h.add_trace(go.Histogram(x=out_r, nbinsx=n_bins, name="±3σ超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=_t3,  line_dash="dash", line_color="orange",
                            annotation_text=f"+3σ({_t3:.1f}ms)")
            fig_h.add_vline(x=-_t3, line_dash="dash", line_color="orange",
                            annotation_text=f"-3σ({_t3:.1f}ms)")
        else:
            fig_h.add_trace(go.Histogram(x=durs_plot, nbinsx=n_bins,
                                         marker_color="teal", opacity=0.8))
        if _delta_mode:
            fig_h.add_vline(x=0, line_color="green", annotation_text=f"基準 {_bl_ref_dur:.1f}ms")
            xaxis_title = "基準値からのずれ [ms]"
        else:
            fig_h.add_vline(x=sig3, line_dash="dot", line_color="gray",
                            annotation_text=f"3σ {sig3:.1f}ms")
            xaxis_title = "所要時間 [ms]"
        fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                             barmode="overlay", height=260, margin=dict(t=8, b=32),
                             showlegend=True)
        st.plotly_chart(fig_h, use_container_width=True, key=f"hist_{pname}_{name}")
        sc = st.columns(4)
        mean_plot = float(np.mean(durs_plot))
        sc[0].metric("N",        len(durs_plot))
        sc[1].metric("平均所要" if not _delta_mode else "差分平均", f"{mean_plot:.1f}ms")
        sc[2].metric("σ",       f"{std_d:.1f}ms")
        sc[3].metric("3σ上限" if not _delta_mode else "基準σ",
                     f"{(_bl_std_dur if _delta_mode else sig3):.1f}ms")
        if not _delta_mode:
            st.caption(f"開始タイミング平均: **{mean_s:.1f} ms**")
        if threshold > 0 and not _delta_mode:
            rate = np.mean(durs <= threshold) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")
        st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                  help="ヒストグラムのビン数を手動調整")

    with w_col:
        st.markdown("**開始/終了波形重ね**")
        vars_needed = tuple(set(v for v in [start_var, end_var] if v))
        if not vars_needed:
            st.warning("変数が設定されていません")
            return
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, vars_needed)
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        view_start = max(0.0, mean_s - mean_d * 0.5)
        view_end   = mean_s + mean_d * 2.5

        _MAX_WAVE = 60   # 表示サイクル上限
        fig_w  = go.Figure()
        _shown = 0
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            if _shown >= _MAX_WAVE:
                continue
            t = sl["time_offset_ms"].values
            if start_var and start_var in sl.columns:
                fig_w.add_trace(go.Scatter(x=t, y=normalize_bool_series(sl[start_var]).values,
                                           mode="lines", showlegend=False,
                                           line=dict(color="rgba(65,105,225,0.10)", width=1)))
            if end_var and end_var != start_var and end_var in sl.columns:
                fig_w.add_trace(go.Scatter(x=t, y=normalize_bool_series(sl[end_var]).values,
                                           mode="lines", showlegend=False,
                                           line=dict(color="rgba(220,100,60,0.10)", width=1)))
            _shown += 1

        # 平均波形
        if start_var:
            ta, mv = mean_waveform(waveforms, start_var)
            if mv:
                fig_w.add_trace(go.Scatter(x=ta, y=mv, mode="lines",
                                           line=dict(color="royalblue", width=2.5),
                                           name=f"開始変数 {start_var}"))
        if end_var and end_var != start_var:
            ta2, mv2 = mean_waveform(waveforms, end_var)
            if mv2:
                fig_w.add_trace(go.Scatter(x=ta2, y=mv2, mode="lines",
                                           line=dict(color="tomato", width=2.5),
                                           name=f"終了変数 {end_var}"))

        fig_w.add_vline(x=mean_s, line_dash="dash", line_color="green",
                        annotation_text=f"開始 {mean_s:.1f}ms")
        fig_w.add_vline(x=mean_s + mean_d, line_dash="dash", line_color="red",
                        annotation_text=f"終了 {mean_s+mean_d:.1f}ms")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title="変数値", height=260,
                            margin=dict(t=8, b=32), showlegend=True,
                            legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_w, use_container_width=True, key=f"wave_{pname}_{name}")
        st.caption(f"開始変数: {start_var} {step.get('start_edge','RISE')}  ／  "
                   f"終了変数: {end_var} {step.get('end_edge','FALL')}")


# ═══════════════════════════════════════════════════════════════
# ダイアログ: 工程追加
# ═══════════════════════════════════════════════════════════════

@st.dialog("工程を追加")
def add_process_dialog(bool_cols: list):
    pname   = st.text_input("工程名", placeholder="例: 組み付け工程", key="_dlg_pname")
    trigger = st.selectbox("基準変数（サイクル開始信号）", bool_cols, key="_dlg_trigger")
    edge    = st.radio(
        "トリガーエッジ", ["RISE", "FALL"], horizontal=True,
        format_func=lambda x: "↑ 立ち上がり（OFF→ON）" if x == "RISE" else "↓ 立ち下がり（ON→OFF）",
        key="_dlg_edge",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("登録する", type="primary", disabled=not pname.strip()):
            name = pname.strip()
            if name in st.session_state["processes"]:
                st.error(f"「{name}」は既に登録されています")
            else:
                st.session_state["processes"][name] = {
                    "trigger_col": trigger, "edge": edge,
                    "takt_target_ms": 0, "steps": [],
                }
                st.session_state["_expand_new"] = name
                st.rerun()
    with c2:
        if st.button("キャンセル"):
            st.rerun()


def _render_numeric_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    """数値条件モードの詳細表示（継続時間ヒストグラム＋変数波形重ね）"""
    name      = step_stat["name"]
    var       = step.get("variable", "")
    op        = step.get("op", "==")
    value     = step.get("value", 0)
    start_col = f"{name}_start[ms]"
    dur_col   = f"{name}_dur[ms]"

    starts = (result_df[start_col].dropna().values
              if start_col in result_df.columns else np.array([]))
    durs   = (result_df[dur_col].dropna().values
              if dur_col in result_df.columns else np.array([]))

    if len(durs) == 0:
        st.warning(f"条件「{var} {op} {value}」が成立するサイクルが見つかりませんでした")
        return

    mean_s, mean_d = (float(np.mean(starts)) if len(starts) > 0 else 0), float(np.mean(durs))
    std_d = float(np.std(durs))
    sig3  = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}_dur"

    st.caption(f"条件: **{var} {op} {value}**　　"
               f"成立タイミング平均: **{mean_s:.1f} ms**")

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "継続時間閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="継続時間閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        st.markdown("**継続時間ヒストグラム**")
        _bkey_n = f"{pname}_{name}_n"
        n_bins  = calc_nice_bins(durs, _bkey_n)   # Freedman-Diaconis で自動算出
        fig_h   = go.Figure()
        if threshold > 0:
            below = [d for d in durs if d <= threshold]
            above = [d for d in durs if d > threshold]
            if below:
                fig_h.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内",
                                             marker_color="royalblue", opacity=0.75))
            if above:
                fig_h.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"閾値 {threshold}ms")
        else:
            fig_h.add_trace(go.Histogram(x=durs, nbinsx=n_bins,
                                         marker_color="teal", opacity=0.8))
        fig_h.add_vline(x=sig3, line_dash="dot", line_color="gray",
                        annotation_text=f"3σ {sig3:.1f}ms")
        fig_h.update_layout(xaxis_title="継続時間[ms]", yaxis_title="頻度",
                             barmode="overlay", height=260, margin=dict(t=8, b=32),
                             showlegend=threshold > 0)
        st.plotly_chart(fig_h, use_container_width=True, key=f"hist_{pname}_{name}")
        st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey_n}",
                  help="ヒストグラムのビン数を手動調整（Freedman-Diaconis による自動算出が既定値）")
        sc = st.columns(4)
        sc[0].metric("N",        len(durs))
        sc[1].metric("平均継続",  f"{mean_d:.1f}ms")
        sc[2].metric("σ",       f"{std_d:.1f}ms")
        sc[3].metric("3σ上限",   f"{sig3:.1f}ms")
        if threshold > 0:
            rate = np.mean(durs <= threshold) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")

    with w_col:
        st.markdown(f"**波形重ね（{var}）**")
        if not var or var not in df.columns:
            st.warning("変数が設定されていません")
            return
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, (var,))
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        half_win   = max(mean_d * 0.8, 30.0)
        view_start = max(0.0, mean_s - half_win * 0.5)
        view_end   = mean_s + mean_d + half_win

        _MAX_WAVE = 60   # 表示サイクル上限
        fig_w  = go.Figure()
        _shown = 0
        all_slices = []
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = sl[var].values
            all_slices.append((t, v))    # 平均計算用（全サイクル使用）
            if _shown < _MAX_WAVE:
                fig_w.add_trace(go.Scatter(x=t, y=v, mode="lines", showlegend=False,
                                           line=dict(color="rgba(100,100,200,0.12)", width=1)))
                _shown += 1
        # 平均ライン（数値なので簡易的に各時刻の平均を取る）
        if all_slices:
            ct = np.linspace(view_start, view_end, 200)
            mv = [np.mean([np.interp(t, ts, vs) for ts, vs in all_slices]) for t in ct]
            fig_w.add_trace(go.Scatter(x=ct, y=mv, mode="lines",
                                       line=dict(color="royalblue", width=2.5), name="平均"))
        fig_w.add_vline(x=mean_s, line_dash="dash", line_color="green",
                        annotation_text=f"条件成立 {mean_s:.1f}ms")
        fig_w.add_vline(x=mean_s + mean_d, line_dash="dash", line_color="red",
                        annotation_text=f"条件終了 {mean_s+mean_d:.1f}ms")
        fig_w.add_hrect(y0=value - 0.3, y1=value + 0.3,
                        fillcolor="yellow", opacity=0.15, line_width=0,
                        annotation_text=f"{op} {value}")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title=var, height=260,
                            margin=dict(t=8, b=32), showlegend=True,
                            legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_w, use_container_width=True, key=f"wave_{pname}_{name}")


# ═══════════════════════════════════════════════════════════════
# ダイアログ群（最小分割）
# ─ cycle_settings_dialog : サイクル設定のみ（⚙️）
# ─ add_step_dialog        : ステップ追加（＋）
# ─ edit_step_dialog       : ステップ編集（ステップをクリック）
# ═══════════════════════════════════════════════════════════════

# ── ① サイクル設定（最小限）──────────────────────────────────

@st.dialog("⚙️ サイクル設定", width="small")
def cycle_settings_dialog(pname: str, bool_cols: list, df: pd.DataFrame):
    st.markdown(f"**{pname}**")

    _, del_col = st.columns([4, 1])
    with del_col:
        if st.button("🗑 削除", type="secondary", use_container_width=True,
                     key=f"_cyc_del_{pname}"):
            del st.session_state["processes"][pname]
            if st.session_state.get("_expand_new") == pname:
                st.session_state["_expand_new"] = None
            st.rerun()

    _cur_t = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    if _cur_t not in bool_cols:
        _cur_t = bool_cols[0]
    trigger_col = st.selectbox(
        "基準変数", bool_cols, index=bool_cols.index(_cur_t),
        key=pk(pname, "trigger"),
    )
    _cur_e = st.session_state.get(pk(pname, "edge"), "RISE")
    if _cur_e not in ["RISE", "FALL"]:
        _cur_e = "RISE"
    st.radio(
        "エッジ", ["RISE", "FALL"], index=["RISE", "FALL"].index(_cur_e),
        format_func=lambda x: "↑ 立ち上がり" if x == "RISE" else "↓ 立ち下がり",
        key=pk(pname, "edge"), horizontal=True,
    )
    st.number_input(
        "タクト目標 [ms]", min_value=0,
        value=st.session_state.get(pk(pname, "takt"), 0),
        step=10, key=pk(pname, "takt"),
    )
    try:
        cs = cached_detect_cycles(df, trigger_col, _cur_e)
        n  = len(cs)
        c1, c2 = st.columns(2)
        c1.metric("サイクル数", n)
        if n > 1:
            t0 = df.loc[cs[0], "Timestamp"]; t1 = df.loc[cs[-1], "Timestamp"]
            c2.metric("平均周期", f"{(t1-t0).total_seconds()*1000/(n-1):.0f}ms")
    except Exception:
        pass


# ── ② ステップ追加（＋ ボタン）──────────────────────────────

@st.dialog("＋ ステップを追加", width="large")
def add_step_dialog(pname: str, bool_cols: list, df: pd.DataFrame):
    trigger_col   = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    edge          = st.session_state.get(pk(pname, "edge"), "RISE")
    steps         = list(st.session_state.get(pk(pname, "steps_list"), []))
    cand_vars     = [v for v in bool_cols if v != trigger_col]

    # 工程周期
    proc_cycle_ms = 1000.0
    try:
        cs = cached_detect_cycles(df, trigger_col, edge)
        if len(cs) > 1:
            t0 = df.loc[cs[0], "Timestamp"]; t1 = df.loc[cs[-1], "Timestamp"]
            proc_cycle_ms = (t1 - t0).total_seconds() * 1000 / (len(cs) - 1)
    except Exception:
        pass

    var_periods = {}
    try:
        var_periods = cached_variable_periods(df, tuple(bool_cols))
    except Exception:
        pass

    added_vars = set()
    for s in steps:
        if s.get("mode", "single") == "single":
            added_vars.add(s.get("variable", ""))
        else:
            added_vars.add(s.get("start_var", ""))
            added_vars.add(s.get("end_var", ""))

    def _period_key(v):
        p = var_periods.get(v)
        return abs(p - proc_cycle_ms) if p else float("inf")

    cand_sorted = sorted(cand_vars, key=_period_key)

    # 検索ボックス（キーバージョン方式でリセット）
    srch_ver = st.session_state.get(f"_srch_ver_{pname}", 0)
    if st.session_state.pop(f"_clear_srch_{pname}", False):
        srch_ver += 1
        st.session_state[f"_srch_ver_{pname}"] = srch_ver

    search_txt = st.text_input(
        "🔍 変数名でフィルタ",
        key=f"step_srch_{pname}_v{srch_ver}",
        placeholder="変数名を入力...",
    )

    cand_filtered = (
        [v for v in cand_sorted if search_txt.lower() in v.lower()]
        if search_txt else cand_sorted
    )

    if cand_filtered:
        lbl = "検索結果:" if search_txt else f"候補（周期 {proc_cycle_ms:.0f}ms に近い順）:"
        st.caption(lbl)

        def _fmt(v):
            p = var_periods.get(v)
            tag = " [追加済]" if v in added_vars else (f"  {p:.0f}ms" if p else "")
            return f"{v}{tag}"

        selected_vars = st.multiselect(
            "追加する変数を選択（複数選択可）",
            options=cand_filtered,
            default=[],
            format_func=_fmt,
            key=f"multi_add_{pname}_v{srch_ver}",
            label_visibility="collapsed",
            placeholder="クリックして変数を選択...",
        )

        add_mode_lbl = st.radio(
            "追加モード",
            ["RISE 時刻（単一変数）", "ON 期間（1 の間）"],
            horizontal=True,
            key=f"bulk_add_mode_{pname}",
        )
        add_mode = "single" if "RISE" in add_mode_lbl else "on_period"

        _new = [v for v in selected_vars if v not in added_vars]
        btn_label = f"＋ {len(_new)} 件を追加" if _new else "＋ 追加"
        if st.button(
            btn_label,
            disabled=(not _new),
            type="primary",
            key=f"bulk_add_{pname}",
            use_container_width=True,
        ):
            for v in _new:
                if add_mode == "on_period":
                    steps.append({
                        "name":     v,
                        "color":    _default_color(len(steps)),
                        "mode":     "on_period",
                        "variable": v,
                    })
                else:
                    steps.append({
                        "name":     v,
                        "color":    _default_color(len(steps)),
                        "mode":     "single",
                        "variable": v,
                        "edge":     "RISE",
                    })
            st.session_state[pk(pname, "steps_list")] = steps
            st.session_state[f"_clear_srch_{pname}"] = True
            st.toast(f"+ {len(_new)}件のステップを追加しました", icon="✅")
            st.rerun()

    if steps:
        st.divider()
        st.caption("追加済み: " + "　→　".join(s["name"] for s in steps))

    # JSON インポート
    with st.expander("📂 設定JSONを読み込む", expanded=False):
        jf = st.file_uploader("JSON", type=["json"], key=f"cfg_json_{pname}",
                               label_visibility="collapsed")
        if jf:
            try:
                loaded = json.load(jf)
                new_list = []
                for i2, s in enumerate(loaded.get("steps", [])):
                    if "mode" in s:
                        new_list.append(s)
                    else:
                        v2 = s.get("variable", "")
                        new_list.append({
                            "name": s.get("label", v2), "color": s.get("color", _default_color(i2)),
                            "mode": "single", "variable": v2, "edge": "RISE",
                        })
                st.session_state[pk(pname, "steps_list")] = new_list
                st.session_state[pk(pname, "takt")] = loaded.get("takt_target_ms", 0)
                st.rerun()
            except Exception as e:
                st.error(f"読み込みエラー: {e}")


# ── ③ ステップ編集（ステップをクリック）────────────────────

@st.dialog("ステップ編集", width="small")
def edit_step_dialog(pname: str, step_idx: int, bool_cols: list, df: pd.DataFrame):
    steps = list(st.session_state.get(pk(pname, "steps_list"), []))
    if step_idx >= len(steps):
        st.error("ステップが見つかりません"); return

    step        = steps[step_idx]
    trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    var_list    = [v for v in bool_cols if v != trigger_col]
    mode        = step.get("mode", "single")
    name        = step.get("name", "")
    color       = step.get("color", _default_color(step_idx))

    # 削除
    _, del_col = st.columns([5, 1])
    with del_col:
        if st.button("🗑", use_container_width=True, key=f"_edel_{pname}_{step_idx}"):
            steps.pop(step_idx)
            st.session_state[pk(pname, "steps_list")] = steps
            st.toast(f"🗑 {name} を削除しました")
            st.rerun()

    # モード切替
    _mode_opts  = ["単一変数", "ON期間", "開始/終了", "数値条件"]
    _mode_idx   = {"single": 0, "on_period": 1, "range": 2, "numeric": 3}.get(mode, 0)
    new_mode_lbl = st.radio(
        "モード", _mode_opts,
        index=_mode_idx, horizontal=True,
        key=f"_emode_{pname}_{step_idx}",
    )
    new_mode = {"単一変数": "single", "ON期間": "on_period", "開始/終了": "range", "数値条件": "numeric"}[new_mode_lbl]

    # 変数設定
    col_types = st.session_state.get("col_types", {})
    num_cols = [c for c, t in col_types.items() if t == "numeric"]

    if new_mode == "single":
        cur_var  = step.get("variable", step.get("start_var", var_list[0] if var_list else ""))
        cur_edge = step.get("edge", "RISE")
        v_i = var_list.index(cur_var) if cur_var in var_list else 0
        new_var = st.selectbox("変数", var_list, index=v_i,
                               key=f"_evar_{pname}_{step_idx}")
        new_edge_lbl = st.radio("エッジ", ["RISE↑", "FALL↓"],
                                index=0 if cur_edge == "RISE" else 1, horizontal=True,
                                key=f"_eedge_{pname}_{step_idx}")
        new_edge = "RISE" if "RISE" in new_edge_lbl else "FALL"
        if new_var:
            st.plotly_chart(make_mini_chart(df, new_var, 48), use_container_width=True,
                            key=f"_emini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "single", "variable": new_var, "edge": new_edge}

    elif new_mode == "on_period":
        cur_var = step.get("variable", step.get("start_var", var_list[0] if var_list else ""))
        v_i     = var_list.index(cur_var) if cur_var in var_list else 0
        new_var = st.selectbox("変数", var_list, index=v_i,
                               key=f"_eopvar_{pname}_{step_idx}")
        st.caption("選択した変数が 1 の間（RISE → FALL）を計測します")
        if new_var:
            st.plotly_chart(make_mini_chart(df, new_var, 48), use_container_width=True,
                            key=f"_eopmini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "on_period", "variable": new_var}

    elif new_mode == "range":
        cur_sv = step.get("start_var", step.get("variable", var_list[0] if var_list else ""))
        cur_se = step.get("start_edge", "RISE")
        cur_ev = step.get("end_var",   step.get("variable", var_list[0] if var_list else ""))
        cur_ee = step.get("end_edge", "FALL")
        sv_i   = var_list.index(cur_sv) if cur_sv in var_list else 0
        ev_i   = var_list.index(cur_ev) if cur_ev in var_list else 0

        c1, c2 = st.columns(2)
        with c1:
            new_sv = st.selectbox("開始変数", var_list, index=sv_i,
                                  key=f"_esv_{pname}_{step_idx}")
        with c2:
            new_se = "RISE" if "RISE" in st.radio(
                "開始エッジ", ["RISE↑", "FALL↓"],
                index=0 if cur_se == "RISE" else 1, horizontal=True,
                key=f"_ese_{pname}_{step_idx}") else "FALL"
        c3, c4 = st.columns(2)
        with c3:
            new_ev = st.selectbox("終了変数", var_list, index=ev_i,
                                  key=f"_eev_{pname}_{step_idx}")
        with c4:
            new_ee = "RISE" if "RISE" in st.radio(
                "終了エッジ", ["RISE↑", "FALL↓"],
                index=0 if cur_ee == "RISE" else 1, horizontal=True,
                key=f"_eee_{pname}_{step_idx}") else "FALL"
        upd = {"mode": "range",
               "start_var": new_sv, "start_edge": new_se,
               "end_var":   new_ev, "end_edge":   new_ee}

    else:  # numeric
        cur_nvar  = step.get("variable", num_cols[0] if num_cols else "")
        cur_op    = step.get("op", "==")
        cur_val   = float(step.get("value", 0))
        nv_i      = num_cols.index(cur_nvar) if cur_nvar in num_cols else 0
        _ops      = ["==", ">=", "<=", ">", "<"]

        cn1, cn2, cn3 = st.columns([3, 1, 2])
        with cn1:
            new_nvar = st.selectbox("数値変数", num_cols if num_cols else ["（数値列なし）"],
                                    index=nv_i, key=f"_envar_{pname}_{step_idx}")
        with cn2:
            new_op = st.selectbox("条件", _ops,
                                  index=_ops.index(cur_op) if cur_op in _ops else 0,
                                  key=f"_eop_{pname}_{step_idx}", label_visibility="hidden")
        with cn3:
            new_val = st.number_input("値", value=cur_val, step=1.0,
                                      key=f"_eval_{pname}_{step_idx}", label_visibility="hidden")
        if new_nvar and new_nvar in df.columns:
            st.caption(f"「{new_nvar} {new_op} {new_val:.0f}」が True の区間を計測します")
            st.plotly_chart(make_mini_chart(df, new_nvar, 48), use_container_width=True,
                            key=f"_enmini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "numeric", "variable": new_nvar, "op": new_op, "value": new_val}

    # 名前 & 色
    nc1, nc2 = st.columns([4, 1])
    with nc1:
        new_name  = st.text_input("ステップ名", value=name,
                                   key=f"_ename_{pname}_{step_idx}")
    with nc2:
        new_color = st.color_picker("色", value=color,
                                     key=f"_ecolor_{pname}_{step_idx}")

    # 並べ替え
    r1, r2, _ = st.columns([1, 1, 3])
    with r1:
        if st.button("↑ 上へ", disabled=step_idx == 0,
                     key=f"_eup_{pname}_{step_idx}", use_container_width=True):
            steps[step_idx], steps[step_idx-1] = steps[step_idx-1], steps[step_idx]
            st.session_state[pk(pname, "steps_list")] = steps
            st.rerun()
    with r2:
        if st.button("↓ 下へ", disabled=step_idx == len(steps)-1,
                     key=f"_edn_{pname}_{step_idx}", use_container_width=True):
            steps[step_idx], steps[step_idx+1] = steps[step_idx+1], steps[step_idx]
            st.session_state[pk(pname, "steps_list")] = steps
            st.rerun()

    # 変更を自動保存
    upd["name"]  = new_name or name
    upd["color"] = new_color
    updated = {**step, **upd}
    if updated != step:
        steps[step_idx] = updated
        st.session_state[pk(pname, "steps_list")] = steps

    # JSON エクスポート（折りたたみ）
    with st.expander("💾 設定JSONを保存", expanded=False):
        all_steps = st.session_state.get(pk(pname, "steps_list"), [])
        cfg = {
            "process_name": pname,
            "trigger_col":  st.session_state.get(pk(pname, "trigger"), ""),
            "edge":         st.session_state.get(pk(pname, "edge"), "RISE"),
            "takt_target_ms": st.session_state.get(pk(pname, "takt"), 0),
            "steps":        all_steps,
        }
        st.download_button(
            "ダウンロード",
            data=json.dumps(cfg, ensure_ascii=False, indent=2),
            file_name=f"{pname}_config.json", mime="application/json",
            key=f"_edl_{pname}_{step_idx}", use_container_width=True,
        )




# ── ④ 基準値登録・編集 ──────────────────────────────────────────

@st.dialog("基準値登録・編集", width="small")
def baseline_dialog(pname: str, step_stats: list, result_df: pd.DataFrame):
    """現在の解析結果から基準値を作成・手動編集する"""
    existing = st.session_state.get(pk(pname, "baseline"), {})

    src_lbl = st.radio(
        "算出方法", ["平均値", "最小値", "特定サイクル", "手動入力"],
        horizontal=True, key=f"_bl_src_{pname}",
    )

    # 特定サイクル選択
    _sel_cyc_row = None
    if src_lbl == "特定サイクル" and "サイクル#" in result_df.columns:
        _cyc_nums = result_df["サイクル#"].dropna().astype(int).tolist()
        _sel_cyc = st.selectbox(
            "サイクルを選択", _cyc_nums,
            format_func=lambda x: f"サイクル {x}",
            key=f"_bl_cyc_sel_{pname}",
        )
        _cyc_mask = result_df["サイクル#"].astype(int) == _sel_cyc
        _sel_cyc_row = result_df[_cyc_mask].iloc[0] if _cyc_mask.any() else None

    new_vals = {}
    st.divider()

    for s in step_stats:
        name = s["name"]
        mode = s["mode"]
        ex   = existing.get(name, {})

        if mode == "single":
            dcol = f"{name}_遅れ[ms]"
            vals = result_df[dcol].dropna().values if dcol in result_df.columns else np.array([])
            if len(vals) == 0:
                continue
            if src_lbl == "平均値":
                default = float(np.mean(vals))
            elif src_lbl == "最小値":
                default = float(np.min(vals))
            elif src_lbl == "特定サイクル":
                default = (float(_sel_cyc_row[dcol])
                           if _sel_cyc_row is not None and dcol in _sel_cyc_row and pd.notna(_sel_cyc_row[dcol])
                           else float(np.mean(vals)))
            else:
                default = float(ex.get("ref_ms", np.mean(vals)))

            st.caption(f"**{name}** — 遅れ時間基準")
            ref_ms = st.number_input(
                "基準値 [ms]", value=default, step=0.5,
                key=f"_bl_{pname}_{name}_rm",
                disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                label_visibility="collapsed",
            )
            new_vals[name] = {
                "mode":   mode,
                "ref_ms": ref_ms,
                "std_ms": float(np.std(vals)),
            }

        else:  # range / on_period / numeric
            dcol  = f"{name}_dur[ms]"
            scol  = f"{name}_start[ms]"
            dvals = result_df[dcol].dropna().values if dcol in result_df.columns else np.array([])
            svals = result_df[scol].dropna().values if scol in result_df.columns else np.array([])
            if len(dvals) == 0:
                continue
            if src_lbl == "平均値":
                default_dur   = float(np.mean(dvals))
                default_start = float(np.mean(svals)) if len(svals) > 0 else 0.0
            elif src_lbl == "最小値":
                default_dur   = float(np.min(dvals))
                default_start = float(np.min(svals)) if len(svals) > 0 else 0.0
            elif src_lbl == "特定サイクル":
                default_dur = (float(_sel_cyc_row[dcol])
                               if _sel_cyc_row is not None and dcol in _sel_cyc_row and pd.notna(_sel_cyc_row[dcol])
                               else float(np.mean(dvals)))
                default_start = (float(_sel_cyc_row[scol])
                                 if _sel_cyc_row is not None and scol in _sel_cyc_row and pd.notna(_sel_cyc_row[scol])
                                 else float(np.mean(svals)) if len(svals) > 0 else 0.0)
            else:
                default_dur   = float(ex.get("ref_dur_ms",   np.mean(dvals)))
                default_start = float(ex.get("ref_start_ms", float(np.mean(svals)) if len(svals) > 0 else 0.0))

            st.caption(f"**{name}** — 開始 / 継続時間基準")
            c1, c2 = st.columns(2)
            with c1:
                ref_start = st.number_input(
                    "開始 [ms]", value=default_start, step=0.5,
                    key=f"_bl_{pname}_{name}_rs",
                    disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                )
            with c2:
                ref_dur = st.number_input(
                    "継続 [ms]", value=default_dur, step=0.5,
                    key=f"_bl_{pname}_{name}_rd",
                    disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                )
            new_vals[name] = {
                "mode":         mode,
                "ref_start_ms": ref_start,
                "ref_dur_ms":   ref_dur,
                "std_dur_ms":   float(np.std(dvals)),
                "std_start_ms": float(np.std(svals)) if len(svals) > 0 else 0.0,
            }
        st.divider()

    n_cyc = len(result_df)
    _btn_label = (
        f"✅ 登録（サイクル {st.session_state.get(f'_bl_cyc_sel_{pname}', '?')} の値）"
        if src_lbl == "特定サイクル" and _sel_cyc_row is not None
        else f"✅ 登録（{n_cyc} サイクルの{src_lbl}）"
    )
    if st.button(_btn_label, type="primary", use_container_width=True):
        st.session_state[pk(pname, "baseline")] = new_vals
        st.session_state[pk(pname, "baseline_meta")] = {
            "source":    src_lbl,
            "n_cycles":  n_cyc,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.toast("✅ 基準値を登録しました")
        st.rerun()

    if existing:
        if st.button("🗑 基準値をクリア", use_container_width=True):
            for k in [pk(pname, "baseline"), pk(pname, "baseline_meta")]:
                st.session_state.pop(k, None)
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# Session State 初期化
# ═══════════════════════════════════════════════════════════════

if "processes"   not in st.session_state:
    st.session_state["processes"]   = {}
if "_expand_new" not in st.session_state:
    st.session_state["_expand_new"] = None


# ═══════════════════════════════════════════════════════════════
# サイドバー: データ読み込み
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📂 データ")
    use_sample = st.toggle("サンプルデータを使用", value=True, key="use_sample")

    if use_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_playback.csv")
        try:
            df, col_types = cached_load_sample(sample_path)
            st.success(f"sample_playback.csv（{len(df):,}行）")
            st.session_state["df"] = df
            st.session_state["uploaded_filename"] = "sample_playback.csv"
        except FileNotFoundError:
            st.error("sample_playback.csv が見つかりません")
            st.stop()
    else:
        uploaded = st.file_uploader("CSVをアップロード", type=["csv"], key="upload_main")
        if uploaded:
            try:
                df       = load_csv(uploaded)
                col_types = detect_bool_columns(df)
                st.success(f"{len(df):,}行 読み込み完了")
                st.session_state["df"] = df
                st.session_state["uploaded_filename"] = uploaded.name
            except Exception as e:
                st.error(f"読み込みエラー: {e}")
                st.stop()
        else:
            st.info("CSVをアップロードしてください")
            st.stop()

    bool_cols = [c for c, t in col_types.items() if t == "bool"]
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]

    # A-3: データ概要カード
    with st.container():
        _n_bool = len(bool_cols)
        _n_num  = len(num_cols)
        if "Timestamp" in df.columns and len(df) > 0:
            _ts0 = str(df["Timestamp"].iloc[0])[:19]
            _ts1 = str(df["Timestamp"].iloc[-1])[:19]
            st.caption(f"📅 {_ts0}  〜  {_ts1}")
        st.caption(f"🔀 Bool列: **{_n_bool}**　　📊 アナログ列: **{_n_num}**")

    # CSV 変更検知 → 工程を自動登録
    csv_hash = hashlib.md5(df.iloc[:, 0].astype(str).str.cat().encode()).hexdigest()[:8]
    if st.session_state.get("_csv_hash") != csv_hash:
        st.session_state["_csv_hash"]   = csv_hash
        st.session_state["processes"]   = {}
        st.session_state["_expand_new"] = None
        starts = [c for c in bool_cols if any(
            kw in c.lower() for kw in ["start", "begin", "trigger", "trig"]
        )]
        if not starts:
            starts = bool_cols[:1]
        for var in starts:
            pname_auto = var
            for sfx in ["_Start", "_start", "_Begin", "_begin", "_Trigger", "_trigger"]:
                pname_auto = pname_auto.replace(sfx, "")
            st.session_state["processes"][pname_auto] = {
                "trigger_col": var, "edge": "RISE", "takt_target_ms": 0, "steps": [],
            }
        if st.session_state["processes"]:
            st.session_state["_expand_new"] = list(st.session_state["processes"].keys())[0]

    for pname, pinfo in st.session_state["processes"].items():
        init_proc_widgets(pname, pinfo, bool_cols)

    # ─── 設定ファイル 保存 / 読み込み ───────────────────────────
    st.divider()
    st.markdown("#### 💾 設定ファイル")

    # 保存（現在の全工程設定をJSONに）
    if st.session_state.get("processes"):
        _exp_procs = {}
        for _ep in st.session_state["processes"]:
            _exp_procs[_ep] = {
                "trigger":        st.session_state.get(pk(_ep, "trigger"), ""),
                "edge":           st.session_state.get(pk(_ep, "edge"), "RISE"),
                "takt_target_ms": st.session_state.get(pk(_ep, "takt"), 0),
                "steps":          st.session_state.get(pk(_ep, "steps_list"), []),
                "baseline":       st.session_state.get(pk(_ep, "baseline"), {}),
                "baseline_meta":  st.session_state.get(pk(_ep, "baseline_meta"), {}),
            }
        _exp_json = json.dumps(
            {"version": "1.0",
             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "processes": _exp_procs},
            ensure_ascii=False, indent=2,
        )
        st.download_button(
            "📥 設定を保存 (JSON)",
            data=_exp_json,
            file_name=f"apb_settings_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
            key="_settings_export_btn",
        )

    # 読み込み
    _imp_file = st.file_uploader(
        "📤 設定JSONを読み込む", type=["json"],
        key="_settings_import", label_visibility="visible",
    )
    if _imp_file is not None:
        try:
            _imp = json.load(_imp_file)
            if "processes" not in _imp:
                st.error("有効な設定JSONではありません（'processes'キーが見つかりません）")
            else:
                _new_procs: dict = {}
                for _ip, _ipd in _imp["processes"].items():
                    _new_procs[_ip] = {
                        "trigger_col":   _ipd.get("trigger", ""),
                        "edge":          _ipd.get("edge", "RISE"),
                        "takt_target_ms": _ipd.get("takt_target_ms", 0),
                        "steps":         _ipd.get("steps", []),
                    }
                    # session_state を直接セット（rerun後に init_proc_widgets が上書きしないよう先行設定）
                    st.session_state[pk(_ip, "trigger")]    = _ipd.get("trigger", "")
                    st.session_state[pk(_ip, "edge")]       = _ipd.get("edge", "RISE")
                    st.session_state[pk(_ip, "takt")]       = int(_ipd.get("takt_target_ms", 0))
                    st.session_state[pk(_ip, "steps_list")] = _ipd.get("steps", [])
                    if _ipd.get("baseline"):
                        st.session_state[pk(_ip, "baseline")] = _ipd["baseline"]
                    if _ipd.get("baseline_meta"):
                        st.session_state[pk(_ip, "baseline_meta")] = _ipd["baseline_meta"]
                st.session_state["processes"]   = _new_procs
                st.session_state["_expand_new"] = next(iter(_new_procs), None)
                st.toast("✅ 設定を読み込みました")
                st.rerun()
        except Exception as _imp_e:
            st.error(f"読み込みエラー: {_imp_e}")

    # ─── ページナビゲーション ─────────────────────────────────
    st.divider()
    st.page_link("pages/4_ロガー設定ナビ.py",   label="📋 ロガー設定ナビ")
    st.page_link("pages/0_このツールについて.py", label="ℹ️ このツールについて")


# ═══════════════════════════════════════════════════════════════
# メインコンテンツ
# ═══════════════════════════════════════════════════════════════

st.title("🏭 APB タイミング解析")

processes = st.session_state["processes"]

hc1, hc2 = st.columns([6, 1])
with hc1:
    st.caption(f"登録済み工程: {len(processes)}件")
with hc2:
    if st.button("＋ 工程を追加", type="primary", use_container_width=True):
        add_process_dialog(bool_cols)

if not processes:
    st.info("「＋ 工程を追加」から工程を登録してください。")
    st.stop()

# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# ページタブ
# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═

_page_tabs = st.tabs(["⚙️ 画面設定", "📐 新データ評価", "📈 傾向解析"])

with _page_tabs[0]:


    # ═══════════════════════════════════════════════════════════════
    # 工程タイムライン概要（アコーディオン形式）
    # ═══════════════════════════════════════════════════════════════

    _proc_items = []
    for _pn in processes:
        _sl = st.session_state.get(pk(_pn, "steps_list"), [])
        if not _sl:
            continue
        _tr = st.session_state.get(pk(_pn, "trigger"), bool_cols[0])
        _ed = st.session_state.get(pk(_pn, "edge"), "RISE")
        _tk = int(st.session_state.get(pk(_pn, "takt"), 0))
        try:
            _rj = json.dumps(_sl, ensure_ascii=False, sort_keys=True)
            _rd = cached_analyze_v2(df, _tr, _ed, _rj)
            if _rd is None or len(_rd) == 0:
                continue
            _, _ss = build_gantt_v2(_rd, _sl, 0)
            if not _ss:
                continue
            _total  = max((s["abs_mean"] for s in _ss), default=0.0)
            _cyc_ms = 0.0
            try:
                _cs = cached_detect_cycles(df, _tr, _ed)
                if len(_cs) > 1:
                    _t0 = df.loc[_cs[0],  "Timestamp"]
                    _t1 = df.loc[_cs[-1], "Timestamp"]
                    _cyc_ms = (_t1 - _t0).total_seconds() * 1000 / (len(_cs) - 1)
            except Exception:
                pass
            _proc_items.append({
                "proc": _pn, "total": _total, "takt": _tk,
                "cyc": _cyc_ms, "n_steps": len(_sl),
                "trigger": _tr, "edge": _ed,
                "steps_list": _sl, "result_df": _rd, "step_stats": _ss,
            })
        except Exception:
            continue

    if _proc_items:
        _max_total = max(p["total"] for p in _proc_items)
        _x_max = max(
            _max_total,
            max((p["cyc"]  for p in _proc_items if p["cyc"]  > 0), default=0),
            max((p["takt"] for p in _proc_items if p["takt"] > 0), default=0),
        ) * 1.06 or 1.0

        _sum_hdr, _ea, _ca = st.columns([8, 1, 1])
        with _sum_hdr:
            st.markdown("**⏱ 工程タイムライン概要**")
        with _ea:
            if st.button("▼ 全展開", key="_expand_all", use_container_width=True):
                for _p in _proc_items:
                    st.session_state[f"_sum_exp_{_p['proc']}"] = True
                st.rerun()
        with _ca:
            if st.button("▶ 全折畳", key="_collapse_all", use_container_width=True):
                for _p in _proc_items:
                    st.session_state[f"_sum_exp_{_p['proc']}"] = False
                st.rerun()
        st.caption(
            "▶ をクリックするとステップ詳細ガントが展開されます　"
            "｜　ガントバーをダブルクリックするとヒストグラムが別タブで開きます"
        )

        for _i, _item in enumerate(_proc_items):
            _clr    = STEP_COLORS[_i % len(STEP_COLORS)]
            _is_bn  = (_item["total"] == _max_total)
            _is_exp = st.session_state.get(f"_sum_exp_{_item['proc']}", False)
            _over   = _item["takt"] > 0 and _item["total"] > _item["takt"]

            # ─ 行レイアウト: [▶] [タイムラインバー全体] ────────────
            _rc0, _rc_bar = st.columns([0.55, 11.45])

            with _rc0:
                if st.button(
                    "▼" if _is_exp else "▶",
                    key=f"_sum_tog_{_item['proc']}",
                    use_container_width=True,
                ):
                    st.session_state[f"_sum_exp_{_item['proc']}"] = not _is_exp
                    st.rerun()

            with _rc_bar:
                _bp = min(99.5, _item["total"] / _x_max * 100)
                _cp = min(99.5, _item["cyc"]   / _x_max * 100) if _item["cyc"]  > 0 else 0
                _tp = min(99.5, _item["takt"]  / _x_max * 100) if _item["takt"] > 0 else -1
                _bdr = "2px solid crimson" if _over else "none"
                _bn_badge = " 🔴" if _is_bn else ""

                if _item["takt"] > 0:
                    _delta = _item["total"] - _item["takt"]
                    _dc = "crimson" if _delta > 0 else "seagreen"
                    _takt_label = (f' &nbsp;<span style="color:{_dc};font-size:11px;">'
                                   f'({_delta:+.0f}ms)</span>')
                else:
                    _takt_label = ""

                _ch = (f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_cp:.1f}%;'
                       f'background:{_clr};opacity:0.13;border-radius:4px;"></div>') if _cp > 0 else ""
                _th = (f'<div style="position:absolute;left:{_tp:.1f}%;top:0;bottom:0;'
                       f'width:2px;background:rgba(210,0,0,0.8);z-index:10;"></div>') if _tp >= 0 else ""

                # ─ プロセス行
                _html = '<div style="position:relative;font-size:13px;">'
                _html += (
                    f'<div style="position:relative;height:28px;margin-bottom:2px;">'
                    f'<div style="position:absolute;left:0;top:0;right:0;bottom:0;'
                    f'background:#f0f0f0;border-radius:4px;border:{_bdr};overflow:hidden;">'
                    f'{_ch}'
                    f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_bp:.1f}%;'
                    f'background:{_clr};border-radius:4px;opacity:0.85;"></div>'
                    f'</div>'
                    f'{_th}'
                    f'<div style="position:absolute;left:{min(99, _bp+0.5):.1f}%;top:50%;'
                    f'transform:translateY(-50%);white-space:nowrap;padding-left:6px;font-weight:bold;">'
                    f'{_item["proc"]}{_bn_badge} '
                    f'<span style="font-weight:normal;color:#666;">{_item["total"]:.0f}ms</span>'
                    f'{_takt_label}'
                    f'</div>'
                    f'</div>'
                )

                # ─ ステップ行（展開時）
                if _is_exp:
                    for _s in _item["step_stats"]:
                        _sl = _s["start"] / _x_max * 100
                        _sw = max(0.5, _s["mean"] / _x_max * 100)
                        _pct = _s["mean"] / (_item["total"] or 1) * 100
                        _sc  = _s["color"]
                        _html += (
                            f'<div style="position:relative;height:22px;margin-bottom:1px;">'
                            f'<div style="position:absolute;left:{_sl:.1f}%;width:{_sw:.1f}%;'
                            f'top:4px;height:14px;background:{_sc};border-radius:3px;opacity:0.75;"></div>'
                            f'<div style="position:absolute;left:{min(99, _sl+_sw+0.5):.1f}%;top:3px;'
                            f'white-space:nowrap;padding-left:3px;font-size:12px;color:#444;">'
                            f'{_s["name"]} '
                            f'<span style="color:#888;">{_s["mean"]:.0f}ms</span> '
                            f'<span style="color:{_sc};font-weight:600;">({_pct:.0f}%)</span>'
                            f'</div>'
                            f'</div>'
                        )

                _html += '</div>'
                st.markdown(_html, unsafe_allow_html=True)

            # ─ ボトルネックメッセージ（展開時）
            if _is_exp and _item["step_stats"]:
                _max_st = max(_item["step_stats"], key=lambda s: s["mean"])
                _pct_max = _max_st["mean"] / (_item["total"] or 1) * 100
                st.caption(
                    f"📌 最長ステップ: **{_max_st['name']}**  "
                    f"{_max_st['mean']:.1f} ms　工程比 {_pct_max:.0f}%"
                )
                if _item["takt"] > 0 and _item["total"] > _item["takt"]:
                    _need = _item["total"] - _item["takt"]
                    st.caption(
                        f"⚠️ タクト目標達成には **{_need:.0f} ms** の短縮が必要。"
                        f"まず **{_max_st['name']}** を改善してください。"
                    )
                elif _item["takt"] > 0:
                    _margin = _item["takt"] - _item["total"]
                    st.caption(f"✅ タクト目標を **{_margin:.0f} ms** 下回っています。")

            st.markdown('<hr style="margin:4px 0 8px 0;opacity:0.2;">', unsafe_allow_html=True)

        st.markdown("")


    # ═══════════════════════════════════════════════════════════════
    # 工程ごとのセクション
    # ═══════════════════════════════════════════════════════════════

    for pname, pinfo in list(processes.items()):

        trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
        if trigger_col not in bool_cols:
            trigger_col = bool_cols[0]
            st.session_state[pk(pname, "trigger")] = trigger_col

        edge        = st.session_state.get(pk(pname, "edge"), "RISE")
        takt_target = st.session_state.get(pk(pname, "takt"), 0)
        edge_s      = "↑" if edge == "RISE" else "↓"
        steps_list  = st.session_state.get(pk(pname, "steps_list"), [])

        try:
            n_cyc = len(cached_detect_cycles(df, trigger_col, edge))
        except Exception:
            n_cyc = 0

        step_lbl = f"{len(steps_list)}ステップ" if steps_list else "ステップ未設定"
        _is_new  = (pname == st.session_state.get("_expand_new"))

        with st.expander(
            f"🏭  **{pname}**　　{trigger_col} {edge_s}　|　{n_cyc}サイクル　|　{step_lbl}",
            expanded=_is_new,
        ):
            # ── ヘッダ: 工程名変更 ＋ サイクル設定 ＋ ステップ追加 ──
            name_col, cyc_col, add_col = st.columns([4, 1, 1.5])
            with name_col:
                new_name = st.text_input(
                    "工程名", value=pname,
                    key=f"rename_input_{pname}",
                    label_visibility="collapsed",
                    placeholder="工程名を変更",
                )
            with cyc_col:
                if st.button("⚙️", key=f"open_cyc_{pname}",
                             use_container_width=True, help="サイクル設定"):
                    cycle_settings_dialog(pname, bool_cols, df)
            with add_col:
                if st.button("＋ ステップを追加", key=f"open_add_{pname}",
                             use_container_width=True, type="primary"):
                    add_step_dialog(pname, bool_cols, df)

            new_name_stripped = new_name.strip()
            if (new_name_stripped and
                    new_name_stripped != pname and
                    new_name_stripped not in processes):
                rename_process(pname, new_name_stripped)
                st.toast(f"✏️ 工程名を {pname} → {new_name_stripped} に変更しました")
                st.rerun()

            # ── ステップチップ（クリックで編集ダイアログ）──────────
            # ボタン自体をステップ色のチップとして表示。
            # CSS :has() で直前マーカー span を起点にボタンを着色。
            if steps_list:
                n_chips = len(steps_list)
                chip_cols = st.columns(min(6, n_chips))
                for ci, step in enumerate(steps_list):
                    clr   = step.get("color", _default_color(ci))
                    lbl   = step.get("name", f"Step{ci+1}")
                    mode  = step.get("mode", "single")
                    icon  = "↔️" if mode == "range" else "→"
                    with chip_cols[ci % min(6, n_chips)]:
                        # マーカー span + CSS injection で、直後ボタンを着色
                        # DOM: stElementContainer > stMarkdown > stMarkdownContainer > span#id
                        #       ↕ (adjacent sibling)
                        #      stElementContainer > stButton > ... > button
                        safe_id = "cm_" + "".join(
                            c if c.isalnum() else "_" for c in f"{pname}_{ci}"
                        )
                        _is_sel = (lbl == st.session_state.get(f"sel_step_{pname}", ""))
                        _sel_css = (
                            f'    background: linear-gradient(90deg,{clr}55 0%,{clr}22 55%) !important;'
                            f'    box-shadow: 0 0 0 2px {clr}88 !important;'
                        ) if _is_sel else (
                            f'    background: linear-gradient(90deg,{clr}28 0%,transparent 55%) !important;'
                        )
                        st.markdown(
                            f'<span id="{safe_id}"></span>'
                            f'<style>'
                            f'[data-testid="stElementContainer"]:has(#{safe_id}) + '
                            f'[data-testid="stElementContainer"] button'
                            f' {{ border-left: 5px solid {clr} !important;'
                            f'    padding-left: 10px !important;'
                            f'{_sel_css}}}'
                            f'</style>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"{icon} {lbl}",
                            key=f"chip_{pname}_{ci}",
                            use_container_width=True,
                            help="クリックして編集",
                        ):
                            edit_step_dialog(pname, ci, bool_cols, df)

            # ── メインビュー ──────────────────────────────────────
            result_df = pd.DataFrame()

            if not steps_list:
                st.info("「＋ ステップを追加」からステップ変数を追加するとガントチャートが表示されます")
            else:
                # ステップ JSON をキャッシュキーに使用
                steps_json = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
                try:
                    result_df = cached_analyze_v2(df, trigger_col, edge, steps_json)
                except Exception as e:
                    st.error(f"解析エラー: {e}")
                    result_df = pd.DataFrame()

                if result_df is None or len(result_df) == 0:
                    st.warning("サイクルデータが取得できません。設定の基準変数を確認してください。")
                else:
                    fig_gantt, step_stats = build_gantt_v2(result_df, steps_list, takt_target)

                    # ── 時系列自動並び替え ──────────────────────────
                    if step_stats:
                        _sort_col, _ = st.columns([2, 8])
                        with _sort_col:
                            if st.button(
                                "⇅ 時系列で自動整列",
                                key=f"auto_sort_{pname}",
                                help="ステップをトリガーからの経過時刻順に並び替えます",
                            ):
                                def _abs_t(stat):
                                    if stat.get("mode") == "range":
                                        return stat.get("abs_start", stat.get("abs_mean", 0))
                                    return stat.get("abs_mean", 0)
                                sort_map = {s["name"]: _abs_t(s) for s in step_stats}
                                sorted_steps = sorted(
                                    steps_list,
                                    key=lambda sl: sort_map.get(sl.get("name", ""), 0),
                                )
                                st.session_state[pk(pname, "steps_list")] = sorted_steps
                                st.toast("⇅ ステップを時系列順に並び替えました")
                                st.rerun()

                    if fig_gantt:
                        # ── ヒストグラム詳細ページ用コンテキストを保存 ─────
                        st.session_state[f"_hist_ctx_{pname}"] = {
                            "step_stats": step_stats,
                            "steps_list": steps_list,
                            "result_df":  result_df,
                            "baseline":   st.session_state.get(pk(pname, "baseline"), {}),
                        }

                        # ── ヒストグラム URL マップ（ダブルクリック遷移用）──
                        _hist_urls = {
                            s["name"]: (
                                f"/histogram"
                                f"?proc={urllib.parse.quote(pname)}"
                                f"&step_name={urllib.parse.quote(s['name'])}"
                            )
                            for s in step_stats
                        }
                        _safe_pn = "".join(c if c.isalnum() else "_" for c in pname)
                        _gm_id   = f"gm_{_safe_pn}"

                        # ── マーカー div（JS がガントチャートを特定するため）──
                        st.markdown(f'<div id="{_gm_id}"></div>', unsafe_allow_html=True)

                        # ── ガントチャート（クリックでステップ詳細連動）──
                        gantt_event = st.plotly_chart(
                            fig_gantt, use_container_width=True,
                            key=f"gantt_{pname}", on_select="rerun",
                        )
                        try:
                            pts = (gantt_event.selection.get("points", [])
                                   if gantt_event and gantt_event.selection else [])
                            if pts:
                                clicked_y  = pts[0].get("y", "")
                                step_names = [s["name"] for s in step_stats]
                                if clicked_y in step_names:
                                    st.session_state[f"sel_step_{pname}"] = clicked_y
                        except Exception:
                            pass

                        # ── ダブルクリック → ヒストグラムを別タブで開く ──────
                        # マーカー div の次の stElementContainer 内 Plotly チャートを特定し
                        # 2 回クリック（< 450ms）で window.open('/histogram?...')
                        _urls_js = json.dumps(_hist_urls, ensure_ascii=False)
                        _flag    = f"_dbl_{_safe_pn}"
                        st.components.v1.html(f"""<script>
    (function(){{
      var URLS={_urls_js};
      var MID="{_gm_id}";
      // ダブルクリック状態を親ウィンドウに保持
      // → Streamlit の rerun でiframeがリロードされても状態が消えない
      if(!window.parent._dbl) window.parent._dbl={{}};
      if(!window.parent._dbl[MID]) window.parent._dbl[MID]={{t:0,n:""}};
      var _st=window.parent._dbl[MID];

      function findGd(){{
        var m=window.parent.document.getElementById(MID);
        if(!m) return null;
        var c=m;
        while(c&&(!c.getAttribute||c.getAttribute('data-testid')!=='stElementContainer'))
          c=c.parentElement;
        if(!c) return null;
        var sib=c.nextElementSibling;
        while(sib){{
          var gd=sib.querySelector('.js-plotly-plot');
          if(gd) return gd;
          sib=sib.nextElementSibling;
        }}
        return null;
      }}
      function attach(){{
        var gd=findGd();
        if(!gd||gd['{_flag}']) return;
        gd['{_flag}']=true;
        gd.on('plotly_click',function(ev){{
          if(!ev||!ev.points||!ev.points.length) return;
          var name=ev.points[0].data.name||"";
          var now=Date.now();
          // 500ms以内に同じステップを2回クリック → ヒストグラムページへ遷移
          if(now-_st.t<500&&name===_st.n&&URLS[name]){{
            window.parent.location.href=URLS[name];
            _st.t=0;
          }} else {{
            _st.t=now; _st.n=name;
          }}
        }});
      }}
      attach();
      setTimeout(attach,400);
      setTimeout(attach,1500);
    }})();
    </script>""", height=0)

                        # ── ステップ統計テーブル（B-3: Cpk含む）──────────
                        total_t = sum(s["mean"] for s in step_stats) or 1.0
                        rows = []
                        for s in step_stats:
                            mean_v = s["mean"]
                            std_v  = s["abs_std"]
                            # B-3: Cpk 計算 (USL=takt_target, LSL=0)
                            if takt_target > 0 and std_v > 0:
                                cpk = min(
                                    (takt_target - mean_v) / (3 * std_v),
                                    mean_v / (3 * std_v)
                                )
                                if cpk >= 1.33:
                                    cpk_str = f"🟢 {cpk:.2f}"
                                elif cpk >= 1.0:
                                    cpk_str = f"🟡 {cpk:.2f}"
                                else:
                                    cpk_str = f"🔴 {cpk:.2f}"
                            else:
                                cpk_str = "—"
                            row = {
                                "ステップ名": s["name"],
                                "モード":    "範囲" if s["mode"] == "range" else "単一",
                                "平均[ms]":  round(mean_v, 2),
                                "σ[ms]":    round(std_v, 2),
                                "min[ms]":  round(s.get("abs_min", s["min"]), 2),
                                "max[ms]":  round(s.get("abs_max", s["max"]), 2),
                                "タクト比率": f"{mean_v/total_t*100:.1f}%",
                                "Cpk":      cpk_str,
                            }
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), hide_index=True,
                                     use_container_width=True)
                        if takt_target == 0:
                            st.caption("💡 Cpk を表示するにはサイクル設定でタクト目標を設定してください")

                        # ── B-2: IQR外れ値検出 警告カード ───────────────
                        _outliers = detect_outliers_iqr(result_df, step_stats)
                        if _outliers:
                            for _ov in _outliers:
                                _cyc_str = ", ".join(str(c) for c in _ov["cycles"])
                                st.warning(
                                    f"⚠️ **{_ov['name']}**: サイクル {_cyc_str} で外れ値検出"
                                    f"（IQR境界: {_ov['lo']:.1f}〜{_ov['hi']:.1f} ms）"
                                )

                        # ── 基準値管理 ──────────────────────────────────
                        st.divider()
                        _bl      = st.session_state.get(pk(pname, "baseline"), {})
                        _bl_meta = st.session_state.get(pk(pname, "baseline_meta"), {})
                        bl_l, bl_r = st.columns([7, 3])
                        with bl_l:
                            if _bl:
                                st.success(
                                    f"📐 基準値登録済み — {_bl_meta.get('source','?')} / "
                                    f"{_bl_meta.get('n_cycles','?')} サイクル "
                                    f"({_bl_meta.get('created_at','')})"
                                )
                            else:
                                st.caption("📐 基準値を登録するとヒストグラムが差分表示になります")
                        with bl_r:
                            if st.button(
                                "📐 基準値を登録・編集",
                                key=f"bl_btn_{pname}",
                                use_container_width=True,
                                disabled=not step_stats,
                            ):
                                baseline_dialog(pname, step_stats, result_df)

                        st.divider()

                        # ── ステップ詳細 ──────────────────────────────
                        st.markdown("**ステップ詳細**")
                        step_names = [s["name"] for s in step_stats]
                        cur_sel = st.session_state.get(f"sel_step_{pname}", step_names[0] if step_names else "")
                        if cur_sel not in step_names and step_names:
                            cur_sel = step_names[0]
                        sel_name = st.selectbox(
                            "詳細を見るステップ", step_names,
                            index=step_names.index(cur_sel) if cur_sel in step_names else 0,
                            key=f"sel_step_{pname}",
                            label_visibility="collapsed",
                        )
                        sel_stat = next((s for s in step_stats if s["name"] == sel_name), None)
                        sel_step = next((s for s in steps_list if s.get("name") == sel_name), None)
                        if sel_stat and sel_step:
                            render_step_detail(df, trigger_col, edge,
                                               sel_stat, sel_step, pname, result_df)

            # ── 詳細解析タブ（補助）────────────────────────────────
            if steps_list and len(result_df) > 0:
                with st.expander("📊 詳細解析タブ", expanded=False):
                    delay_cols = [c for c in result_df.columns
                                  if c.endswith("_遅れ[ms]") or c.endswith("_dur[ms]")]
                    thr_tab = st.number_input(
                        "閾値[ms]（0=なし）", min_value=0.0, value=0.0, step=0.5,
                        key=f"thr_tab_{pname}",
                    )
                    try:
                        cycle_starts = cached_detect_cycles(df, trigger_col, edge)
                    except Exception:
                        cycle_starts = []

                    tabs = st.tabs(["サイクル一覧", "ヒストグラム", "時系列波形", "📈 トレンド"])

                    with tabs[0]:
                        st.dataframe(result_df, use_container_width=True)
                        st.download_button("CSVダウンロード",
                                           result_df.to_csv(index=False, encoding="utf-8-sig"),
                                           f"{pname}_cycles.csv", key=f"dl_csv_{pname}")

                    with tabs[1]:
                        for col in delay_cols:
                            vn    = col.replace("_遅れ[ms]", "").replace("_dur[ms]", " (所要時間)")
                            dl    = result_df[col].dropna().values
                            if len(dl) == 0:
                                continue
                            _bkey_t = f"{pname}_{col}_t"
                            nb  = calc_nice_bins(dl, _bkey_t)   # Freedman-Diaconis 自動算出
                            st_ = calc_statistics(dl)
                            sg3 = st_.get("3σ上限[ms]", 0)
                            fig = go.Figure()
                            if thr_tab > 0:
                                bl = [d for d in dl if d <= thr_tab]
                                ab = [d for d in dl if d > thr_tab]
                                if bl:
                                    fig.add_trace(go.Histogram(x=bl, nbinsx=nb, name="閾値以内",
                                                               marker_color="royalblue", opacity=0.7))
                                if ab:
                                    fig.add_trace(go.Histogram(x=ab, nbinsx=nb, name="閾値超過",
                                                               marker_color="crimson", opacity=0.7))
                                fig.add_vline(x=thr_tab, line_dash="dash", line_color="orange",
                                              annotation_text=f"閾値 {thr_tab}ms")
                            else:
                                fig.add_trace(go.Histogram(x=dl, nbinsx=nb,
                                                           marker_color="steelblue", opacity=0.8))
                            fig.add_vline(x=sg3, line_dash="dot", line_color="gray",
                                          annotation_text=f"3σ {sg3:.1f}ms")
                            fig.update_layout(title=vn, xaxis_title="時間[ms]",
                                              barmode="overlay", height=250, margin=dict(t=28))
                            st.plotly_chart(fig, use_container_width=True, key=f"t2_{pname}_{vn}")
                            st.slider("ビン数", 3, 60, nb, key=f"_bins_{_bkey_t}",
                                      help="ヒストグラムのビン数を手動調整（Freedman-Diaconis による自動算出が既定値）")
                            if st_:
                                sc = st.columns(min(6, len(st_)))
                                for i, (k, v) in enumerate(list(st_.items())[:6]):
                                    sc[i].metric(k, v)
                            st.markdown("---")

                    with tabs[2]:
                        all_vars_disp = steps_all_vars(steps_list, bool_cols)
                        ts_b = st.multiselect("Bool変数", bool_cols,
                                              default=all_vars_disp[:3],
                                              key=f"ts_b_{pname}")
                        ts_n = st.multiselect("数値変数（第2Y軸）", num_cols,
                                              default=[], key=f"ts_n_{pname}")
                        if ts_b or ts_n:
                            fig = go.Figure()
                            for var in ts_b:
                                fig.add_trace(go.Scatter(
                                    x=df["Timestamp"],
                                    y=normalize_bool_series(df[var]),
                                    name=var, fill="tozeroy", mode="lines",
                                    line=dict(width=1)))
                            for var in ts_n:
                                fig.add_trace(go.Scatter(
                                    x=df["Timestamp"], y=df[var],
                                    name=var, mode="lines", yaxis="y2"))
                            if ts_n:
                                fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
                            for idx_c in cycle_starts[:50]:
                                if idx_c in df.index:
                                    fig.add_vline(x=df.loc[idx_c, "Timestamp"],
                                                  line=dict(color="green", width=0.5, dash="dash"))
                            fig.update_layout(xaxis_title="時刻", height=420)
                            st.plotly_chart(fig, use_container_width=True, key=f"t3_{pname}")

                    with tabs[3]:
                        # B-1: トレンドチャート
                        _trend_step_stats = []
                        try:
                            _, _trend_step_stats = build_gantt_v2(result_df, steps_list, 0)
                        except Exception:
                            pass
                        _trend_fig = build_trend_chart(result_df, _trend_step_stats, takt_target)
                        if _trend_fig:
                            st.caption(
                                "各サイクルの遅れ時間推移。赤丸は外れ値（|値-平均| > 2σ）、"
                                "点線は mean+3σ 上限。"
                            )
                            st.plotly_chart(_trend_fig, use_container_width=True,
                                            key=f"trend_{pname}")
                        else:
                            st.info("トレンドを表示するにはステップを追加してください")

        st.markdown("")

    if False:  # 工程間タイムライン比較（削除）
        with st.expander("", expanded=False):
            proc_names = list(processes.keys())
            selected_procs = st.multiselect(
                "比較する工程",
                proc_names,
                default=proc_names,
                key="cmp_procs_sel",
            )

            if selected_procs:
                # 各工程のステップ統計を収集
                all_bars: list[dict] = []
                takt_lines: list[tuple] = []   # (takt_ms, proc_label)

                for cmp_pname in selected_procs:
                    cmp_steps  = st.session_state.get(pk(cmp_pname, "steps_list"), [])
                    if not cmp_steps:
                        continue
                    cmp_trig   = st.session_state.get(pk(cmp_pname, "trigger"), bool_cols[0])
                    cmp_edge   = st.session_state.get(pk(cmp_pname, "edge"), "RISE")
                    cmp_takt   = int(st.session_state.get(pk(cmp_pname, "takt"), 0))

                    try:
                        cmp_json = json.dumps(cmp_steps, ensure_ascii=False, sort_keys=True)
                        cmp_res  = cached_analyze_v2(df, cmp_trig, cmp_edge, cmp_json)
                    except Exception:
                        continue
                    if cmp_res is None or len(cmp_res) == 0:
                        continue

                    _, cmp_stats = build_gantt_v2(cmp_res, cmp_steps, 0)
                    for s in cmp_stats:
                        all_bars.append({
                            **s,
                            "label": f"{cmp_pname}　/　{s['name']}",
                            "proc":  cmp_pname,
                        })
                    if cmp_takt > 0:
                        takt_lines.append((cmp_takt, cmp_pname))

                if all_bars:
                    fig_cmp = go.Figure()
                    for b in all_bars:
                        is_range = (b.get("mode") == "range")
                        if is_range:
                            ht = (
                                f"<b>%{{y}}</b> [範囲]<br>"
                                f"開始: {b.get('abs_start', b['start']):.1f}ms<br>"
                                "長さ（平均）: %{customdata[1]:.1f} ms<br>"
                                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms"
                                "<extra></extra>"
                            )
                        else:
                            ht = (
                                "<b>%{y}</b><br>"
                                "区間長（平均）: %{customdata[1]:.1f} ms<br>"
                                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms"
                                "<extra></extra>"
                            )
                        fig_cmp.add_trace(go.Bar(
                            name=b["label"],
                            y=[b["label"]], x=[b["mean"]], base=[b["start"]],
                            orientation="h", width=0.6,
                            marker_color=b["color"],
                            marker_pattern_shape="/" if is_range else "",
                            customdata=[[b["start"], b["mean"], b["min"], b["max"]]],
                            hovertemplate=ht,
                        ))
                        # ばらつき帯
                        bw = b["max"] - b["min"]
                        if bw > 0:
                            fig_cmp.add_trace(go.Bar(
                                y=[b["label"]], x=[bw], base=[b["start"] + b["min"]],
                                orientation="h", width=0.6,
                                marker_color=b["color"], opacity=0.18,
                                showlegend=False, hoverinfo="skip",
                            ))

                    # タクト目標ライン（工程ごとに色分け）
                    takt_colors = ["red", "darkorange", "purple", "brown"]
                    for ti, (takt_ms, takt_proc) in enumerate(takt_lines):
                        fig_cmp.add_vline(
                            x=takt_ms,
                            line_dash="dash",
                            line_color=takt_colors[ti % len(takt_colors)],
                            annotation_text=f"{takt_proc} 目標 {takt_ms}ms",
                            annotation_position="top",
                        )

                    fig_cmp.update_layout(
                        barmode="overlay",
                        xaxis_title="サイクル開始からの経過時間 [ms]",
                        yaxis=dict(autorange="reversed"),
                        height=max(300, len(all_bars) * 50 + 120),
                        margin=dict(l=0, r=20, t=16, b=40),
                        showlegend=False,
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_cmp, use_container_width=True,
                                    key="cmp_gantt_multi")

                    # 工程別タクト消費サマリー
                    st.markdown("**工程別タクト消費サマリー**")
                    rows_cmp = []
                    for cmp_pname in selected_procs:
                        proc_bars = [b for b in all_bars if b["proc"] == cmp_pname]
                        if not proc_bars:
                            continue
                        total_ms  = sum(b["mean"] for b in proc_bars)
                        cmp_takt  = int(st.session_state.get(pk(cmp_pname, "takt"), 0))
                        usage_pct = f"{total_ms / cmp_takt * 100:.1f}%" if cmp_takt > 0 else "—"
                        rows_cmp.append({
                            "工程名":         cmp_pname,
                            "ステップ数":     len(proc_bars),
                            "合計時間[ms]":   round(total_ms, 1),
                            "タクト目標[ms]": cmp_takt if cmp_takt > 0 else "—",
                            "タクト消費率":   usage_pct,
                        })
                    if rows_cmp:
                        st.dataframe(pd.DataFrame(rows_cmp), hide_index=True,
                                     use_container_width=True)
                else:
                    st.info("比較対象の工程にステップが設定されていません。")


# Tab 2: 新データ評価
# ═══════════════════════════════════════════════════════════════

with _page_tabs[1]:
    st.subheader("📐 新データ評価")

    if not processes:
        st.info("「解析」タブで工程を設定してください")
    else:
        _ev_proc_list = list(processes.keys())
        _ev_pname = st.selectbox("工程を選択", _ev_proc_list, key="ev_page_proc")
        _ev_trigger = st.session_state.get(pk(_ev_pname, "trigger"), bool_cols[0] if bool_cols else "")
        _ev_edge    = st.session_state.get(pk(_ev_pname, "edge"), "RISE")
        _ev_steps   = st.session_state.get(pk(_ev_pname, "steps_list"), [])
        _ev_takt    = int(st.session_state.get(pk(_ev_pname, "takt"), 0))
        _ev_bl      = st.session_state.get(pk(_ev_pname, "baseline"), {})
        _ev_bl_meta = st.session_state.get(pk(_ev_pname, "baseline_meta"), {})

        if not _ev_steps:
            st.info("「解析」タブでこの工程のステップを設定してください")
        elif not _ev_bl:
            st.warning("基準値が未登録です。「解析」タブのガントチャート下「📐 基準値を登録・編集」で登録してください。")
        else:
            st.success(
                f"基準値: {_ev_bl_meta.get('source','?')} / "
                f"{_ev_bl_meta.get('n_cycles','?')} サイクル "
                f"({_ev_bl_meta.get('created_at','')})"
            )

            ev_page_key = f"_evpage_df_{_ev_pname}"
            _evh, _evc = st.columns([5, 1])
            with _evh:
                ev_page_file = st.file_uploader(
                    "評価するCSVをここにドロップ", type=["csv"],
                    key=f"ev_page_up_{_ev_pname}", label_visibility="collapsed",
                )
            with _evc:
                if ev_page_key in st.session_state:
                    if st.button("✕ クリア", key=f"ev_page_clr_{_ev_pname}",
                                 use_container_width=True):
                        del st.session_state[ev_page_key]
                        st.session_state.pop(f"{ev_page_key}_sig", None)
                        st.rerun()

            if ev_page_file:
                # ファイル名+サイズをシグネチャとして使い、同一ファイルの再読み込みを防ぐ
                _ev_sig = f"{ev_page_file.name}_{ev_page_file.size}"
                _ev_sig_key = f"{ev_page_key}_sig"
                if st.session_state.get(_ev_sig_key) != _ev_sig:
                    try:
                        st.session_state[ev_page_key] = load_csv(ev_page_file)
                        st.session_state[_ev_sig_key] = _ev_sig
                    except Exception as _e:
                        st.error(f"読み込みエラー: {_e}")

            if ev_page_key in st.session_state:
                _ev_df = st.session_state[ev_page_key]
                st.info(f"評価データ {len(_ev_df):,} 行")

                _ev_steps_json = json.dumps(_ev_steps, ensure_ascii=False, sort_keys=True)
                try:
                    _ev_result = cached_analyze_v2(_ev_df, _ev_trigger, _ev_edge, _ev_steps_json)
                except Exception as _e:
                    st.error(f"解析エラー: {_e}")
                    _ev_result = None

                if _ev_result is not None and len(_ev_result) > 0:
                    # ガントチャート（キャッシュ済みラッパーで再計算を抑止）
                    _ev_fig, _ev_step_stats = cached_build_gantt(
                        _ev_result, _ev_steps_json, _ev_takt
                    )
                    if _ev_fig:
                        st.markdown("**ガントチャート（評価データ）**")
                        st.plotly_chart(_ev_fig, use_container_width=True,
                                        key=f"evpage_gantt_{_ev_pname}")

                    # σ倍率 & サマリーテーブル
                    _ev_sigma = st.number_input(
                        "異常判定 (σ倍)", value=3.0, step=0.5, min_value=1.0,
                        key=f"ev_page_sigma_{_ev_pname}",
                    )
                    _ev_delta_rows = []
                    for _s in (_ev_step_stats or []):
                        _sn = _s["name"]
                        _sm = _s["mode"]
                        _ble = _ev_bl.get(_sn, {})
                        _col = f"{_sn}_遅れ[ms]" if _sm == "single" else f"{_sn}_dur[ms]"
                        _ref = _ble.get("ref_ms") if _sm == "single" else _ble.get("ref_dur_ms")
                        _std = _ble.get("std_ms", 0.0) if _sm == "single" else _ble.get("std_dur_ms", 0.0)
                        if _col not in _ev_result.columns or _ref is None:
                            continue
                        _vals = _ev_result[_col].dropna().values
                        if len(_vals) == 0:
                            continue
                        _deltas = _vals - _ref
                        _thresh = _ev_sigma * _std if _std > 0 else None
                        _ng = int(np.sum(np.abs(_deltas) > _thresh)) if _thresh else 0
                        _rate = _ng / len(_deltas) * 100
                        _ev_delta_rows.append({
                            "ステップ":     _sn,
                            "基準[ms]":    round(_ref, 1),
                            "評価平均[ms]": round(float(np.mean(_vals)), 1),
                            "差分平均[ms]": round(float(np.mean(_deltas)), 1),
                            "差分σ[ms]":   round(float(np.std(_deltas)), 1),
                            "NG件数":       _ng,
                            "NG率":         f"{_rate:.1f}%",
                            "判定":         "🔴" if _rate > 0 else "🟢",
                        })
                    if _ev_delta_rows:
                        st.markdown("**ステップ別サマリー**")
                        st.dataframe(pd.DataFrame(_ev_delta_rows),
                                     hide_index=True, use_container_width=True)

                    # ステップ詳細（差分ヒストグラム）
                    st.divider()
                    st.markdown("**ステップ詳細（基準値との差分）**")
                    _ev_step_names = [s["name"] for s in (_ev_step_stats or [])]
                    if _ev_step_names:
                        _ev_sel = st.selectbox(
                            "詳細を見るステップ", _ev_step_names,
                            key=f"ev_page_sel_{_ev_pname}",
                            label_visibility="collapsed",
                        )
                        _ev_stat = next(
                            (s for s in (_ev_step_stats or []) if s["name"] == _ev_sel), None)
                        _ev_step_cfg = next(
                            (s for s in _ev_steps if s.get("name") == _ev_sel), None)
                        if _ev_stat and _ev_step_cfg:
                            # ウィジェットキー重複を避けるためサフィックス付き pname を使用
                            _pname_ev = _ev_pname + "::ev"
                            # 基準値を ev 用キーにコピーして render 内の baseline 参照を解決
                            _bl_src = pk(_ev_pname, "baseline")
                            if _bl_src in st.session_state:
                                st.session_state[pk(_pname_ev, "baseline")] = (
                                    st.session_state[_bl_src]
                                )
                            render_step_detail(
                                _ev_df, _ev_trigger, _ev_edge,
                                _ev_stat, _ev_step_cfg, _pname_ev, _ev_result,
                            )

                    # サイクル別判定テーブル（ベクトル化で高速化）
                    st.divider()
                    st.markdown("**サイクル別判定**")
                    _ev_cyc_df = pd.DataFrame({"サイクル#": _ev_result["サイクル#"].astype(int)})
                    _ng_any = pd.Series(False, index=_ev_result.index)
                    for _s in (_ev_step_stats or []):
                        _sn = _s["name"]
                        _sm = _s["mode"]
                        _ble = _ev_bl.get(_sn, {})
                        _col = (f"{_sn}_遅れ[ms]" if _sm == "single"
                                else f"{_sn}_dur[ms]")
                        _ref = (_ble.get("ref_ms") if _sm == "single"
                                else _ble.get("ref_dur_ms"))
                        _std = (_ble.get("std_ms", 0.0) if _sm == "single"
                                else _ble.get("std_dur_ms", 0.0))
                        if _col not in _ev_result.columns or _ref is None:
                            continue
                        _delta_s = _ev_result[_col] - _ref          # Series演算（高速）
                        _thresh = _ev_sigma * _std if _std > 0 else None
                        _is_ng_s = (
                            _delta_s.abs() > _thresh
                            if _thresh is not None
                            else pd.Series(False, index=_ev_result.index)
                        )
                        _ng_any |= _is_ng_s
                        # 書式はリスト内包表記で（行ループより大幅に速い）
                        _ev_cyc_df[_sn] = [
                            f"{'🔴 ' if ng else ''}{d:+.1f}ms" if pd.notna(d) else ""
                            for d, ng in zip(_delta_s, _is_ng_s)
                        ]
                    _ev_cyc_df["総合判定"] = [
                        "🔴 NG" if ng else "🟢 OK" for ng in _ng_any
                    ]
                    if not _ev_cyc_df.empty:
                        st.dataframe(_ev_cyc_df, hide_index=True, use_container_width=True)
                        st.download_button(
                            "サイクル判定CSVダウンロード",
                            _ev_cyc_df.to_csv(index=False, encoding="utf-8-sig"),
                            f"{_ev_pname}_eval.csv",
                            key=f"ev_page_dl_{_ev_pname}",
                        )
            else:
                st.caption("評価したいCSVをアップロードしてください")


# ═══════════════════════════════════════════════════════════════
# 傾向解析タブ
# ═══════════════════════════════════════════════════════════════

with _page_tabs[2]:
    import re as _re

    st.subheader("📈 傾向解析")
    st.caption("複数時期のCSVを登録し、ステップ毎のタイミング傾向を可視化します")

    if not processes:
        st.info("「⚙️ 画面設定」タブで工程を設定してください")
    else:
        _tr_pname = st.selectbox(
            "工程を選択", list(processes.keys()), key="tr_page_proc"
        )
        _tr_trigger = st.session_state.get(pk(_tr_pname, "trigger"), bool_cols[0] if bool_cols else "")
        _tr_edge    = st.session_state.get(pk(_tr_pname, "edge"), "RISE")
        _tr_steps   = st.session_state.get(pk(_tr_pname, "steps_list"), [])
        _tr_bl      = st.session_state.get(pk(_tr_pname, "baseline"), {})

        if not _tr_steps:
            st.info("ステップが設定されていません。「⚙️ 画面設定」タブでステップを追加してください")
        else:
            # ── ファイル管理 ────────────────────────────────────────
            st.markdown("#### 📂 時期別CSVの登録")
            st.caption("時系列順に複数のCSVをアップロードしてください。ファイル名から日付を自動検出します。")

            _tr_uploaded = st.file_uploader(
                "CSVを選択（複数可）", type=["csv"],
                accept_multiple_files=True, key=f"tr_files_{_tr_pname}",
            )

            # ラベル管理（session_stateで保持）
            _tr_labels_key = f"_tr_labels_{_tr_pname}"
            if _tr_labels_key not in st.session_state:
                st.session_state[_tr_labels_key] = {}

            if _tr_uploaded:
                st.markdown("**ファイルラベル設定**")
                st.caption("各ファイルの時期ラベルを編集できます（グラフの X 軸に使用）")
                _n_cols = min(len(_tr_uploaded), 4)
                _label_cols = st.columns(_n_cols)
                for _ti, _tf in enumerate(_tr_uploaded):
                    _lk = _tf.name
                    if _lk not in st.session_state[_tr_labels_key]:
                        _dm = _re.search(r"(\d{4}[-_/]?\d{2}[-_/]?\d{2})", _tf.name)
                        _auto = (
                            _dm.group(1).replace("_", "-").replace("/", "-")
                            if _dm else _tf.name.replace(".csv", "")
                        )
                        st.session_state[_tr_labels_key][_lk] = _auto
                    with _label_cols[_ti % _n_cols]:
                        st.session_state[_tr_labels_key][_lk] = st.text_input(
                            _tf.name,
                            value=st.session_state[_tr_labels_key][_lk],
                            key=f"tr_lbl_{_tr_pname}_{_tf.name}",
                        )

                # ── 解析実行 ────────────────────────────────────────
                st.divider()
                if st.button(
                    "📊 傾向解析を実行", type="primary",
                    use_container_width=True, key=f"tr_run_{_tr_pname}",
                ):
                    _tr_steps_json = json.dumps(_tr_steps)
                    _res_list = []
                    _prog = st.progress(0, text="解析中...")
                    for _ti, _tf in enumerate(_tr_uploaded):
                        _prog.progress(
                            (_ti + 1) / len(_tr_uploaded),
                            text=f"解析中 {_ti + 1}/{len(_tr_uploaded)}: {_tf.name}",
                        )
                        try:
                            _tdf = load_csv(_tf)
                            _tres = cached_analyze_v2(
                                _tdf, _tr_trigger, _tr_edge, _tr_steps_json
                            )
                            _lbl = st.session_state[_tr_labels_key].get(_tf.name, _tf.name)
                            _res_list.append(
                                {"label": _lbl, "fname": _tf.name, "result": _tres}
                            )
                        except Exception as _te:
                            st.warning(f"{_tf.name}: 解析失敗 ({_te})")
                    _prog.empty()
                    st.session_state[f"_tr_results_{_tr_pname}"] = _res_list
                    st.toast(f"✅ {len(_res_list)} ファイルの解析が完了しました")
                    st.rerun()

                # ── 結果表示 ────────────────────────────────────────
                _tr_res_list = st.session_state.get(f"_tr_results_{_tr_pname}", [])
                if _tr_res_list:
                    st.divider()

                    # ── チャートタイプ切り替え ────────────────────────
                    _chart_mode = st.radio(
                        "表示形式",
                        ["📈 傾向チャート（平均 ± σ）", "📊 Xbar-R 管理図"],
                        horizontal=True,
                        key=f"tr_chart_mode_{_tr_pname}",
                    )

                    _tr_summary_rows = []

                    # ══════════════════════════════════════════════════
                    # A) 傾向チャート（従来）
                    # ══════════════════════════════════════════════════
                    if _chart_mode == "📈 傾向チャート（平均 ± σ）":
                        st.markdown("#### 📈 傾向チャート")
                        _tr_sigma = st.slider(
                            "異常判定 σ 倍数", 1.0, 5.0, 3.0, 0.5,
                            key=f"tr_sigma_{_tr_pname}",
                            help="基準値±Nσ を超えたポイントを 🔴 でマーク",
                        )

                        for _ts in _tr_steps:
                            _tsn  = _ts["name"]
                            _tsm  = _ts["mode"]
                            _tsc  = _ts.get("color", "#1f77b4")
                            _dcol = (
                                f"{_tsn}_遅れ[ms]" if _tsm == "single"
                                else f"{_tsn}_dur[ms]"
                            )

                            _tr_lbls, _tr_means, _tr_stds = [], [], []
                            for _r in _tr_res_list:
                                _rdf = _r["result"]
                                if _dcol not in _rdf.columns:
                                    continue
                                _rv = _rdf[_dcol].dropna().values
                                if len(_rv) == 0:
                                    continue
                                _tr_lbls.append(_r["label"])
                                _tr_means.append(float(np.mean(_rv)))
                                _tr_stds.append(float(np.std(_rv)))

                            if not _tr_lbls:
                                continue

                            _tbl  = _tr_bl.get(_tsn, {})
                            _tref = (
                                _tbl.get("ref_ms") if _tsm == "single"
                                else _tbl.get("ref_dur_ms")
                            )
                            _tbls = (
                                _tbl.get("std_ms", 0.0) if _tsm == "single"
                                else _tbl.get("std_dur_ms", 0.0)
                            )

                            _is_ng = [
                                bool(
                                    _tref is not None and _tbls > 0
                                    and abs(m - _tref) > _tr_sigma * _tbls
                                )
                                for m in _tr_means
                            ]

                            _sr = {"ステップ": _tsn, "モード": _tsm}
                            for _li, (_lbl, _m, _s) in enumerate(
                                zip(_tr_lbls, _tr_means, _tr_stds)
                            ):
                                _sr[_lbl] = f"{'🔴 ' if _is_ng[_li] else ''}{_m:.1f} ± {_s:.1f}"
                            _tr_summary_rows.append(_sr)

                            _fig_tr = go.Figure()
                            _hex = _tsc.lstrip("#")
                            _rgb = tuple(int(_hex[i:i+2], 16) for i in (0, 2, 4)) if len(_hex) == 6 else (31, 119, 180)
                            _fill_col = f"rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.15)"
                            _fig_tr.add_trace(go.Scatter(
                                x=_tr_lbls + _tr_lbls[::-1],
                                y=(
                                    [m + s for m, s in zip(_tr_means, _tr_stds)]
                                    + [m - s for m, s in zip(_tr_means[::-1], _tr_stds[::-1])]
                                ),
                                fill="toself", fillcolor=_fill_col,
                                line=dict(width=0), showlegend=True, name="±1σ",
                            ))
                            _pt_colors  = ["#e74c3c" if ng else _tsc for ng in _is_ng]
                            _pt_symbols = ["x" if ng else "circle" for ng in _is_ng]
                            _fig_tr.add_trace(go.Scatter(
                                x=_tr_lbls, y=_tr_means,
                                mode="lines+markers",
                                line=dict(color=_tsc, width=2),
                                marker=dict(color=_pt_colors, size=10,
                                            symbol=_pt_symbols,
                                            line=dict(width=2, color="white")),
                                name="平均値",
                                hovertemplate="%{x}<br>平均: %{y:.2f} ms<br>σ: %{customdata:.2f} ms<extra></extra>",
                                customdata=_tr_stds,
                            ))
                            if _tref is not None:
                                _fig_tr.add_hline(
                                    y=_tref, line_dash="dash", line_color="#27ae60",
                                    annotation_text=f"基準 {_tref:.1f} ms",
                                    annotation_position="bottom right",
                                )
                                if _tbls > 0:
                                    _fig_tr.add_hrect(
                                        y0=_tref - _tr_sigma * _tbls,
                                        y1=_tref + _tr_sigma * _tbls,
                                        fillcolor="#27ae60", opacity=0.08, line_width=0,
                                        annotation_text=f"±{_tr_sigma:.0f}σ",
                                        annotation_position="top right",
                                    )
                            _fig_tr.update_layout(
                                title=dict(text=f"<b>{_tsn}</b>　傾向", font=dict(size=14)),
                                xaxis_title="時期", yaxis_title="[ms]",
                                height=290, margin=dict(t=48, b=40, l=60, r=20),
                                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                                hovermode="x unified",
                            )
                            st.plotly_chart(_fig_tr, use_container_width=True)

                    # ══════════════════════════════════════════════════
                    # B) Xbar-R 管理図
                    # ══════════════════════════════════════════════════
                    else:
                        st.markdown("#### 📊 Xbar-R 管理図")
                        st.caption(
                            "各時期のデータをサブグループとして管理限界線（UCL/CL/LCL）を算出。"
                            "n ≤ 25 は Xbar-R、n > 25 は Xbar-S（s̄ベース）で算出します。"
                        )

                        for _ts in _tr_steps:
                            _tsn  = _ts["name"]
                            _tsm  = _ts["mode"]
                            _tsc  = _ts.get("color", "#1f77b4")
                            _dcol = (
                                f"{_tsn}_遅れ[ms]" if _tsm == "single"
                                else f"{_tsn}_dur[ms]"
                            )

                            # ── サブグループデータ収集 ────────────────
                            _sg_lbls, _sg_xbar, _sg_r, _sg_s, _sg_n = [], [], [], [], []
                            for _r in _tr_res_list:
                                _rdf = _r["result"]
                                if _dcol not in _rdf.columns:
                                    continue
                                _rv = _rdf[_dcol].dropna().values
                                if len(_rv) < 2:
                                    continue
                                _sg_lbls.append(_r["label"])
                                _sg_xbar.append(float(np.mean(_rv)))
                                _sg_r.append(float(np.max(_rv) - np.min(_rv)))
                                _sg_s.append(float(np.std(_rv, ddof=1)))
                                _sg_n.append(len(_rv))

                            if len(_sg_lbls) < 2:
                                st.info(f"{_tsn}: データが 2 時期以上必要です")
                                continue

                            # ── サマリー行（共通）────────────────────
                            _sr = {"ステップ": _tsn, "モード": _tsm}
                            for _lbl, _m, _s in zip(_sg_lbls, _sg_xbar, _sg_s):
                                _sr[_lbl] = f"{_m:.1f} ± {_s:.1f}"
                            _tr_summary_rows.append(_sr)

                            # ── 管理限界計算 ──────────────────────────
                            _xbar_bar = float(np.mean(_sg_xbar))
                            _r_bar    = float(np.mean(_sg_r))
                            _s_bar    = float(np.mean(_sg_s))
                            _n_avg    = float(np.mean(_sg_n))
                            _n_rep    = int(round(_n_avg))

                            _A2, _D3, _D4, _c4, _B3, _B4 = _spc_consts(_n_rep)
                            _use_s = (_n_rep > 25)   # n>25 は S チャートの方が有効

                            if _use_s:
                                # Xbar-S 管理限界
                                _sigma_hat = _s_bar / _c4
                                _UCL_x = _xbar_bar + 3 * _sigma_hat / (_n_avg ** 0.5)
                                _LCL_x = _xbar_bar - 3 * _sigma_hat / (_n_avg ** 0.5)
                                _UCL_sub = _B4 * _s_bar
                                _LCL_sub = _B3 * _s_bar
                                _sub_vals  = _sg_s
                                _sub_label = "S（標準偏差）[ms]"
                                _sub_cl    = _s_bar
                                _cl_label  = f"S̄ = {_s_bar:.2f} ms"
                            else:
                                # Xbar-R 管理限界
                                _UCL_x = _xbar_bar + _A2 * _r_bar
                                _LCL_x = _xbar_bar - _A2 * _r_bar
                                _UCL_sub = _D4 * _r_bar
                                _LCL_sub = _D3 * _r_bar
                                _sub_vals  = _sg_r
                                _sub_label = "R（範囲）[ms]"
                                _sub_cl    = _r_bar
                                _cl_label  = f"R̄ = {_r_bar:.2f} ms"

                            # 管理外判定（UCL/LCL を超えたら NG）
                            _xbar_ng = [v > _UCL_x or v < _LCL_x for v in _sg_xbar]
                            _sub_ng  = [v > _UCL_sub or (_LCL_sub > 0 and v < _LCL_sub)
                                        for v in _sub_vals]

                            # ── サブプロット（上: Xbar / 下: R or S）────
                            _fig_xbr = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                subplot_titles=(
                                    f"X̄ 管理図  （UCL={_UCL_x:.2f}  CL={_xbar_bar:.2f}  LCL={_LCL_x:.2f}）",
                                    f"{'S' if _use_s else 'R'} 管理図  "
                                    f"（UCL={_UCL_sub:.2f}  CL={_sub_cl:.2f}"
                                    + (f"  LCL={_LCL_sub:.2f}" if _LCL_sub > 0 else "  LCL=0") + "）",
                                ),
                                vertical_spacing=0.14,
                                row_heights=[0.6, 0.4],
                            )

                            # X̄ チャート ──────────────────────────────
                            _xc  = ["#e74c3c" if ng else _tsc for ng in _xbar_ng]
                            _xs  = ["x-thin" if ng else "circle" for ng in _xbar_ng]
                            _fig_xbr.add_trace(go.Scatter(
                                x=_sg_lbls, y=_sg_xbar,
                                mode="lines+markers",
                                name="X̄",
                                line=dict(color=_tsc, width=2),
                                marker=dict(color=_xc, size=11, symbol=_xs,
                                            line=dict(width=2, color="white")),
                                hovertemplate="%{x}<br>X̄ = %{y:.3f} ms<extra></extra>",
                            ), row=1, col=1)

                            # X̄ 管理限界線
                            for _yv, _dash, _col, _ann in [
                                (_UCL_x,    "dash",  "#e74c3c", f"UCL={_UCL_x:.2f}"),
                                (_xbar_bar, "solid", "#27ae60", f"X̄̄={_xbar_bar:.2f}"),
                                (_LCL_x,    "dash",  "#e74c3c", f"LCL={_LCL_x:.2f}"),
                            ]:
                                _fig_xbr.add_hline(
                                    y=_yv, line_dash=_dash, line_color=_col,
                                    line_width=1.5,
                                    annotation_text=_ann,
                                    annotation_position="right",
                                    row=1, col=1,
                                )

                            # ±1σ 帯（X̄ チャート）
                            _hex2 = _tsc.lstrip("#")
                            _rgb2 = tuple(int(_hex2[i:i+2], 16) for i in (0, 2, 4)) if len(_hex2) == 6 else (31, 119, 180)
                            _sigma_x = (_UCL_x - _xbar_bar) / 3
                            _fig_xbr.add_hrect(
                                y0=_xbar_bar - _sigma_x,
                                y1=_xbar_bar + _sigma_x,
                                fillcolor=f"rgba({_rgb2[0]},{_rgb2[1]},{_rgb2[2]},0.10)",
                                line_width=0,
                                annotation_text="±1σ帯",
                                annotation_position="right",
                                row=1, col=1,
                            )

                            # R / S チャート ──────────────────────────
                            _rc  = ["#e74c3c" if ng else "#7f8c8d" for ng in _sub_ng]
                            _fig_xbr.add_trace(go.Scatter(
                                x=_sg_lbls, y=_sub_vals,
                                mode="lines+markers",
                                name="S" if _use_s else "R",
                                line=dict(color="#7f8c8d", width=2),
                                marker=dict(color=_rc, size=10,
                                            symbol=["x-thin" if ng else "circle" for ng in _sub_ng],
                                            line=dict(width=2, color="white")),
                                hovertemplate="%{x}<br>" + ("S" if _use_s else "R") + " = %{y:.3f} ms<extra></extra>",
                            ), row=2, col=1)

                            # R/S 管理限界線
                            _sub_lines = [
                                (_UCL_sub, "dash",  "#e74c3c", f"UCL={_UCL_sub:.2f}"),
                                (_sub_cl,  "solid", "#27ae60", _cl_label),
                            ]
                            if _LCL_sub > 0:
                                _sub_lines.append(
                                    (_LCL_sub, "dash", "#e74c3c", f"LCL={_LCL_sub:.2f}")
                                )
                            for _yv, _dash, _col, _ann in _sub_lines:
                                _fig_xbr.add_hline(
                                    y=_yv, line_dash=_dash, line_color=_col,
                                    line_width=1.5,
                                    annotation_text=_ann,
                                    annotation_position="right",
                                    row=2, col=1,
                                )

                            # 基準値ライン（X̄ チャートに重ねる）
                            _tbl  = _tr_bl.get(_tsn, {})
                            _tref = (
                                _tbl.get("ref_ms") if _tsm == "single"
                                else _tbl.get("ref_dur_ms")
                            )
                            if _tref is not None:
                                _fig_xbr.add_hline(
                                    y=_tref,
                                    line_dash="dot", line_color="#8e44ad", line_width=2,
                                    annotation_text=f"基準値 {_tref:.1f} ms",
                                    annotation_position="left",
                                    row=1, col=1,
                                )

                            _fig_xbr.update_layout(
                                title=dict(
                                    text=(
                                        f"<b>{_tsn}</b>　"
                                        f"Xbar-{'S' if _use_s else 'R'} 管理図"
                                        + (f"　（n={_n_rep}、Xbar-S で算出）" if _use_s else f"　（n={_n_rep}）")
                                    ),
                                    font=dict(size=14),
                                ),
                                height=480,
                                margin=dict(t=60, b=48, l=80, r=120),
                                showlegend=True,
                                legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                                hovermode="x unified",
                            )
                            _fig_xbr.update_yaxes(title_text="X̄ [ms]", row=1, col=1)
                            _fig_xbr.update_yaxes(title_text=_sub_label, row=2, col=1)
                            _fig_xbr.update_xaxes(title_text="時期", row=2, col=1)

                            # NG 点サマリー
                            _xbar_ng_cnt = sum(_xbar_ng)
                            _sub_ng_cnt  = sum(_sub_ng)
                            if _xbar_ng_cnt > 0 or _sub_ng_cnt > 0:
                                st.warning(
                                    f"**{_tsn}**: "
                                    f"X̄ 管理外 {_xbar_ng_cnt} 点　"
                                    f"{'S' if _use_s else 'R'} 管理外 {_sub_ng_cnt} 点"
                                )

                            st.plotly_chart(_fig_xbr, use_container_width=True,
                                            key=f"xbr_{_tr_pname}_{_tsn}")

                    # ── サマリーテーブル ────────────────────────────
                    if _tr_summary_rows:
                        st.divider()
                        st.markdown("#### 📋 サマリーテーブル（平均 ± σ）")
                        st.caption("🔴 は基準値から±Nσ 以上の逸脱")
                        _tr_sum_df = pd.DataFrame(_tr_summary_rows)
                        st.dataframe(_tr_sum_df, hide_index=True, use_container_width=True)

                        # CSV ダウンロード
                        _tr_dl_rows = []
                        for _ts2 in _tr_steps:
                            _tsn2  = _ts2["name"]
                            _tsm2  = _ts2["mode"]
                            _dcol2 = (
                                f"{_tsn2}_遅れ[ms]" if _tsm2 == "single"
                                else f"{_tsn2}_dur[ms]"
                            )
                            for _r2 in _tr_res_list:
                                _rdf2 = _r2["result"]
                                if _dcol2 not in _rdf2.columns:
                                    continue
                                _rv2 = _rdf2[_dcol2].dropna().values
                                if len(_rv2) == 0:
                                    continue
                                _tr_dl_rows.append({
                                    "時期":         _r2["label"],
                                    "ステップ":       _tsn2,
                                    "サイクル数":     len(_rv2),
                                    "平均[ms]":      round(float(np.mean(_rv2)), 3),
                                    "標準偏差[ms]":   round(float(np.std(_rv2)), 3),
                                    "最小[ms]":      round(float(np.min(_rv2)), 3),
                                    "最大[ms]":      round(float(np.max(_rv2)), 3),
                                })
                        if _tr_dl_rows:
                            st.download_button(
                                "📥 傾向データCSVダウンロード",
                                pd.DataFrame(_tr_dl_rows).to_csv(
                                    index=False, encoding="utf-8-sig"
                                ),
                                f"{_tr_pname}_trend.csv",
                                key=f"tr_dl_{_tr_pname}",
                                use_container_width=True,
                            )
            else:
                st.info("CSVファイルをアップロードして「傾向解析を実行」を押してください")
