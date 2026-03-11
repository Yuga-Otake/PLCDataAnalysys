"""
app.py - APB タイミング解析ツール v6
・テキスト検索＋予測候補からステップ変数を追加
・単一変数モード / 開始-終了範囲モード
・ガントクリックでステップ詳細連動
・工程ごと異常比較インライン
"""
import os, json, hashlib, urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
            interval = max(0.0, mean_d - prev_mean)
            step_stats.append(dict(
                name=name, color=color, mode="single",
                start=prev_mean, mean=interval,
                min=max(0.0, min_d - prev_mean), max=max(0.0, max_d - prev_mean),
                abs_mean=mean_d, abs_std=std_d, abs_min=min_d, abs_max=max_d,
            ))
            prev_mean = mean_d

        else:  # range
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
                name=name, color=color, mode="range",
                start=mean_s, mean=mean_d,
                min=min_d, max=max_d,
                abs_mean=mean_s + mean_d, abs_std=std_d,
                abs_min=mean_s + min_d, abs_max=mean_s + max_d,
                abs_start=mean_s,
            ))
            prev_mean = mean_s + mean_d

    if not step_stats:
        return None, []

    total = sum(s["mean"] for s in step_stats) or 1.0
    fig   = go.Figure()

    for s in step_stats:
        pct      = round(s["mean"] / total * 100, 1)
        is_range = (s["mode"] == "range")
        if is_range:
            ht = (
                "<b>%{y}</b> [範囲]<br>"
                f"開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "長さ（平均）: %{customdata[1]:.1f} ms<br>"
                "長さばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        else:
            ht = (
                "<b>%{y}</b><br>"
                "区間長（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )

        fig.add_trace(go.Bar(
            name=s["name"], y=[s["name"]], x=[s["mean"]], base=[s["start"]],
            orientation="h", width=0.5, marker_color=s["color"],
            marker_pattern_shape="/" if is_range else "",
            customdata=[[s["start"], s["mean"], s["min"], s["max"], pct]],
            hovertemplate=ht,
            text=[f"{pct:.0f}%"],
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
# ステップ詳細
# ═══════════════════════════════════════════════════════════════

def render_step_detail(df: pd.DataFrame, trigger_col: str, edge: str,
                       step_stat: dict, step: dict, pname: str,
                       result_df: pd.DataFrame):
    mode = step.get("mode", "single")
    name = step_stat["name"]

    if mode == "single":
        _render_single_detail(df, trigger_col, edge, step_stat, step, pname, result_df)
    else:
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
        st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        st.markdown("**ヒストグラム**")
        n_bins = calc_sturges_bins(len(delays))
        fig_h  = go.Figure()
        if threshold > 0:
            below = [d for d in delays if d <= threshold]
            above = [d for d in delays if d > threshold]
            if below:
                fig_h.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内",
                                             marker_color="royalblue", opacity=0.75))
            if above:
                fig_h.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過",
                                             marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"閾値 {threshold}ms")
        else:
            fig_h.add_trace(go.Histogram(x=delays, nbinsx=n_bins,
                                         marker_color="steelblue", opacity=0.8))
        fig_h.add_vline(x=sig3, line_dash="dot", line_color="gray",
                        annotation_text=f"3σ {sig3:.1f}ms")
        fig_h.update_layout(xaxis_title="遅れ時間[ms]", yaxis_title="頻度",
                             barmode="overlay", height=260, margin=dict(t=8, b=32),
                             showlegend=threshold > 0)
        st.plotly_chart(fig_h, use_container_width=True, key=f"hist_{pname}_{name}")
        sc = st.columns(4)
        sc[0].metric("N",      len(delays))
        sc[1].metric("平均",   f"{mean_d:.1f}ms")
        sc[2].metric("σ",     f"{std_d:.1f}ms")
        sc[3].metric("3σ上限", f"{sig3:.1f}ms")
        if threshold > 0:
            rate = np.mean(delays <= threshold) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")

    with w_col:
        st.markdown("**波形重ね（全サイクル＋平均）**")
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, (var,))
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        step_start = step_stat.get("abs_mean", 0.0) - step_stat.get("mean", 30.0)
        step_mean  = step_stat.get("mean", 30.0)
        view_start = max(0.0, step_start - step_mean * 0.5)
        view_end   = step_start + step_mean * 2.5

        fig_w   = go.Figure()
        all_t, all_v = [], []
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = normalize_bool_series(sl[var]).values
            all_t.append(t); all_v.append(v)
            fig_w.add_trace(go.Scatter(x=t, y=v, mode="lines",
                                       line=dict(color="rgba(100,100,200,0.10)", width=1),
                                       showlegend=False))
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
    name      = step_stat["name"]
    start_var = step.get("start_var", "")
    end_var   = step.get("end_var", "")
    start_col = f"{name}_start[ms]"
    dur_col   = f"{name}_dur[ms]"

    starts = (result_df[start_col].dropna().values
              if start_col in result_df.columns else np.array([]))
    durs   = (result_df[dur_col].dropna().values
              if dur_col in result_df.columns else np.array([]))

    if len(durs) == 0:
        st.warning(f"範囲データが取得できませんでした（開始: {start_var} / 終了: {end_var}）")
        return

    mean_s, mean_d = (float(np.mean(starts)) if len(starts) > 0 else 0), float(np.mean(durs))
    std_d = float(np.std(durs))
    sig3  = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}_dur"

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
        st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        st.markdown("**所要時間ヒストグラム**")
        n_bins = calc_sturges_bins(len(durs))
        fig_h  = go.Figure()
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
        fig_h.update_layout(xaxis_title="所要時間[ms]", yaxis_title="頻度",
                             barmode="overlay", height=260, margin=dict(t=8, b=32),
                             showlegend=threshold > 0)
        st.plotly_chart(fig_h, use_container_width=True, key=f"hist_{pname}_{name}")
        sc = st.columns(4)
        sc[0].metric("N",       len(durs))
        sc[1].metric("平均所要", f"{mean_d:.1f}ms")
        sc[2].metric("σ",      f"{std_d:.1f}ms")
        sc[3].metric("3σ上限",  f"{sig3:.1f}ms")
        st.caption(f"開始タイミング平均: **{mean_s:.1f} ms**")
        if threshold > 0:
            rate = np.mean(durs <= threshold) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")

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

        fig_w = go.Figure()
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
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
                steps.append({
                    "name":     v,
                    "color":    _default_color(len(steps)),
                    "mode":     "single",
                    "variable": v,
                    "edge":     "RISE",
                })
            st.session_state[pk(pname, "steps_list")] = steps
            st.session_state[f"_clear_srch_{pname}"] = True
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
            st.rerun()

    # モード切替
    new_mode_lbl = st.radio(
        "モード", ["単一変数", "開始/終了"],
        index=0 if mode == "single" else 1, horizontal=True,
        key=f"_emode_{pname}_{step_idx}",
    )
    new_mode = "single" if new_mode_lbl == "単一変数" else "range"

    # 変数設定
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
    else:
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
            except Exception as e:
                st.error(f"読み込みエラー: {e}")
                st.stop()
        else:
            st.info("CSVをアップロードしてください")
            st.stop()

    bool_cols = [c for c, t in col_types.items() if t == "bool"]
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]

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

    st.markdown("**⏱ 工程タイムライン概要**")
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
                    st.markdown(
                        f'<span id="{safe_id}"></span>'
                        f'<style>'
                        f'[data-testid="stElementContainer"]:has(#{safe_id}) + '
                        f'[data-testid="stElementContainer"] button'
                        f' {{ border-left: 5px solid {clr} !important;'
                        f'    padding-left: 10px !important;'
                        f'    background: linear-gradient(90deg,{clr}28 0%,transparent 55%) !important;}}'
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
                            st.rerun()

                if fig_gantt:
                    # ── ヒストグラム URL マップ（ダブルクリック別タブ用）──
                    _steps_enc = urllib.parse.quote(
                        json.dumps(steps_list, ensure_ascii=False, sort_keys=True))
                    _hist_urls = {
                        s["name"]: (
                            f"/histogram"
                            f"?proc={urllib.parse.quote(pname)}"
                            f"&trigger={urllib.parse.quote(trigger_col)}"
                            f"&edge={edge}"
                            f"&step_name={urllib.parse.quote(s['name'])}"
                            f"&steps={_steps_enc}"
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
  var lastT=0,lastN="";
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
      if(now-lastT<450&&name===lastN&&URLS[name]){{
        try{{window.parent.open(URLS[name],'_blank');}}
        catch(e){{window.open(URLS[name],'_blank');}}
      }}
      lastT=now; lastN=name;
    }});
  }}
  attach();
  setTimeout(attach,400);
  setTimeout(attach,1500);
}})();
</script>""", height=0)

                    # ── ステップ統計テーブル ──────────────────────
                    total_t = sum(s["mean"] for s in step_stats) or 1.0
                    rows = []
                    for s in step_stats:
                        row = {
                            "ステップ名": s["name"],
                            "モード":    "範囲" if s["mode"] == "range" else "単一",
                            "平均[ms]":  round(s["mean"], 2),
                            "σ[ms]":    round(s["abs_std"], 2),
                            "min[ms]":  round(s.get("abs_min", s["min"]), 2),
                            "max[ms]":  round(s.get("abs_max", s["max"]), 2),
                            "タクト比率": f"{s['mean']/total_t*100:.1f}%",
                        }
                        rows.append(row)
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True)

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

        # ── 異常比較セクション ────────────────────────────────
        if steps_list:
            st.divider()
            ng_key = f"_ng_df_{pname}"
            comp_hdr, comp_clr = st.columns([5, 1])
            with comp_hdr:
                st.markdown("**🔴 異常比較**")
            with comp_clr:
                if ng_key in st.session_state:
                    if st.button("✕ クリア", key=f"clear_ng_{pname}",
                                 use_container_width=True):
                        del st.session_state[ng_key]
                        st.rerun()

            ng_file = st.file_uploader(
                "NGデータCSVをここにドロップ", type=["csv"],
                key=f"ng_up_{pname}", label_visibility="collapsed",
            )
            if ng_file:
                try:
                    ng_df_loaded = load_csv(ng_file)
                    st.session_state[ng_key] = ng_df_loaded
                    st.rerun()
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

            if ng_key in st.session_state:
                ng_df = st.session_state[ng_key]
                st.success(f"✓ NGデータ {len(ng_df):,}行　vs　正常 {len(df):,}行")
                all_vars = steps_all_vars(steps_list, bool_cols)

                try:
                    nr, ar = compare_normal_abnormal(
                        df, ng_df, trigger_col, edge, all_vars
                    )
                    anoms = detect_anomalous_variables(nr, ar, all_vars)
                    if anoms:
                        anom_labels = [
                            f"**{a['variable']}** ({a['exceed_rate']*100:.0f}%超過)"
                            for a in anoms
                        ]
                        st.warning(f"⚠️ 異常候補: {' ／ '.join(anom_labels)}")
                    else:
                        st.info("異常候補変数は検出されませんでした")

                    ctabs = st.tabs(["📊 重ねヒストグラム", "📈 ずれランキング",
                                     "〰️ 波形重ね比較"])

                    with ctabs[0]:
                        st.caption("X軸は**正常データの平均値を0**としたずれ量。"
                                   "異常時にどちらへ何ms外れているかが一目でわかります。")
                        for col in all_vars:
                            dcol = f"{col}_遅れ[ms]"
                            if dcol not in nr.columns:
                                continue
                            nd = nr[dcol].dropna().values
                            ad = ar[dcol].dropna().values
                            if len(nd) == 0:
                                continue

                            nm = float(np.mean(nd))
                            ns = float(np.std(nd))

                            # 正常平均を原点に変換
                            nd_d = nd - nm
                            ad_d = ad - nm

                            nb  = calc_sturges_bins(max(len(nd_d), len(ad_d), 1))
                            fig = go.Figure()
                            if len(nd_d):
                                fig.add_trace(go.Histogram(
                                    x=nd_d, name="正常", opacity=0.65,
                                    marker_color="royalblue", nbinsx=nb))
                            if len(ad_d):
                                fig.add_trace(go.Histogram(
                                    x=ad_d, name="異常", opacity=0.65,
                                    marker_color="crimson", nbinsx=nb))

                            # 原点（正常平均）
                            fig.add_vline(x=0, line_dash="solid", line_color="green",
                                          line_width=2,
                                          annotation_text=f"正常平均 {nm:.1f}ms",
                                          annotation_position="top right")
                            # ±3σ / ±4σ
                            if ns > 0:
                                for sig, dash, col_v in [
                                    (3, "dash",  "orange"),
                                    (4, "dot",   "red"),
                                ]:
                                    for sign in [1, -1]:
                                        v = sign * sig * ns
                                        lbl = f"+{sig}σ" if sign > 0 else f"-{sig}σ"
                                        fig.add_vline(
                                            x=v, line_dash=dash, line_color=col_v,
                                            annotation_text=f"{lbl} ({v:+.1f}ms)",
                                            annotation_position=(
                                                "top right" if sign > 0 else "top left"),
                                        )

                            fig.update_layout(
                                barmode="overlay",
                                title=f"{col}　正常平均: {nm:.1f}ms　σ: {ns:.1f}ms",
                                xaxis_title="正常平均からのずれ [ms]",
                                height=260, margin=dict(t=36, b=28),
                                showlegend=True,
                            )
                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"cmp_hist_{pname}_{col}")

                            # 異常サンプルの超過率サマリー
                            if len(ad_d) > 0 and ns > 0:
                                e3 = np.mean(np.abs(ad_d) > 3 * ns) * 100
                                e4 = np.mean(np.abs(ad_d) > 4 * ns) * 100
                                st.caption(
                                    f"異常データの **3σ超過率: {e3:.1f}%** ／ "
                                    f"**4σ超過率: {e4:.1f}%**"
                                )
                            st.markdown("---")

                    with ctabs[1]:
                        ddf = calc_diff_ranking(nr, ar, all_vars)
                        if len(ddf) > 0:
                            fig = px.bar(ddf, x="変数名", y="遅れ差分[ms]",
                                         color="遅れ差分[ms]",
                                         color_continuous_scale=["royalblue", "yellow", "crimson"])
                            fig.update_layout(height=280)
                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"cmp_rank_{pname}")
                            st.dataframe(ddf, use_container_width=True, hide_index=True)

                    with ctabs[2]:
                        wv = st.selectbox("比較する変数", all_vars, key=f"cmp_wv_{pname}")
                        if wv:
                            ncs = pd.Index(cached_detect_cycles(df, trigger_col, edge))
                            acs = pd.Index(cached_detect_cycles(ng_df, trigger_col, edge))
                            nw  = get_cycle_waveforms(df,    ncs, [wv])
                            aw  = get_cycle_waveforms(ng_df, acs, [wv])
                            fig = go.Figure()
                            for c in nw:
                                fig.add_trace(go.Scatter(
                                    x=c["time_offset_ms"], y=normalize_bool_series(c[wv]),
                                    mode="lines", line=dict(color="rgba(65,105,225,0.15)"),
                                    showlegend=False))
                            for c in aw:
                                fig.add_trace(go.Scatter(
                                    x=c["time_offset_ms"], y=normalize_bool_series(c[wv]),
                                    mode="lines", line=dict(color="rgba(220,20,60,0.15)"),
                                    showlegend=False))
                            nt, nm  = mean_waveform(nw, wv)
                            at_, am = mean_waveform(aw, wv)
                            if nm:
                                fig.add_trace(go.Scatter(x=nt, y=nm, mode="lines",
                                                         line=dict(color="royalblue", width=3),
                                                         name="正常 平均"))
                            if am:
                                fig.add_trace(go.Scatter(x=at_, y=am, mode="lines",
                                                         line=dict(color="crimson", width=3),
                                                         name="異常 平均"))
                            fig.update_layout(title=f"{wv} 波形重ね比較",
                                              xaxis_title="経過時間[ms]", height=360)
                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"cmp_wave_{pname}")

                except Exception as e:
                    st.error(f"比較エラー: {e}")
            else:
                st.caption("↑ NGデータCSVをアップロードすると即座に比較が始まります")

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

                tabs = st.tabs(["サイクル一覧", "ヒストグラム", "時系列波形"])

                with tabs[0]:
                    st.dataframe(result_df, use_container_width=True)
                    st.download_button("CSVダウンロード",
                                       result_df.to_csv(index=False, encoding="utf-8-sig"),
                                       f"{pname}_cycles.csv", key=f"dl_csv_{pname}")

                with tabs[1]:
                    for col in delay_cols:
                        vn = col.replace("_遅れ[ms]", "").replace("_dur[ms]", " (所要時間)")
                        dl = result_df[col].dropna().values
                        if len(dl) == 0:
                            continue
                        nb  = calc_sturges_bins(len(dl))
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

    st.markdown("")

# ═══════════════════════════════════════════════════════════════
# 工程間タイムライン比較（ボトルネック分析用）
# ═══════════════════════════════════════════════════════════════

if len(processes) >= 1:
    st.divider()
    with st.expander("📊 工程間タイムライン比較（ボトルネック分析）", expanded=False):
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
