"""
pages/histogram.py – ステップ別ヒストグラム詳細（別タブ表示専用）

ガントチャートのバーをダブルクリックすると新規タブでこのページが開く。
URL パラメータ:
  proc       工程名（表示用）
  trigger    基準変数名
  edge       RISE | FALL
  step_name  ステップ名
  steps      URL-encoded JSON (steps_list)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json, urllib.parse
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from analyzer import (
    load_csv, detect_bool_columns,
    analyze_cycles_v2, detect_cycles, get_cycle_waveforms,
    calc_sturges_bins, normalize_bool_series,
)

st.set_page_config(page_title="ヒストグラム詳細", page_icon="📊", layout="wide")

# サイドバーナビ非表示
st.markdown(
    '<style>[data-testid="stSidebarNav"]{display:none!important}</style>',
    unsafe_allow_html=True,
)

# ── メイン画面に戻るボタン ───────────────────────────────────────
# st.components.v1.html() はサンドボックス iframe 内で動作するため
# window.parent.location.href は allow-top-navigation がなくブロックされる。
# st.link_button はメインフレームで直接レンダリングされるため確実に動作する。
st.link_button("← メイン画面に戻る", url="/")


# ── キャッシュ付きラッパー ────────────────────────────────────────────

@st.cache_data
def _load_sample(path: str):
    df = load_csv(path)
    return df, detect_bool_columns(df)


@st.cache_data
def _analyze(df, trigger: str, edge: str, steps_json: str):
    steps = json.loads(steps_json)
    if not steps:
        return None
    return analyze_cycles_v2(df, trigger, edge, steps)


@st.cache_data
def _get_waveforms(df, trigger: str, edge: str, vars_t: tuple) -> list:
    idx = list(detect_cycles(df, trigger, edge))
    return get_cycle_waveforms(df, idx, list(vars_t))


# ── URL パラメータ読み込み ─────────────────────────────────────────────

params    = st.query_params
proc      = params.get("proc",      "")
trigger   = params.get("trigger",   "")
edge      = params.get("edge",      "RISE")
step_name = params.get("step_name", "")
steps_enc = params.get("steps",     "")

if not (proc and trigger and step_name and steps_enc):
    st.error("パラメータが不足しています。ガントチャートのバーをダブルクリックして開いてください。")
    st.stop()

try:
    steps_list = json.loads(urllib.parse.unquote(steps_enc))
except Exception as e:
    st.error(f"ステップ設定の解析に失敗しました: {e}")
    st.stop()

step = next((s for s in steps_list if s.get("name") == step_name), None)
if step is None:
    st.error(f"ステップ '{step_name}' が見つかりません。steps パラメータを確認してください。")
    st.stop()

# ── データ読み込み ─────────────────────────────────────────────────────

sample_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sample_playback.csv",
)
try:
    df, _ = _load_sample(sample_path)
except FileNotFoundError:
    st.error(f"sample_playback.csv が見つかりません（{sample_path}）")
    st.stop()
except Exception as e:
    st.error(f"データ読み込みエラー: {e}")
    st.stop()

# ── 解析実行 ───────────────────────────────────────────────────────────

steps_json_canonical = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
try:
    with st.spinner("解析中…"):
        result_df = _analyze(df, trigger, edge, steps_json_canonical)
except Exception as e:
    st.error(f"解析エラー: {e}")
    st.stop()

if result_df is None or len(result_df) == 0:
    st.warning("サイクルデータを取得できません。基準変数と edge 設定を確認してください。")
    st.stop()

# ── ページヘッダ ───────────────────────────────────────────────────────

edge_s = "↑" if edge == "RISE" else "↓"
mode   = step.get("mode", "single")
color  = step.get("color", "#4472C4")
icon   = "↔" if mode == "range" else "→"

st.title(f"📊 {proc}　—　{icon} {step_name}")
st.caption(
    f"基準変数: `{trigger}` {edge_s}　｜　"
    f"モード: {'範囲（開始/終了）' if mode == 'range' else '単一変数'}"
)
st.divider()

# ═══════════════════════════════════════════════════════════════
# 単一変数モード
# ═══════════════════════════════════════════════════════════════

if mode == "single":
    var       = step.get("variable", "")
    delay_col = f"{step_name}_遅れ[ms]"

    if delay_col not in result_df.columns:
        st.error(f"データ列 '{delay_col}' が見つかりません。ステップ名・基準変数を確認してください。")
        st.stop()

    delays = result_df[delay_col].dropna().values
    if len(delays) == 0:
        st.error("タイミングデータが空です。")
        st.stop()

    mean_d = float(np.mean(delays))
    std_d  = float(np.std(delays))
    sig3   = mean_d + 3 * std_d
    n      = len(delays)

    # ── 統計カード ────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("N サイクル",   n)
    m2.metric("最小",         f"{float(np.min(delays)):.1f} ms")
    m3.metric("平均",         f"{mean_d:.2f} ms")
    m4.metric("σ",           f"{std_d:.2f} ms")
    m5.metric("3σ 上限",     f"{sig3:.2f} ms")

    # ── 閾値設定 ──────────────────────────────────────────────────
    tc, _ = st.columns([2, 4])
    with tc:
        threshold = st.number_input(
            "閾値 [ms]（0 = なし）",
            min_value=0.0, value=0.0, step=0.5,
            key="threshold",
        )

    h_col, w_col = st.columns(2)

    # ── ヒストグラム ──────────────────────────────────────────────
    with h_col:
        st.markdown("**ヒストグラム**")
        n_bins = calc_sturges_bins(n)
        fig_h  = go.Figure()

        if threshold > 0:
            below = delays[delays <= threshold]
            above = delays[delays >  threshold]
            if len(below):
                fig_h.add_trace(go.Histogram(
                    x=below, nbinsx=n_bins, name="閾値以内",
                    marker_color="royalblue", opacity=0.75))
            if len(above):
                fig_h.add_trace(go.Histogram(
                    x=above, nbinsx=n_bins, name="閾値超過",
                    marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange",
                            annotation_text=f"閾値 {threshold}ms")
        else:
            fig_h.add_trace(go.Histogram(
                x=delays, nbinsx=n_bins, marker_color=color, opacity=0.85))

        fig_h.add_vline(x=mean_d, line_dash="dash", line_color="navy",
                        annotation_text=f"平均 {mean_d:.1f}ms")
        fig_h.add_vline(x=sig3,   line_dash="dot",  line_color="gray",
                        annotation_text=f"3σ {sig3:.1f}ms")
        fig_h.update_layout(
            xaxis_title="遅れ時間 [ms]", yaxis_title="頻度",
            barmode="overlay", height=380,
            margin=dict(t=24, b=44), plot_bgcolor="white",
            showlegend=(threshold > 0),
        )
        st.plotly_chart(fig_h, use_container_width=True, key="hist_main")

        if threshold > 0:
            rate = float(np.mean(delays <= threshold)) * 100
            st.caption(f"閾値達成率: **{rate:.1f}%**")

        # 統計テーブル
        stat_rows = [
            {"指標": "サンプル数",    "値": str(n)},
            {"指標": "最小 [ms]",     "値": f"{float(np.min(delays)):.3f}"},
            {"指標": "最大 [ms]",     "値": f"{float(np.max(delays)):.3f}"},
            {"指標": "平均 [ms]",     "値": f"{mean_d:.3f}"},
            {"指標": "標準偏差 [ms]", "値": f"{std_d:.3f}"},
            {"指標": "3σ 上限 [ms]", "値": f"{sig3:.3f}"},
        ]
        if threshold > 0:
            stat_rows.append({
                "指標": "閾値達成率",
                "値":   f"{float(np.mean(delays <= threshold))*100:.1f}%",
            })
        import pandas as pd
        st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    # ── 波形重ね ──────────────────────────────────────────────────
    with w_col:
        st.markdown("**波形重ね（全サイクル + 平均）**")
        waveforms = []
        if var:
            try:
                waveforms = _get_waveforms(df, trigger, edge, (var,))
            except Exception:
                st.warning("波形データを取得できませんでした")

        if waveforms:
            view_start = max(0.0, mean_d - std_d * 4 - 30)
            view_end   = mean_d + std_d * 5 + 30

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
                fig_w.add_trace(go.Scatter(
                    x=t, y=v, mode="lines",
                    line=dict(color="rgba(100,100,200,0.08)", width=1),
                    showlegend=False))

            if all_t:
                tmin = min(t[0]  for t in all_t)
                tmax = max(t[-1] for t in all_t)
                ct   = np.linspace(tmin, tmax, 300)
                mv   = [
                    np.mean([v[np.searchsorted(t, tp)]
                             for t, v in zip(all_t, all_v)
                             if np.searchsorted(t, tp) < len(v)])
                    for tp in ct
                ]
                fig_w.add_trace(go.Scatter(
                    x=ct, y=mv, mode="lines",
                    line=dict(color=color, width=2.5), name="平均波形"))

            fig_w.add_vline(x=mean_d, line_dash="dash", line_color=color,
                            annotation_text=f"平均 {mean_d:.1f}ms")
            fig_w.update_layout(
                xaxis_title="サイクル開始からの経過時間 [ms]",
                yaxis_title="変数値", height=380,
                margin=dict(t=24, b=44), plot_bgcolor="white",
                showlegend=bool(all_t),
            )
            st.plotly_chart(fig_w, use_container_width=True, key="wave_main")
            st.caption(f"{len(all_t)} サイクル重ね　表示範囲: {view_start:.0f}～{view_end:.0f} ms")

    # ── 時系列 ────────────────────────────────────────────────────
    st.divider()
    st.markdown("**⏱ サイクル別 遅れ時間（時系列）**")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        y=delays, mode="lines+markers",
        marker=dict(size=3, color=color),
        line=dict(color=color, width=1), name=step_name))
    fig_ts.add_hline(y=mean_d, line_dash="dash", line_color="navy",
                     annotation_text=f"平均 {mean_d:.1f}ms")
    fig_ts.add_hline(y=sig3,   line_dash="dot",  line_color="red",
                     annotation_text=f"3σ {sig3:.1f}ms")
    if threshold > 0:
        fig_ts.add_hline(y=threshold, line_dash="dash", line_color="orange",
                         annotation_text=f"閾値 {threshold}ms")
    fig_ts.update_layout(
        xaxis_title="サイクル番号", yaxis_title="遅れ時間 [ms]",
        height=220, margin=dict(t=10, b=40), plot_bgcolor="white",
    )
    st.plotly_chart(fig_ts, use_container_width=True, key="ts_main")

# ═══════════════════════════════════════════════════════════════
# 範囲モード（開始/終了）
# ═══════════════════════════════════════════════════════════════

else:
    dur_col = f"{step_name}_dur[ms]"

    if dur_col not in result_df.columns:
        st.error(f"データ列 '{dur_col}' が見つかりません。")
        st.stop()

    durs = result_df[dur_col].dropna().values
    if len(durs) == 0:
        st.error("タイミングデータが空です。")
        st.stop()

    mean_d = float(np.mean(durs))
    std_d  = float(np.std(durs))
    sig3   = mean_d + 3 * std_d
    n      = len(durs)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("N サイクル",   n)
    m2.metric("最小",         f"{float(np.min(durs)):.1f} ms")
    m3.metric("平均",         f"{mean_d:.2f} ms")
    m4.metric("σ",           f"{std_d:.2f} ms")
    m5.metric("3σ 上限",     f"{sig3:.2f} ms")

    n_bins = calc_sturges_bins(n)
    fig_h  = go.Figure()
    fig_h.add_trace(go.Histogram(x=durs, nbinsx=n_bins, marker_color=color, opacity=0.85))
    fig_h.add_vline(x=mean_d, line_dash="dash", line_color="navy",
                    annotation_text=f"平均 {mean_d:.1f}ms")
    fig_h.add_vline(x=sig3,   line_dash="dot",  line_color="gray",
                    annotation_text=f"3σ {sig3:.1f}ms")
    fig_h.update_layout(
        xaxis_title="継続時間 [ms]", yaxis_title="頻度",
        height=380, margin=dict(t=24, b=44), plot_bgcolor="white",
    )
    st.plotly_chart(fig_h, use_container_width=True, key="hist_range")

    # 統計テーブル
    import pandas as pd
    stat_rows = [
        {"指標": "サンプル数",    "値": str(n)},
        {"指標": "最小 [ms]",     "値": f"{float(np.min(durs)):.3f}"},
        {"指標": "最大 [ms]",     "値": f"{float(np.max(durs)):.3f}"},
        {"指標": "平均 [ms]",     "値": f"{mean_d:.3f}"},
        {"指標": "標準偏差 [ms]", "値": f"{std_d:.3f}"},
        {"指標": "3σ 上限 [ms]", "値": f"{sig3:.3f}"},
    ]
    st.dataframe(pd.DataFrame(stat_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**⏱ サイクル別 継続時間（時系列）**")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        y=durs, mode="lines+markers",
        marker=dict(size=3, color=color),
        line=dict(color=color, width=1), name=step_name))
    fig_ts.add_hline(y=mean_d, line_dash="dash", line_color="navy")
    fig_ts.add_hline(y=sig3,   line_dash="dot",  line_color="red")
    fig_ts.update_layout(
        xaxis_title="サイクル番号", yaxis_title="継続時間 [ms]",
        height=220, margin=dict(t=10, b=40), plot_bgcolor="white",
    )
    st.plotly_chart(fig_ts, use_container_width=True, key="ts_range")
