"""
pages/詳細.py - ステップ別 ヒストグラム詳細・波形重ね
（app.py の「詳細 →」ボタンから遷移）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from analyzer import (
    detect_cycles,
    analyze_cycles,
    get_cycle_waveforms,
    calc_sturges_bins,
    normalize_bool_series,
)

st.set_page_config(page_title="ステップ詳細", page_icon="📈", layout="wide")

# ─── 前提チェック ─────────────────────────────────────────────
if st.session_state.get("_df") is None:
    st.warning("メインページからステップを選択してアクセスしてください。")
    if st.button("← メインへ"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state["_df"]
trigger_col = st.session_state.get("_trigger_col", "")
edge = st.session_state.get("_edge", "RISE")
step_stats = st.session_state.get("step_stats", [])
selected_step = st.session_state.get("selected_step", "")
step_names = [s["name"] for s in step_stats]

# ─── ヘッダ ───────────────────────────────────────────────────
hc1, hc2 = st.columns([1, 6])
with hc1:
    if st.button("← メインに戻る"):
        st.switch_page("app.py")
with hc2:
    st.title(f"📈 {selected_step}（ステップ詳細）")

# ─── 前後ナビゲーション ───────────────────────────────────────
current_idx = step_names.index(selected_step) if selected_step in step_names else 0

nav1, nav2, nav3 = st.columns([1, 6, 1])
with nav1:
    if st.button("◀ 前", disabled=(current_idx == 0)):
        st.session_state["selected_step"] = step_names[current_idx - 1]
        st.rerun()
with nav2:
    new_step = st.selectbox(
        "ステップ選択",
        step_names,
        index=current_idx,
        key="step_nav_select",
        label_visibility="collapsed",
    )
    if new_step != selected_step:
        st.session_state["selected_step"] = new_step
        st.rerun()
with nav3:
    if st.button("次 ▶", disabled=(current_idx >= len(step_names) - 1)):
        st.session_state["selected_step"] = step_names[current_idx + 1]
        st.rerun()

# ─── ステップ情報取得 ─────────────────────────────────────────
current_step = next((s for s in step_stats if s["name"] == selected_step), None)
if not current_step:
    st.warning("ステップ情報が見つかりません。メインページに戻ってください。")
    st.stop()

var = current_step["variable"]

# ─── 解析（毎回実行・比較的軽量）────────────────────────────
with st.spinner(f"{var} を解析中..."):
    result = analyze_cycles(df, trigger_col, edge, [var])
    cycle_starts_idx = detect_cycles(df, trigger_col, edge)
    waveforms = get_cycle_waveforms(df, cycle_starts_idx, [var])

delay_col = f"{var}_遅れ[ms]"
delays = (
    result[delay_col].dropna().values
    if result is not None and delay_col in result.columns
    else np.array([])
)

# ─── サイドバー: 閾値 ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 設定")
    st.markdown(f"**変数**: `{var}`")
    st.markdown(f"**サイクル数**: {len(waveforms)}")

    if len(delays) > 0:
        m_d, s_d = float(np.mean(delays)), float(np.std(delays))
        st.caption(f"推奨閾値（平均 + 3σ）: **{m_d + 3 * s_d:.1f} ms**")
    else:
        m_d, s_d = 0.0, 0.0

    thresh_key = f"thresh_{var}"
    threshold = st.number_input(
        "閾値[ms]（0 = 設定なし）",
        min_value=0.0,
        value=float(st.session_state.get(thresh_key, 0.0)),
        step=0.5,
        key=f"thresh_input_{var}",
    )
    st.session_state[thresh_key] = threshold

    st.markdown("---")
    if len(delays) > 0:
        st.metric("平均", f"{m_d:.2f} ms")
        st.metric("標準偏差", f"{s_d:.2f} ms")
        st.metric("3σ上限", f"{m_d + 3 * s_d:.2f} ms")
        if threshold > 0:
            rate = float(np.mean(delays <= threshold)) * 100
            st.metric("閾値達成率", f"{rate:.1f}%")

# ─── メインコンテンツ ─────────────────────────────────────────
hist_col, wave_col = st.columns(2)

# ── ヒストグラム ─────────────────────────────────────────────
with hist_col:
    st.subheader("ヒストグラム")

    if len(delays) == 0:
        st.warning("この変数の検出データがありません。")
    else:
        n_bins = calc_sturges_bins(len(delays))
        sig3 = m_d + 3 * s_d

        fig_h = go.Figure()
        if threshold > 0:
            below = [d for d in delays if d <= threshold]
            above = [d for d in delays if d > threshold]
            if below:
                fig_h.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内", marker_color="royalblue", opacity=0.75))
            if above:
                fig_h.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過", marker_color="crimson", opacity=0.75))
            fig_h.add_vline(x=threshold, line_dash="dash", line_color="orange", annotation_text=f"閾値 {threshold}ms")
        else:
            fig_h.add_trace(go.Histogram(x=delays, nbinsx=n_bins, marker_color="steelblue", opacity=0.8))

        fig_h.add_vline(x=sig3, line_dash="dot", line_color="gray", annotation_text=f"3σ {sig3:.1f}ms")
        fig_h.update_layout(
            xaxis_title="遅れ時間[ms]",
            yaxis_title="頻度",
            barmode="overlay",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_h, width="stretch")

        # 統計テーブル
        stat_rows = [
            {"指標": "サンプル数", "値": str(len(delays))},
            {"指標": "最小[ms]", "値": f"{np.min(delays):.3f}"},
            {"指標": "最大[ms]", "値": f"{np.max(delays):.3f}"},
            {"指標": "平均[ms]", "値": f"{m_d:.3f}"},
            {"指標": "標準偏差[ms]", "値": f"{s_d:.3f}"},
            {"指標": "3σ上限[ms]", "値": f"{sig3:.3f}"},
        ]
        if threshold > 0:
            stat_rows.append({"指標": "閾値達成率", "値": f"{np.mean(delays <= threshold)*100:.1f}%"})

        st.dataframe(pd.DataFrame(stat_rows), hide_index=True, width="stretch")

        with st.expander("サイクル別 遅れ時間一覧"):
            if result is not None:
                st.dataframe(
                    result[["サイクル#", "開始時刻", delay_col]],
                    width="stretch",
                    hide_index=True,
                )

# ── 波形重ね ─────────────────────────────────────────────────
with wave_col:
    st.subheader("波形重ね表示（全サイクル + 平均）")

    if not waveforms:
        st.warning("波形データがありません。")
    else:
        step_start_ms = current_step.get("start", 0.0)
        step_mean_ms = current_step.get("mean", 30.0)
        view_start = max(0.0, step_start_ms - step_mean_ms * 0.5)
        view_end = step_start_ms + step_mean_ms * 2.5

        fig_w = go.Figure()
        all_t, all_v = [], []

        for cyc in waveforms:
            mask = (cyc["time_offset_ms"] >= view_start) & (cyc["time_offset_ms"] <= view_end)
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = normalize_bool_series(sl[var]).values
            all_t.append(t)
            all_v.append(v)
            fig_w.add_trace(
                go.Scatter(
                    x=t, y=v, mode="lines",
                    line=dict(color="rgba(100,100,200,0.10)", width=1),
                    showlegend=False,
                )
            )

        # 平均波形
        if all_t:
            tmin = min(t[0] for t in all_t if len(t) > 0)
            tmax = max(t[-1] for t in all_t if len(t) > 0)
            ct = np.linspace(tmin, tmax, 300)
            mv = []
            for tp in ct:
                vs = [v[np.searchsorted(t, tp)] for t, v in zip(all_t, all_v) if np.searchsorted(t, tp) < len(v)]
                mv.append(np.mean(vs) if vs else 0)
            fig_w.add_trace(
                go.Scatter(x=ct, y=mv, mode="lines", line=dict(color="royalblue", width=3), name="平均波形")
            )

        # ステップ境界
        fig_w.add_vline(x=step_start_ms, line_dash="dash", line_color="green", annotation_text="開始")
        if step_start_ms + step_mean_ms <= view_end:
            fig_w.add_vline(x=step_start_ms + step_mean_ms, line_dash="dash", line_color="orange", annotation_text="終了（平均）")

        fig_w.update_layout(
            xaxis_title="サイクル開始からの経過時間 [ms]",
            yaxis_title="変数値",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_w, width="stretch")
        st.caption(f"{len(all_t)} サイクル重ね表示　表示範囲: {view_start:.1f} ～ {view_end:.1f} ms")
