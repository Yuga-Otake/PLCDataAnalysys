"""
5_histogram.py — ヒストグラム詳細ページ
ガントチャートのダブルクリックで遷移してくる専用ページ
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="ヒストグラム詳細",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── クエリパラメータ取得 ──────────────────────────────────────────
params    = st.query_params
proc      = params.get("proc", "")
step_name = params.get("step_name", "")

# ── セッション状態からコンテキスト取得 ───────────────────────────
ctx        = st.session_state.get(f"_hist_ctx_{proc}", {})
result_df  = ctx.get("result_df")
step_stats = ctx.get("step_stats", [])
steps_list = ctx.get("steps_list", [])
baseline   = ctx.get("baseline", {})

step_stat = next((s for s in step_stats if s["name"] == step_name), None)
step_cfg  = next((s for s in steps_list if s.get("name") == step_name), None)

# ── ヘッダー ─────────────────────────────────────────────────────
col_back, col_title = st.columns([1, 6])
with col_back:
    st.page_link("app.py", label="← メイン画面", icon="↩️")
with col_title:
    if step_name:
        st.title(f"📊 {step_name}")
        st.caption(f"工程: {proc}" if proc else "")
    else:
        st.title("📊 ヒストグラム詳細")

st.divider()

# ── データ不在の場合 ──────────────────────────────────────────────
if result_df is None or step_stat is None or step_cfg is None:
    st.warning(
        "データが見つかりません。\n\n"
        "メイン画面でCSVを読み込み → 解析実行 → ガントチャートのバーをダブルクリックして遷移してください。"
    )
    st.page_link("app.py", label="🏠 メイン画面に戻る")
    st.stop()

# ── データ準備 ────────────────────────────────────────────────────
mode = step_cfg.get("mode", "single")

if mode == "single":
    col     = f"{step_name}_遅れ[ms]"
    xlabel  = "遅れ時間 [ms]"
elif mode in ("range", "numeric"):
    col     = f"{step_name}_dur[ms]"
    xlabel  = "所要時間 [ms]"
else:
    col     = f"{step_name}_遅れ[ms]"
    xlabel  = "遅れ時間 [ms]"

if col not in result_df.columns:
    st.error(f"列 `{col}` が見つかりません（解析結果を確認してください）")
    st.stop()

raw_vals = result_df[col].dropna().values
if len(raw_vals) == 0:
    st.warning("有効なデータがありません")
    st.stop()

# ── 基準値情報 ────────────────────────────────────────────────────
_bl_entry   = baseline.get(step_name, {})
if mode == "single":
    _bl_ref = _bl_entry.get("ref_ms")
    _bl_std = float(_bl_entry.get("std_ms", 0.0))
else:
    _bl_ref = _bl_entry.get("ref_dur_ms")
    _bl_std = float(_bl_entry.get("std_dur_ms", 0.0))

_delta_mode = _bl_ref is not None
vals_plot   = (raw_vals - _bl_ref) if _delta_mode else raw_vals.copy()

mean_raw = float(np.mean(raw_vals))
std_raw  = float(np.std(raw_vals))
sig3     = mean_raw + 3 * std_raw

# ── ビン数の自動計算（Freedman-Diaconis 法）─────────────────────
def _auto_bins(data: np.ndarray) -> int:
    n   = len(data)
    iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
    rng = float(np.max(data) - np.min(data)) if n > 1 else 1.0
    if iqr > 0 and n > 1:
        bw = 2.0 * iqr / (n ** (1 / 3))
        # 切りのいい値に丸める
        import math
        mag = 10 ** math.floor(math.log10(bw))
        for nw in [mag, mag*2, mag*2.5, mag*5, mag*10]:
            if nw >= bw * 0.8:
                bw = nw
                break
        return max(5, min(80, int(math.ceil(rng / bw))))
    return max(5, min(50, int(1 + np.log2(n))))

_auto = _auto_bins(vals_plot)

# ── コントロールパネル ────────────────────────────────────────────
st.subheader("⚙️ 表示設定")
ctrl1, ctrl2, ctrl3 = st.columns([3, 3, 3])

with ctrl1:
    n_bins = st.slider(
        "📐 ビン数",
        min_value=3, max_value=100, value=_auto, step=1,
        help=f"Freedman-Diaconis による自動算出: {_auto} ビン",
        key="hist_bins",
    )

with ctrl2:
    threshold = st.number_input(
        "📏 閾値 [ms]（0 = なし）",
        min_value=0.0, value=0.0, step=0.5,
        help="この値を超えたサンプルを赤で強調表示",
        key="hist_thresh",
    )

with ctrl3:
    if _delta_mode:
        st.metric("基準値", f"{_bl_ref:.2f} ms",
                  help="この値を基準とした差分を表示中")
        st.metric("基準σ", f"{_bl_std:.2f} ms")
    else:
        st.metric("平均 + 3σ（推奨閾値）", f"{sig3:.2f} ms")

st.divider()

# ── ヒストグラム描画 ──────────────────────────────────────────────
fig = go.Figure()

if threshold > 0 and not _delta_mode:
    below = vals_plot[vals_plot <= threshold]
    above = vals_plot[vals_plot >  threshold]
    if len(below):
        fig.add_trace(go.Histogram(x=below, nbinsx=n_bins, name="閾値以内",
                                   marker_color="royalblue", opacity=0.78))
    if len(above):
        fig.add_trace(go.Histogram(x=above, nbinsx=n_bins, name="閾値超過",
                                   marker_color="crimson",   opacity=0.78))
    fig.add_vline(x=threshold, line_dash="dash", line_color="orange",
                  annotation_text=f"閾値 {threshold:.1f} ms",
                  annotation_position="top right")

elif _delta_mode and _bl_std > 0:
    _t3   = 3 * _bl_std
    in_r  = vals_plot[np.abs(vals_plot) <= _t3]
    out_r = vals_plot[np.abs(vals_plot) >  _t3]
    if len(in_r):
        fig.add_trace(go.Histogram(x=in_r, nbinsx=n_bins, name="±3σ以内",
                                   marker_color="royalblue", opacity=0.78))
    if len(out_r):
        fig.add_trace(go.Histogram(x=out_r, nbinsx=n_bins, name="±3σ超過",
                                   marker_color="crimson",   opacity=0.78))
    fig.add_vline(x= _t3, line_dash="dash", line_color="orange",
                  annotation_text=f"+3σ ({_t3:.1f} ms)", annotation_position="top right")
    fig.add_vline(x=-_t3, line_dash="dash", line_color="orange",
                  annotation_text=f"-3σ ({_t3:.1f} ms)", annotation_position="top left")

else:
    fig.add_trace(go.Histogram(x=vals_plot, nbinsx=n_bins,
                               marker_color="steelblue", opacity=0.82))

# 基準線
if _delta_mode:
    fig.add_vline(x=0, line_color="green", line_width=2,
                  annotation_text=f"基準値 {_bl_ref:.1f} ms",
                  annotation_position="top left")
    xlabel_final = f"基準値からのずれ [ms]"
else:
    fig.add_vline(x=sig3, line_dash="dot", line_color="gray",
                  annotation_text=f"平均+3σ = {sig3:.1f} ms",
                  annotation_position="top right")
    if threshold > 0:
        pass  # already added
    xlabel_final = xlabel

fig.update_layout(
    title=dict(
        text=f"<b>{step_name}</b>　ヒストグラム（工程: {proc}）",
        font=dict(size=18),
    ),
    xaxis_title=xlabel_final,
    yaxis_title="頻度（サイクル数）",
    barmode="overlay",
    height=500,
    margin=dict(t=70, b=60, l=60, r=40),
    showlegend=True,
    legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch", key="hist_main")

# ── 統計サマリー ──────────────────────────────────────────────────
st.divider()
st.subheader("📈 統計サマリー")

mc = st.columns(6)
mc[0].metric("サンプル数",  f"{len(raw_vals)} cyc")
mc[1].metric("平均",        f"{mean_raw:.2f} ms")
mc[2].metric("σ",          f"{std_raw:.2f} ms")
mc[3].metric("最小",        f"{float(np.min(raw_vals)):.2f} ms")
mc[4].metric("最大",        f"{float(np.max(raw_vals)):.2f} ms")
mc[5].metric("平均+3σ",    f"{sig3:.2f} ms")

if threshold > 0 and not _delta_mode:
    ok_cnt  = int(np.sum(raw_vals <= threshold))
    ng_cnt  = len(raw_vals) - ok_cnt
    rate    = ok_cnt / len(raw_vals) * 100
    col_ok, col_ng = st.columns(2)
    col_ok.success(f"✅ 閾値以内: **{ok_cnt}** サイクル（{rate:.1f}%）")
    col_ng.error(  f"❌ 閾値超過: **{ng_cnt}** サイクル（{100-rate:.1f}%）")

if _delta_mode and _bl_std > 0:
    ng_cnt = int(np.sum(np.abs(vals_plot) > 3 * _bl_std))
    ok_cnt = len(raw_vals) - ng_cnt
    rate   = ok_cnt / len(raw_vals) * 100
    col_ok, col_ng = st.columns(2)
    col_ok.success(f"✅ ±3σ以内: **{ok_cnt}** サイクル（{rate:.1f}%）")
    col_ng.error(  f"❌ ±3σ超過: **{ng_cnt}** サイクル（{100-rate:.1f}%）")

# ── サイクル別データテーブル ──────────────────────────────────────
st.divider()
with st.expander("📋 サイクル別データを表示", expanded=False):
    show_df = result_df[[col]].copy()
    show_df.index = range(1, len(show_df) + 1)
    show_df.index.name = "サイクル番号"
    show_df.columns = [f"{step_name} [{xlabel}]"]
    st.dataframe(show_df, width="stretch")
    st.download_button(
        "📥 CSVダウンロード",
        show_df.to_csv(encoding="utf-8-sig"),
        file_name=f"{proc}_{step_name}_histogram.csv",
        width="stretch",
    )
