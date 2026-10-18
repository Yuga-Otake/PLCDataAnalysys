"""
views/evaluate.py — 最新データの評価

比較データ 1 件を基準値と突き合わせ、工程ごとの状態と、外れの大きいステップを示す。
ステップの行を選ぶとステップ詳細（views/step.py）へ移る。
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.analysis import MODE_LABELS
from core.data import df_summary
from core.evaluate import STATUS, compare_keys_by_time, evaluate_process
from core.state import pk
from core.ui import STATUS_COLOR, chip, numeric, page_header, rule_popover, status_cell_style

ss = st.session_state
page_header(
    "最新データの評価",
    "比較データを基準値と比べ、遅れやばらつきが出ている工程・ステップを示します。行を選ぶとステップの詳細を開きます。",
    eyebrow="点検",
)

store = ss.get("csv_store", {})
processes = ss.get("processes", {})
ready = [p for p in processes if ss.get(pk(p, "baseline"))]
cmp_keys = compare_keys_by_time()
if not ready:
    st.info("基準値が登録された工程がありません。先に基準値を登録してください。")
    if st.button("基準値の登録へ", type="primary"):
        st.switch_page("views/baseline.py")
    st.stop()
if not cmp_keys:
    st.info("比較データがありません。点検したい時期の CSV をデータ画面で追加してください。")
    if st.button("データ画面へ", type="primary"):
        st.switch_page("views/data.py")
    st.stop()

# ── 対象データ ───────────────────────────────────────────────
if ss.get("eval_target") not in cmp_keys:
    ss["eval_target"] = cmp_keys[-1]
c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
target = c1.selectbox("評価するデータ", cmp_keys, key="eval_target",
                      format_func=lambda k: store[k]["label"] + ("（最新）" if k == cmp_keys[-1] else ""))
with c2:
    rule_popover()
entry = store[target]
info = df_summary(entry)
st.markdown(f"<div class='apb-kv'><span>期間 <b>{info['period']}</b></span>"
            f"<span>行数 <b class='apb-num'>{info['rows']:,}</b></span></div>", unsafe_allow_html=True)

results = {p: evaluate_process(entry["df"], p) for p in ready}

# ── 工程ごとの状態 ───────────────────────────────────────────
st.subheader("工程ごとの状態", anchor=False)
cols = st.columns(min(3, len(results)))
for i, (p, ev) in enumerate(results.items()):
    label, kind, _ = STATUS[ev["status"]]
    with cols[i % len(cols)].container(border=True):
        st.markdown(f"**{p}**　{chip(label, kind)}", unsafe_allow_html=True)
        takt_txt = ""
        if ev["takt"] and ev["last_end"] is not None:
            over = ev["last_end"] > ev["takt"]
            takt_txt = (f"<span>最終ステップ終了 <b class='apb-num'>{ev['last_end']:.0f}</b> / "
                        f"タクト目標 {ev['takt']} ms{'（超過）' if over else ''}</span>")
        st.markdown(
            f"<div class='apb-kv'><span>サイクル <b class='apb-num'>{ev['n_cycles']}</b></span>"
            f"<span>平均周期 <b class='apb-num'>{ev['cycle_ms']:.0f} ms</b></span>"
            f"<span>異常 <b class='apb-num'>{ev['n_ng_steps']}</b>・注意 <b class='apb-num'>{ev['n_warn_steps']}</b>"
            f" ステップ</span>{takt_txt}</div>", unsafe_allow_html=True)

# ── 工程の詳細 ───────────────────────────────────────────────
st.subheader("ステップの判定", anchor=False)
if ss.get("sel_process") not in ready:
    ss["sel_process"] = max(ready, key=lambda p: STATUS[results[p]["status"]][2])
pname = st.selectbox("工程", ready, key="sel_process",
                     format_func=lambda p: f"{p}（{STATUS[results[p]['status']][0]}）")
ev = results[pname]
if 0 < ev["n_base"] < 5:
    st.markdown(chip("基準のサイクル数が少ない", "warn") +
                f"　基準値は {ev['n_base']} サイクルから求めています。許容幅が広くなり、小さな変化を見逃しやすくなります。"
                "正常時のデータを加えて 10 サイクル以上で登録し直すことを勧めます。", unsafe_allow_html=True)
if ev["n_cycles"] == 0:
    st.warning("このデータでは工程のサイクルを検出できませんでした。サイクル開始信号を確認してください。")
    st.stop()

# タイミング比較図: 基準（灰色）と今回（判定色）
steps = ev["steps"]
fig = go.Figure()
names = [s["name"] for s in steps]
for s in steps:
    if s["mean"] is None:
        continue
    color = STATUS_COLOR[s["status"]]
    if s["mode"] == "single":
        if s["ref"] is not None:
            fig.add_trace(go.Scatter(x=[s["ref"]], y=[s["name"]], mode="markers", showlegend=False,
                                     marker=dict(symbol="line-ns", size=18, line=dict(width=2, color="#8b949e")),
                                     hovertemplate=f"基準 {s['ref']:.1f} ms<extra></extra>"))
        fig.add_trace(go.Scatter(x=[s["mean"]], y=[s["name"]], mode="markers", showlegend=False,
                                 marker=dict(symbol="diamond", size=11, color=color),
                                 hovertemplate=f"<b>{s['name']}</b><br>今回 平均 {s['mean']:.1f} ms<extra></extra>"))
    else:
        if s["ref"] is not None:
            fig.add_trace(go.Bar(y=[s["name"]], x=[s["ref"]], base=[s["ref_start"]], orientation="h",
                                 marker=dict(color="rgba(139,148,158,0.30)", line=dict(color="#8b949e", width=1)),
                                 width=0.8, showlegend=False,
                                 hovertemplate=f"基準 開始 {s['ref_start']:.1f} ／ 長さ {s['ref']:.1f} ms<extra></extra>"))
        fig.add_trace(go.Bar(y=[s["name"]], x=[s["mean"]], base=[s["start_mean"]], orientation="h",
                             marker_color=color, width=0.4, showlegend=False,
                             hovertemplate=(f"<b>{s['name']}</b><br>今回 開始 {s['start_mean']:.1f} ／ "
                                            f"長さ {s['mean']:.1f} ms<extra></extra>")))
if ev["takt"]:
    fig.add_vline(x=ev["takt"], line_dash="dash", line_color=STATUS_COLOR["ng"],
                  annotation_text=f"タクト目標 {ev['takt']} ms", annotation_position="top")
fig.update_layout(barmode="overlay", height=max(240, len(names) * 32 + 90), margin=dict(l=0, r=16, t=28, b=40),
                  xaxis=dict(title="サイクル開始からの時間 [ms]", rangemode="tozero"),
                  yaxis=dict(categoryorder="array", categoryarray=names, autorange="reversed"))
st.caption("灰色が基準、色付きが今回の平均です（赤＝異常、黄＝注意、緑＝正常）。")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_eval_fig_{pname}")

# 判定表（異常→注意→正常、同じ判定は差の大きい順）
order = sorted(steps, key=lambda s: (-STATUS[s["status"]][2], -abs(s["delta"] or 0)))
table = pd.DataFrame([{
    "判定": STATUS[s["status"]][0],
    "ステップ": s["name"],
    "測り方": MODE_LABELS.get(s["mode"], ""),
    "基準 [ms]": s["ref"],
    "今回 平均 [ms]": s["mean"],
    "差 [ms]": s["delta"],
    "許容幅 ± [ms]": s["tol"],
    "NG率 [%]": s["ng_rate"],
    "NG / サイクル": f"{int(s['ng_mask'].sum())} / {s['n']}" if s["n"] else "—",
} for s in order])
table = numeric(table, ["基準 [ms]", "今回 平均 [ms]", "差 [ms]", "許容幅 ± [ms]", "NG率 [%]"])
ss.setdefault("_eval_tbl_v", 0)
sel = st.dataframe(
    table.style.map(status_cell_style, subset=["判定"]).format(
        {c: "{:.1f}" for c in ("基準 [ms]", "今回 平均 [ms]", "差 [ms]", "許容幅 ± [ms]", "NG率 [%]")},
        na_rep="—"),
    hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    key=f"_eval_tbl_{pname}_{ss['_eval_tbl_v']}",
)
st.caption("時刻ステップは発生時刻、期間ステップは長さを比べています。行を選ぶとステップの詳細を開きます。")
if sel and sel.selection.rows:
    picked = order[sel.selection.rows[0]]
    ss["detail"] = {"proc": pname, "step": picked["name"], "data": target}
    ss["_eval_tbl_v"] += 1  # 戻ってきたときに選択が残って再遷移しないように
    st.switch_page("views/step.py")
