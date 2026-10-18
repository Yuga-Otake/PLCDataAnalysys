"""
views/trend.py — 推移

複数時期の比較データを並べ、いつから・どのステップが基準から離れ始めたかを見る。
全ステップを一覧するヒートマップ → 1 ステップの推移と管理図、の順に掘り下げる。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.evaluate import STATUS, compare_keys_by_time, control_limits, evaluate_process, rule
from core.state import pk
from core.ui import REF_LINE, STATUS_COLOR, numeric, page_header, rule_popover, tolerance_band
from core.wave_inspect import _compute_wi_det_stats_for_csv

ss = st.session_state
page_header(
    "推移",
    "時期の違う比較データを古い順に並べ、ステップごとに基準からどれだけ離れたかを示します。",
    eyebrow="点検",
)
store = ss.get("csv_store", {})
processes = ss.get("processes", {})
ready = [p for p in processes if ss.get(pk(p, "baseline"))]
all_keys = compare_keys_by_time()
if not ready or len(all_keys) < 2:
    st.info("推移を見るには、基準値の登録と、時期の違う比較データが 2 件以上必要です。")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("データ画面へ", type="primary"):
        st.switch_page("views/data.py")
    if c2.button("基準値へ"):
        st.switch_page("views/baseline.py")
    st.stop()

c1, c2, c3 = st.columns([2, 4, 1], vertical_alignment="bottom")
if ss.get("sel_process") not in ready:
    ss["sel_process"] = ready[0]
pname = c1.selectbox("工程", ready, key="sel_process")
if "trend_keys" not in ss or not set(ss["trend_keys"]) <= set(all_keys):
    ss["trend_keys"] = all_keys
keys = c2.multiselect("並べるデータ（古い順に並びます）", all_keys, key="trend_keys",
                      format_func=lambda k: store[k]["label"], placeholder="データを選んでください")
with c3:
    rule_popover()
keys = [k for k in all_keys if k in keys]
if len(keys) < 2:
    st.caption("データを 2 件以上選んでください。")
    st.stop()

r = rule()
evals = {k: evaluate_process(store[k]["df"], pname, r) for k in keys}
labels = [store[k]["label"] for k in keys]
steps = ss.get(pk(pname, "steps_list"), [])
names = [s["name"] for s in steps]

# ── 全ステップの一覧（ヒートマップ）─────────────────────────────
st.subheader("全ステップの推移", anchor=False)
z, text, hover = [], [], []
for i, n in enumerate(names):
    zr, tr, hr = [], [], []
    for k in keys:
        s = evals[k]["steps"][i]
        if s["delta"] is None or not s["tol"]:
            zr.append(None); tr.append(""); hr.append("—")
            continue
        zr.append(s["delta"] / s["tol"])
        tr.append(f"{s['delta']:+.0f}")
        hr.append(f"差 {s['delta']:+.1f} ms ／ 許容幅 ±{s['tol']:.1f} ／ NG率 {s['ng_rate']:.0f}%"
                  f" ／ {STATUS[s['status']][0]}")
    z.append(zr); text.append(tr); hover.append(hr)
fig = go.Figure(go.Heatmap(
    z=z, x=labels, y=names, text=text, texttemplate="%{text}", customdata=hover,
    hovertemplate="%{y}<br>%{x}<br>%{customdata}<extra></extra>",
    zmin=-2, zmax=2, zmid=0,
    colorscale=[[0, "#548cff"], [0.25, "rgba(84,140,255,0.35)"], [0.5, "rgba(139,148,158,0.15)"],
                [0.75, "rgba(229,83,75,0.45)"], [1, "#e5534b"]],
    colorbar=dict(title="差 ÷ 許容幅", tickvals=[-2, -1, 0, 1, 2]),
))
fig.update_layout(height=max(260, len(names) * 28 + 90), margin=dict(l=0, r=8, t=10, b=40),
                  yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_trend_heat_{pname}")
st.caption("数字は基準との差 [ms]。赤いほど基準より長く（遅く）、青いほど短く（早く）なっています。"
           "±1 を超えると許容幅の外です。")

# ── 1 ステップの推移と管理図 ─────────────────────────────────
st.subheader("ステップの推移と管理図", anchor=False)
# 最後の時期で外れが大きいステップを初期表示にする
last = evals[keys[-1]]["steps"]
default = max(range(len(names)), key=lambda i: abs(last[i]["delta"] or 0) / (last[i]["tol"] or 1))
sk = f"_trend_step_{pname}"
if ss.get(sk) not in names:
    ss[sk] = names[default]
sname = st.selectbox("ステップ", names, key=sk)
i = names.index(sname)
measure = "発生時刻" if steps[i].get("mode", "single") == "single" else "長さ"
series = [evals[k]["steps"][i] for k in keys]
groups = [s["series"]["value"].to_numpy(dtype=float) for s in series]
ref, tol = series[-1]["ref"], series[-1]["tol"]

left, right = st.columns(2)
with left:
    st.markdown(f"**{measure}の推移（平均 ± σ）**")
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=sum([[lbl] * len(g) for lbl, g in zip(labels, groups)], []),
        y=np.concatenate(groups) if groups else [], boxpoints=False,
        marker_color="#8b949e", line=dict(width=1), fillcolor="rgba(139,148,158,0.15)",
        name="分布", showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=labels, y=[s["mean"] for s in series], mode="lines+markers", showlegend=False,
        marker=dict(size=10, color=[STATUS_COLOR[s["status"]] for s in series]),
        line=dict(color="#8b949e"),
        customdata=[s["ng_rate"] or 0 for s in series],
        hovertemplate="%{x}<br>平均 %{y:.1f} ms<br>NG率 %{customdata:.1f}%<extra></extra>"))
    if ref is not None:
        tolerance_band(fig, ref, tol)
    fig.update_layout(height=340, margin=dict(l=0, r=8, t=10, b=40), yaxis_title=f"{measure} [ms]")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_trend_line_{pname}")
    st.caption("箱はサイクルのばらつき、点は平均（色は判定）、帯は許容幅です。")

with right:
    lim = control_limits(groups)
    if lim is None:
        st.caption("管理図には、2 サイクル以上ある時期が 2 つ以上必要です。")
    else:
        sub_name = "R（範囲）" if lim["kind"] == "R" else "S（標準偏差）"
        st.markdown(f"**X̄-{lim['kind']} 管理図**（サブグループ = 各時期、平均 {lim['n']} サイクル）")
        out_x = [v > lim["ucl"] or v < lim["lcl"] for v in lim["xbar"]]
        out_s = [v > lim["sub_ucl"] or (lim["sub_lcl"] > 0 and v < lim["sub_lcl"]) for v in lim["sub"]]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, row_heights=[0.6, 0.4])
        for row, ys, outs, cl, ucl, lcl in (
                (1, lim["xbar"], out_x, lim["cl"], lim["ucl"], lim["lcl"]),
                (2, lim["sub"], out_s, lim["sub_cl"], lim["sub_ucl"], lim["sub_lcl"])):
            fig.add_trace(go.Scatter(x=labels, y=ys, mode="lines+markers", showlegend=False,
                                     line=dict(color="#8b949e"),
                                     marker=dict(size=9, color=[STATUS_COLOR["ng"] if o else REF_LINE for o in outs])),
                          row=row, col=1)
            for y, dash in ((cl, "solid"), (ucl, "dash"), (lcl, "dash")):
                if row == 2 and y == lcl and lcl <= 0:
                    continue
                fig.add_hline(y=y, line_dash=dash, line_width=1,
                              line_color="#8b949e" if dash == "solid" else STATUS_COLOR["ng"], row=row, col=1)
        fig.update_yaxes(title_text="X̄ [ms]", row=1, col=1)
        fig.update_yaxes(title_text=f"{sub_name} [ms]", row=2, col=1)
        fig.update_layout(height=340, margin=dict(l=0, r=8, t=10, b=40))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_trend_xbr_{pname}")
        n_out = sum(out_x) + sum(out_s)
        st.caption(f"実線が中心線、破線が管理限界です。限界の外に出た時期は {n_out} 点（赤い点）。"
                   "基準値ではなく、並べた時期どうしのばらつきから求めています。")

if st.button(f"「{sname}」の詳細を開く（{labels[-1]}）", icon=":material/open_in_new:", key="_trend_detail"):
    ss["detail"] = {"proc": pname, "step": sname, "data": keys[-1]}
    st.switch_page("views/step.py")

# 表
st.dataframe(numeric(pd.DataFrame([{
    "データ": lbl, "サイクル数": s["n"], "平均 [ms]": s["mean"], "σ [ms]": s["std"],
    "基準との差 [ms]": s["delta"], "NG率 [%]": s["ng_rate"], "判定": STATUS[s["status"]][0],
} for lbl, s in zip(labels, series)]), ["平均 [ms]", "σ [ms]", "基準との差 [ms]", "NG率 [%]"]).style.format(
    {c: "{:.1f}" for c in ("平均 [ms]", "σ [ms]", "基準との差 [ms]", "NG率 [%]")}, na_rep="—"),
    hide_index=True, width="stretch")

# ── 波形検査の検出点 ─────────────────────────────────────────
st.subheader("波形検査の検出点の推移", anchor=False)
det_series, det_errors = {}, []
for k, lbl in zip(keys, labels):
    d = store[k]["df"]
    nums = [c for c, t in store[k]["col_types"].items() if t == "numeric"]
    try:
        stats = _compute_wi_det_stats_for_csv(d, nums, ss)
    except Exception as e:
        det_errors.append(f"{lbl}: {e}")
        continue
    for dk, dv in stats.items():
        if dk.startswith("__"):
            continue
        det_series.setdefault(dk, {"info": dv, "rows": []})["rows"].append({"label": lbl, **dv})
for e in det_errors:
    st.error(f"検出点を計算できませんでした（{e}）")
configured = any(v is True and k.startswith("wvol___global_") and k.endswith("_trend_on")
                 for k, v in ss.items() if isinstance(k, str))
if not det_series and configured:
    st.warning("「傾向解析に出す」をオンにした検出点はありますが、どのデータでも検出できませんでした。"
               "検出点の種類・閾値・検索範囲、変数ごとのサイクル開始信号を波形検査条件の画面で確認してください。")
    if st.button("波形検査条件へ", key="_trend_to_wave_cfg"):
        st.switch_page("views/wave.py")
elif not det_series:
    st.caption("波形検査条件の画面で検出点を追加し、「傾向解析に出す」をオンにすると、ここに時期ごとの変化が表示されます。")
    if st.button("波形検査条件へ", key="_trend_to_wave"):
        st.switch_page("views/wave.py")
else:
    st.caption("波形検査条件で「傾向解析に出す」をオンにした検出点の、時期ごとの平均 ± σ です。")
    for dk, d in det_series.items():
        info, rows = d["info"], d["rows"]
        xs = [r["label"] for r in rows]
        panels = [("t_mean", "t_std", info.get("x_label", "時刻 [ms]"))]
        if not info.get("is_formula"):
            panels.append(("v_mean", "v_std", info.get("y_label", "値")))
        st.markdown(f"**{info.get('label', dk)}**")
        cols = st.columns(len(panels))
        for col, (mk, sk_, title) in zip(cols, panels):
            fig = go.Figure(go.Scatter(
                x=xs, y=[r[mk] for r in rows], mode="lines+markers",
                error_y=dict(type="data", array=[r.get(sk_, 0) for r in rows], color="#8b949e", thickness=1),
                marker=dict(size=9, color=REF_LINE), line=dict(color="#8b949e"), showlegend=False,
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>"))
            fig.update_layout(height=240, margin=dict(l=0, r=8, t=10, b=40), yaxis_title=title)
            col.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_trend_det_{dk}_{mk}")

