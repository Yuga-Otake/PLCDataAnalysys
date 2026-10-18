"""
views/step.py — ステップ詳細（メニューには出さない掘り下げ専用ページ）

評価・推移から開く。分布・サイクルごとの値・時期ごとの推移・サイクル一覧・アナログ波形を 1 画面に並べる。
旧画面の詳細解析タブ・ヒストグラム詳細ページ・詳細ページを置き換える。
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analyzer import get_cycle_waveforms, normalize_bool_series
from core.analysis import MODE_LABELS, cycle_starts, describe_step
from core.evaluate import STATUS, compare_keys_by_time, evaluate_process, rule
from core.state import pk
from core.ui import REF_LINE, STATUS_COLOR, chip, page_header, rule_popover, tolerance_band

ss = st.session_state
store = ss.get("csv_store", {})
processes = ss.get("processes", {})
ready = [p for p in processes if ss.get(pk(p, "baseline"))]
cmp_keys = compare_keys_by_time()
detail = ss.get("detail") or {}

if not ready or not cmp_keys:
    page_header("ステップ詳細", eyebrow="点検")
    st.info("基準値と比較データがそろうと、ステップの詳細を確認できます。")
    if st.button("最新データの評価へ", type="primary"):
        st.switch_page("views/evaluate.py")
    st.stop()

# ── 対象（工程・ステップ・データ）────────────────────────────
pname = detail.get("proc") if detail.get("proc") in ready else ready[0]
steps = ss.get(pk(pname, "steps_list"), [])
names = [s["name"] for s in steps]
sname = detail.get("step") if detail.get("step") in names else (names[0] if names else None)
target = detail.get("data") if detail.get("data") in cmp_keys else cmp_keys[-1]

page_header(sname or "ステップ詳細", crumbs=["最新データの評価", pname, sname or ""])
b1, b2, b3, b4, b5 = st.columns([1.1, 2.4, 2.6, 0.9, 0.9], vertical_alignment="bottom")
if b1.button("評価に戻る", icon=":material/arrow_back:", key="_step_back"):
    st.switch_page("views/evaluate.py")


def _set(**kw):
    ss["detail"] = {**{"proc": pname, "step": sname, "data": target}, **kw}


new_p = b2.selectbox("工程", ready, index=ready.index(pname), key=f"_step_p_{pname}")
if new_p != pname:
    _set(proc=new_p, step=None)
    st.rerun()
if not names:
    st.info("この工程にはステップがありません。")
    st.stop()
new_s = b3.selectbox("ステップ", names, index=names.index(sname), key=f"_step_s_{pname}_{sname}")
if new_s != sname:
    _set(step=new_s)
    st.rerun()
idx = names.index(sname)
if b4.button("前", disabled=idx == 0, key="_step_prev", icon=":material/chevron_left:"):
    _set(step=names[idx - 1])
    st.rerun()
if b5.button("次", disabled=idx == len(names) - 1, key="_step_next", icon=":material/chevron_right:"):
    _set(step=names[idx + 1])
    st.rerun()

d1, d2 = st.columns([4, 1], vertical_alignment="bottom")
new_t = d1.selectbox("データ", cmp_keys, index=cmp_keys.index(target), key=f"_step_t_{target}",
                     format_func=lambda k: store[k]["label"] + ("（最新）" if k == cmp_keys[-1] else ""))
if new_t != target:
    _set(data=new_t)
    st.rerun()
with d2:
    rule_popover()

# ── 判定 ─────────────────────────────────────────────────────
step = steps[idx]
ev = evaluate_process(store[target]["df"], pname)
s = ev["steps"][idx]
series = s["series"]
label, kind, _ = STATUS[s["status"]]
measure = "発生時刻" if s["mode"] == "single" else "長さ"
st.caption(f"{MODE_LABELS.get(s['mode'], '')}｜{describe_step(step)}｜{measure}で判定")
if s["mean"] is None:
    st.warning("このデータではステップを検出できませんでした。")
    st.stop()
st.markdown(
    f"{chip(label, kind)}　<span class='apb-kv'>"
    f"<span>基準 <b class='apb-num'>{s['ref']:.1f} ms</b></span>"
    f"<span>今回 平均 <b class='apb-num'>{s['mean']:.1f} ms</b></span>"
    f"<span>差 <b class='apb-num'>{s['delta']:+.1f} ms</b></span>"
    f"<span>許容幅 <b class='apb-num'>± {s['tol']:.1f} ms</b></span>"
    f"<span>NG <b class='apb-num'>{int(s['ng_mask'].sum())} / {s['n']}</b> サイクル（{s['ng_rate']:.1f}%）</span>"
    "</span>" if s["ref"] is not None else chip(label, kind), unsafe_allow_html=True)

vals = series["value"].to_numpy(dtype=float)
ng = s["ng_mask"]
lo, hi = (s["ref"] - s["tol"], s["ref"] + s["tol"]) if s["ref"] is not None else (None, None)


def _band(fig, horizontal: bool):
    if lo is not None:
        tolerance_band(fig, s["ref"], s["tol"], horizontal=horizontal)


# ── 分布とサイクルごとの値 ───────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**{measure}の分布**")
    nb = max(8, min(40, int(np.sqrt(len(vals)) * 2)))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals[~ng], nbinsx=nb, name="許容幅内", marker_color=STATUS_COLOR["ok"], opacity=0.8))
    if ng.any():
        fig.add_trace(go.Histogram(x=vals[ng], nbinsx=nb, name="NG", marker_color=STATUS_COLOR["ng"], opacity=0.9))
    _band(fig, horizontal=False)
    fig.update_layout(barmode="overlay", height=300, margin=dict(l=0, r=8, t=30, b=40),
                      xaxis_title=f"{measure} [ms]", yaxis_title="サイクル数",
                      legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="_step_hist")
with c2:
    st.markdown("**サイクルごとの値**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series["サイクル#"], y=vals, mode="lines+markers",
                             line=dict(color="#8b949e", width=1),
                             marker=dict(size=7, color=[STATUS_COLOR["ng"] if x else STATUS_COLOR["ok"] for x in ng]),
                             hovertemplate="サイクル %{x}<br>%{y:.1f} ms<extra></extra>", showlegend=False))
    _band(fig, horizontal=True)
    fig.update_layout(height=300, margin=dict(l=0, r=8, t=30, b=40),
                      xaxis_title="サイクル", yaxis_title=f"{measure} [ms]")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="_step_seq")

# ── 時期ごとの推移 ───────────────────────────────────────────
if len(cmp_keys) >= 2:
    st.markdown("**時期ごとの推移**")
    r = rule()
    n_base = ev["n_base"]
    trig, edge = ss.get(pk(pname, "trigger")), ss.get(pk(pname, "edge"), "RISE")
    rows = []
    for k in cmp_keys:
        res = evaluate_process(store[k]["df"], pname, r)["steps"][idx]
        if res["mean"] is not None:
            rows.append({"label": store[k]["label"], "mean": res["mean"], "std": res["std"],
                         "status": res["status"], "rate": res["ng_rate"], "key": k})
    if rows:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[x["label"] for x in rows], y=[x["mean"] for x in rows], mode="lines+markers",
            error_y=dict(type="data", array=[x["std"] for x in rows], color="#8b949e", thickness=1),
            line=dict(color="#8b949e"),
            marker=dict(size=[14 if x["key"] == target else 9 for x in rows],
                        color=[STATUS_COLOR[x["status"]] for x in rows]),
            customdata=[x["rate"] or 0 for x in rows],
            hovertemplate="%{x}<br>平均 %{y:.1f} ms<br>NG率 %{customdata:.1f}%<extra></extra>",
            showlegend=False))
        _band(fig, horizontal=True)
        fig.update_layout(height=280, margin=dict(l=0, r=8, t=24, b=40), yaxis_title=f"{measure} [ms]")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="_step_trend")
        st.caption("点は各時期の平均、縦線は ±σ、帯は許容幅です。大きい点が表示中のデータです。")

# ── アナログ波形 ─────────────────────────────────────────────
wv = [v for v in step.get("waveform_vars", []) if v in store[target]["df"].columns]
if wv:
    st.markdown("**アナログ波形（サイクル開始を 0 にそろえて重ね表示）**")
    df = store[target]["df"]
    starts = pd.Index(cycle_starts(df, ss.get(pk(pname, "trigger")), ss.get(pk(pname, "edge"), "RISE")))
    waves = get_cycle_waveforms(df, starts, wv)
    ng_cycles = set(series.loc[ng, "サイクル#"].astype(int))
    for var in wv:
        fig = go.Figure()
        for i, w in enumerate(waves):
            bad = (i + 1) in ng_cycles
            fig.add_trace(go.Scatter(x=w["time_offset_ms"], y=w[var], mode="lines", showlegend=False,
                                     line=dict(width=1.6 if bad else 0.8,
                                               color=STATUS_COLOR["ng"] if bad else "rgba(139,148,158,0.45)"),
                                     hovertemplate=f"サイクル {i + 1}<br>%{{x:.0f}} ms / %{{y:.2f}}<extra></extra>"))
        fig.add_vline(x=s["start_mean"], line_dash="dot", line_color=REF_LINE,
                      annotation_text="ステップ開始（平均）", annotation_position="top")
        fig.update_layout(height=260, margin=dict(l=0, r=8, t=28, b=40),
                          xaxis_title="サイクル開始からの時間 [ms]", yaxis_title=var)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=f"_step_wave_{var}")
    st.caption("赤い線が NG のサイクルです。")

# ── 元の信号（時系列）─────────────────────────────────────────
with st.expander("元の信号（時系列）を見る"):
    df_t = store[target]["df"]
    trig = ss.get(pk(pname, "trigger"))
    starts_t = cycle_starts(df_t, trig, ss.get(pk(pname, "edge"), "RISE"))
    if len(starts_t) < 1:
        st.caption("サイクルを検出できません。")
    else:
        step_vars = [v for v in (trig, step.get("variable"), step.get("start_var"), step.get("end_var"))
                     if v and v in df_t.columns and store[target]["col_types"].get(v) == "bool"]
        num_opts = [c for c, t in store[target]["col_types"].items() if t == "numeric"]
        ng_list = sorted(series.loc[ng, "サイクル#"].astype(int))
        first = max(1, (ng_list[0] - 1) if ng_list else 1)
        n_cyc = len(starts_t)
        t1, t2 = st.columns([2, 3])
        rng = t1.slider("表示するサイクル", 1, n_cyc, (first, min(n_cyc, first + 4)),
                        key=f"_step_ts_rng_{pname}_{sname}_{target}")
        extra = t2.multiselect("一緒に表示するアナログ変数", num_opts, placeholder="なし",
                               default=[v for v in step.get("waveform_vars", []) if v in num_opts],
                               key=f"_step_ts_num_{pname}_{sname}")
        a_idx = starts_t[rng[0] - 1]
        b_idx = starts_t[rng[1]] if rng[1] < n_cyc else df_t.index[-1]
        win = df_t.loc[a_idx:b_idx]
        fig = make_subplots(rows=len(step_vars) + (1 if extra else 0), cols=1, shared_xaxes=True,
                            vertical_spacing=0.04)
        for i, v in enumerate(step_vars, start=1):
            fig.add_trace(go.Scatter(x=win["Timestamp"], y=normalize_bool_series(win[v]), mode="lines",
                                     line=dict(width=1.2, shape="hv", color=REF_LINE), fill="tozeroy",
                                     name=v, showlegend=False), row=i, col=1)
            fig.update_yaxes(title_text=v, title_font=dict(size=10), tickvals=[0, 1], row=i, col=1)
        for v in extra:
            fig.add_trace(go.Scatter(x=win["Timestamp"], y=win[v], mode="lines", name=v,
                                     line=dict(width=1)), row=len(step_vars) + 1, col=1)
        for c in range(rng[0], rng[1] + 1):
            ts = df_t.loc[starts_t[c - 1], "Timestamp"]
            fig.add_vline(x=ts, line_width=0.8, line_dash="dash",
                          line_color=STATUS_COLOR["ng"] if c in ng_list else "#8b949e")
        fig.update_layout(height=110 * len(step_vars) + (200 if extra else 0) + 60,
                          margin=dict(l=0, r=8, t=10, b=30), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="_step_ts")
        st.caption("縦の破線がサイクルの開始です（赤は NG のサイクル）。")

# ── サイクル一覧 ─────────────────────────────────────────────
st.markdown("**サイクル一覧**")
only_ng = st.toggle("NG のサイクルだけ表示", value=bool(ng.any()), key=f"_step_only_ng_{pname}_{sname}")
tbl = pd.DataFrame({
    "判定": np.where(ng, "NG", "OK"),
    "サイクル#": series["サイクル#"].astype(int),
    "開始時刻": series["開始時刻"].astype(str).str[:23],
    "ステップ開始 [ms]": series["start"].round(1),
    f"{measure} [ms]": series["value"].round(1),
    "基準との差 [ms]": (series["value"] - s["ref"]).round(1) if s["ref"] is not None else None,
})
if only_ng:
    tbl = tbl[tbl["判定"] == "NG"]
st.dataframe(tbl.style.map(lambda v: "background-color: rgba(229, 83, 75, 0.22)" if v == "NG" else "",
                           subset=["判定"]),
             hide_index=True, width="stretch")
st.download_button("この一覧を CSV で保存", data=tbl.to_csv(index=False).encode("utf-8-sig"),
                   file_name=f"{pname}_{sname}_cycles.csv", mime="text/csv", icon=":material/download:")
