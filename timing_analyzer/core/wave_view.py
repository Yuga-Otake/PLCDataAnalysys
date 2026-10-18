"""
core/wave_view.py — 波形検査点の表示（工程 → 検査波形 → 検査点）

波形検査条件の登録サマリーと、工程とステップ画面の「この工程の波形検査点」で使う。
データの整理は core/wave_inspect.inspection_tree。
"""
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.profile import analog_vars, wave_hypotheses_all, wave_repeatability
from core.wave_inspect import _wi_edit_dialog, adopt_point, assign_waveform, delete_point

_ACCENT = "#1f6feb"
_POINT = "#e5534b"


def short_var(var: str) -> str:
    """変数名から工程の部分（先頭の「工程名.」）を省く"""
    return var.split(".", 1)[1] if "." in var else var


def _unit(var: str) -> str:
    m = re.search(r"\[([^\]]+)\]\s*$", var)
    return m.group(1) if m else "値"


def wave_figure(w: dict, stats: list) -> go.Figure:
    """検査波形 1 本の図。上：基準データの平均波形（帯は 5〜95%）と検査点（◇）。下：工程のステップの帯（同じ時間軸）"""
    rows = 2 if stats else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.7, 0.3] if rows == 2 else [1.0])
    ys = []
    if w["wave"]:
        t, mean, lo, hi = w["wave"]
        ys = [v for v in lo + hi if v == v]
        fig.add_trace(go.Scatter(x=t + t[::-1], y=hi + lo[::-1], fill="toself", mode="lines",
                                 line=dict(width=0), fillcolor="rgba(31,111,235,0.12)",
                                 hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=mean, mode="lines", line=dict(color=_ACCENT, width=1.6),
                                 hovertemplate="%{x:.0f} ms　%{y:.4g}<extra>平均</extra>", showlegend=False),
                      row=1, col=1)
    pts = [p for p in w["points"] if p["t"] is not None]
    if pts:
        ys += [p["v"] for p in pts]
        fig.add_trace(go.Scatter(
            x=[p["t"] for p in pts], y=[p["v"] for p in pts], mode="markers+text",
            marker=dict(size=11, symbol="diamond", color="rgba(0,0,0,0)", line=dict(color=_POINT, width=2)),
            text=[p["name"] or p["type"] for p in pts], textposition="top center",
            textfont=dict(size=11, color=_POINT),
            customdata=[[p["type"], p["where"]] for p in pts],
            hovertemplate="<b>%{text}</b>（%{customdata[0]}）<br>%{x:.1f} ms　値 %{y:.4g}"
                          "<br>%{customdata[1]}<extra></extra>",
            showlegend=False), row=1, col=1)
        for p in pts:   # 検査点の時刻をステップの帯まで縦線でつなぐ
            fig.add_vline(x=p["t"], line_width=1, line_dash="dot", line_color=_POINT, opacity=0.5, row="all", col=1)
    if ys:   # 点の名前が上で切れないよう余白をとる
        y0, y1 = min(ys), max(ys)
        pad = (y1 - y0) or 1.0
        fig.update_yaxes(range=[y0 - 0.08 * pad, y1 + 0.3 * pad], row=1, col=1)
    names = []
    for s in stats:
        names.append(s["name"])
        if s["mode"] == "single":
            fig.add_trace(go.Scatter(x=[s["start_mean"]], y=[s["name"]], mode="markers",
                                     marker=dict(symbol="diamond", size=8, color=s["color"]),
                                     hovertemplate=f"{s['name']}　{s['start_mean']:.0f} ms<extra></extra>",
                                     showlegend=False), row=2, col=1)
        else:
            fig.add_trace(go.Bar(x=[s["dur_mean"]], base=[s["start_mean"]], y=[s["name"]], orientation="h",
                                 marker_color=s["color"], opacity=0.75, width=0.6, showlegend=False,
                                 hovertemplate=f"{s['name']}　{s['start_mean']:.0f}〜{s['end_mean']:.0f} ms"
                                               "<extra></extra>"), row=2, col=1)
    if rows == 2:
        fig.update_yaxes(categoryorder="array", categoryarray=names, autorange="reversed",
                         tickfont=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text=_unit(w["var"]), row=1, col=1)
    fig.update_xaxes(ticksuffix=" ms", row=rows, col=1)
    fig.update_layout(height=200 + (16 * len(stats) if stats else 0), barmode="overlay",
                      margin=dict(l=0, r=8, t=10, b=30))
    return fig


def _position(p: dict) -> str:
    if p["t"] is None:
        return "<small>位置なし（判定・XY・未検出）</small>"
    txt = f"{p['t']:.0f} ms" + (f"　{p['where']}" if p["where"] else "")
    if p["rate"] is not None and p["rate"] < 0.9:
        txt += f"　<small>（検出 {p['rate'] * 100:.0f}% のサイクル）</small>"
    return txt


def _edit_target(p: dict) -> dict:
    """編集ダイアログ（_wi_edit_dialog）が受け取る形"""
    return {"変数": p["var"], "トリガー": p["trigger"], "エッジ": p["edge"], "グラフ": p["graph"], "No": p["no"],
            "タイプ": p["type"], "名前": p["name"], "有効": "✅" if p["on"] else "−",
            "傾向解析": "📈" if p["trend"] else "−", "_svkey": p["vkey"], "_did": p["did"], "_sv": p["var"],
            "_graph": p["graph"], "_dtype": p["type"]}


_ADD_TYPES = ["最大値点", "最小値点", "閾値超え検出", "傾き変化点", "上下判定比較", "最大値判定", "最小値判定",
              "検出点比較", "数式"]
_UNASSIGNED = "（工程なし）"


def _open_editor(p: dict):
    st.session_state["wi_edit_target"] = _edit_target(p)
    _wi_edit_dialog()


def render_points(points: list, editable: bool, key: str):
    """検査点の一覧（タイプ・名前・位置・有効・推移、編集・削除）"""
    # スマホでは列が縦に積まれて読めなくなるので、横並びのまま横スクロールさせる（core/ui.py の CSS）
    with st.container(key=f"wisumgrid_{key}"):
        widths = [1.5, 1.5, 3.2, 0.5, 0.5] + ([0.55, 0.55] if editable else [])
        for lbl, col in zip(["タイプ", "名前", "位置（基準データ）", "有効", "推移", "", ""], st.columns(widths)):
            col.caption(lbl)
        for p in points:
            c = st.columns(widths, vertical_alignment="center")
            c[0].markdown(p["type"] + ("　<small>XY</small>" if p["graph"] == "XY" else ""), unsafe_allow_html=True)
            c[1].markdown(f"**{p['name']}**" if p["name"] else "−")
            c[2].markdown(_position(p), unsafe_allow_html=True)
            c[3].markdown("✅" if p["on"] else "−")
            c[4].markdown("📈" if p["trend"] else "−")
            if not editable:
                continue
            if c[5].button("⚙️", key=f"_wisum_edit_{p['dkey']}", help="波形を見ながら編集"):
                _open_editor(p)
            with c[6].popover("🗑", help="削除"):
                st.markdown(f"「{p['name'] or p['type']}」を削除しますか？")
                st.button("削除する", key=f"_wisum_del_{p['dkey']}", type="primary",
                          on_click=delete_point, args=(p["vkey"], p["did"], p["graph"]))


def _reassign(var: str, key: str, procs: dict):
    name = st.session_state.get(key)
    if name in procs:
        assign_waveform(var, name, procs[name]["trigger"], procs[name]["edge"])


def _hypothesis(ref_df, var: str, trig: str, edge: str) -> dict:
    if ref_df is None or not trig:
        return {}
    return wave_hypotheses_all(ref_df, ((var, trig, edge),)).get(var, {})


def _add_point(var: str, node: dict, procs: dict, ref_df, key: str):
    """波形に検出点を 1 件追加して、すぐ編集ダイアログを開く（しきい値などは波形から決めた初期値）"""
    ss = st.session_state
    dtype, name = ss.get(f"{key}_type", "最大値点"), (ss.get(f"{key}_name") or "").strip()
    pts = next((w["points"] for w in node["waves"] if w["var"] == var), [])
    trig = pts[0]["trigger"] if pts else procs.get(node["process"], {}).get("trigger", "")
    edge = pts[0]["edge"] if pts else procs.get(node["process"], {}).get("edge", "RISE")
    h = _hypothesis(ref_df, var, trig, edge)
    did = adopt_point(var, dtype, name, dict(h.get("defaults", {}).get(dtype, {})), trig, edge,
                      h.get("wpost", 300), True, process=node["process"])
    ss["wi_edit_target"] = {"_svkey": f"wvol___global_{var}", "_did": did, "_sv": var, "_graph": "時間軸",
                            "_dtype": dtype, "No": len(pts) + 1, "変数": var, "グラフ": "時間軸", "タイプ": dtype,
                            "名前": name, "有効": "✅", "傾向解析": "📈", "トリガー": trig, "エッジ": edge}
    ss["wi_transition_to_edit"] = True


def render_process(node: dict, editable: bool, procs: dict | None = None, ref_df=None):
    """工程 1 つ分：検査波形ごとに、図と検査点の一覧。editable のときは工程の付け替え・検査点の追加・削除もできる"""
    procs = procs or {}
    for w in node["waves"]:
        n = len(w["points"])
        wid = abs(hash((node["process"], w["var"]))) % 10**8
        with st.container(border=True):
            if editable:
                h1, h2, h3 = st.columns([3.2, 2, 1.3], vertical_alignment="bottom")
                h1.markdown(f"**{short_var(w['var'])}**　<span class='apb-lead'>{w['var']}・検査点 {n}</span>",
                            unsafe_allow_html=True)
                opts = list(procs) + ([] if node["process"] else [_UNASSIGNED])
                k = f"_wassign_{w['var']}"
                st.session_state[k] = node["process"] or _UNASSIGNED
                h2.selectbox("工程", opts, key=k, on_change=_reassign, args=(w["var"], k, procs),
                             help="この波形を検査する工程。変えると、その工程のサイクル開始信号を基準に検出し直します"
                                  "（しきい値などの設定はそのまま）")
                with h3.popover("＋ 検査点", width="stretch"):
                    ak = f"_wadd_{wid}"
                    st.selectbox("検出点タイプ", _ADD_TYPES, key=f"{ak}_type",
                                 help="点取得系: 最大値点・最小値点・閾値超え検出・傾き変化点｜判定系: それ以外")
                    st.text_input("名前（任意）", key=f"{ak}_name", placeholder="例: 押し込みピーク")
                    st.button("追加して波形で調整", type="primary", key=f"{ak}_go", width="stretch",
                              on_click=_add_point, args=(w["var"], node, procs, ref_df, ak))
                if w.get("how"):
                    st.caption(f"工程の決め方: {w['how']}")
            else:
                st.markdown(f"**{short_var(w['var'])}**　<span class='apb-lead'>{w['var']}・検査点 {n}</span>",
                            unsafe_allow_html=True)
            if w["wave"] and w["graph"] == "時間軸":
                st.plotly_chart(wave_figure(w, node["stats"]), width="stretch", config={"displayModeBar": False},
                                key=f"_wtree_{node['process']}_{w['var']}_{editable}")
            render_points(w["points"], editable, str(wid))


def _add_waveform(node: dict, key: str, ref_df):
    """工程に検査波形を追加：選んだ変数の、波形から出した候補をまとめて登録する（候補がなければ最大値点を 1 つ）"""
    ss = st.session_state
    var = ss.get(f"{key}_var")
    if not var:
        return
    h = _hypothesis(ref_df, var, node["trigger"], node["edge"])
    cands = [c for c in h.get("cands", []) if c["include"]]
    if cands:
        for c in cands:
            adopt_point(var, c["dtype"], c["name"], dict(c["params"]), node["trigger"], node["edge"],
                        h.get("wpost", 300), True, process=node["process"], how="手動")
        ss["_wave_msgs"] = [f"「{short_var(var)}」を「{node['process']}」に追加し、候補の検出点を "
                            f"{len(cands)} 件登録しました。⚙️ で調整できます。"]
    else:
        did = adopt_point(var, "最大値点", "", {}, node["trigger"], node["edge"], h.get("wpost", 300), True,
                          process=node["process"], how="手動")
        ss["wi_edit_target"] = {"_svkey": f"wvol___global_{var}", "_did": did, "_sv": var, "_graph": "時間軸",
                                "_dtype": "最大値点", "No": 1, "変数": var, "グラフ": "時間軸", "タイプ": "最大値点",
                                "名前": "", "有効": "✅", "傾向解析": "📈", "トリガー": node["trigger"],
                                "エッジ": node["edge"]}
        ss["wi_transition_to_edit"] = True


def add_waveform_popover(node: dict, ref_entry: dict | None):
    """工程に検査波形を追加するポップオーバー。変数は、この工程の開始信号で波形がそろう順に並べる"""
    if not ref_entry or not node["process"]:
        return
    used = {w["var"] for w in node["waves"]}
    avs = [v for v in analog_vars(ref_entry) if v not in used]
    if not avs:
        return
    sc = wave_repeatability(ref_entry["df"], tuple(avs), ((node["trigger"], node["edge"]),))
    avs.sort(key=lambda v: -sc.get(v, {}).get((node["trigger"], node["edge"]), -1))
    key = f"_waddw_{abs(hash(node['process'])) % 10**8}"
    with st.popover("＋ 検査波形を追加"):
        st.selectbox("変数", avs, key=f"{key}_var",
                     format_func=lambda v: f"{v}（そろい {sc.get(v, {}).get((node['trigger'], node['edge']), 0):.2f}）",
                     help="この工程のサイクル開始信号で切ったとき、波形が毎サイクルそろう（1 に近い）順に並べています")
        st.caption("波形の形から出した検出点の候補をまとめて登録します。候補がない波形は、最大値点を 1 つ追加して編集画面を開きます。")
        st.button("追加する", type="primary", key=f"{key}_go", on_click=_add_waveform, args=(node, key, ref_entry["df"]))


def process_header(node: dict) -> str:
    n_w = len(node["waves"])
    n_p = sum(len(w["points"]) for w in node["waves"])
    if not node["process"]:
        return f"工程に紐づいていない検査点　波形 {n_w} 本・検査点 {n_p}"
    edge = "立ち上がり" if node["edge"] == "RISE" else "立ち下がり"
    return f"{node['process']}　波形 {n_w} 本・検査点 {n_p}　（サイクル開始 {node['trigger']}・{edge}）"
