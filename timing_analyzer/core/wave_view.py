"""
core/wave_view.py — 波形検査点の表示（工程 → 検査波形 → 検査点）

波形検査条件の登録サマリーと、工程とステップ画面の「この工程の波形検査点」で使う。
データの整理は core/wave_inspect.inspection_tree。
"""
import re

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.wave_inspect import _wi_edit_dialog

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


def render_points(points: list, editable: bool, key: str):
    """検査点の一覧（タイプ・名前・位置・有効・推移、編集ボタン）"""
    # スマホでは列が縦に積まれて読めなくなるので、横並びのまま横スクロールさせる（core/ui.py の CSS）
    with st.container(key=f"wisumgrid_{key}"):
        widths = [1.5, 1.5, 3.2, 0.5, 0.5] + ([0.6] if editable else [])
        for lbl, col in zip(["タイプ", "名前", "位置（基準データ）", "有効", "推移", ""], st.columns(widths)):
            col.caption(lbl)
        for p in points:
            c = st.columns(widths)
            c[0].markdown(p["type"] + ("　<small>XY</small>" if p["graph"] == "XY" else ""), unsafe_allow_html=True)
            c[1].markdown(f"**{p['name']}**" if p["name"] else "−")
            c[2].markdown(_position(p), unsafe_allow_html=True)
            c[3].markdown("✅" if p["on"] else "−")
            c[4].markdown("📈" if p["trend"] else "−")
            if editable and c[5].button("⚙️", key=f"_wisum_edit_{p['dkey']}", help="波形を見ながら編集"):
                st.session_state["wi_edit_target"] = _edit_target(p)
                _wi_edit_dialog()


def render_process(node: dict, editable: bool):
    """工程 1 つ分：検査波形ごとに、図と検査点の一覧"""
    for w in node["waves"]:
        n = len(w["points"])
        with st.container(border=True):
            st.markdown(f"**{short_var(w['var'])}**　<span class='apb-lead'>{w['var']}・検査点 {n}</span>",
                        unsafe_allow_html=True)
            if w["wave"] and w["graph"] == "時間軸":
                st.plotly_chart(wave_figure(w, node["stats"]), width="stretch", config={"displayModeBar": False},
                                key=f"_wtree_{node['process']}_{w['var']}_{editable}")
            render_points(w["points"], editable, f"{abs(hash((node['process'], w['var']))) % 10**8}")


def process_header(node: dict) -> str:
    n_w = len(node["waves"])
    n_p = sum(len(w["points"]) for w in node["waves"])
    if not node["process"]:
        return f"工程に紐づいていない検査点　波形 {n_w} 本・検査点 {n_p}"
    edge = "立ち上がり" if node["edge"] == "RISE" else "立ち下がり"
    return f"{node['process']}　波形 {n_w} 本・検査点 {n_p}　（サイクル開始 {node['trigger']}・{edge}）"
