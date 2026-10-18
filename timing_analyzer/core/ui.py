"""
core/ui.py — 画面共通の見た目と部品

ライトは青と灰色のシンプルな配色、ダークは Sysmac Studio 風（Visual Studio 系の濃灰＋青）。
配色の大枠は .streamlit/config.toml の [theme.light] / [theme.dark]、
ここではチップ・段階表示・表など Streamlit 標準にない部品の色をモードごとに当てる。
"""
import html

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_PALETTE = {
    "light": {
        "accent": "#1f6feb", "accent-ink": "#0b4fb3", "accent-pale": "#ddebff",
        "ink": "#1f2328", "body": "#424a53", "label": "#6e7781", "hair": "#d0d7de", "band": "#f1f3f5",
        "ok": "#1a7f37", "ok-pale": "#dafbe1", "warn": "#9a6700", "warn-pale": "#fff1c5",
        "ng": "#cf222e", "ng-pale": "#ffebe9", "gray-pale": "#eaeef2",
    },
    "dark": {
        "accent": "#3794ff", "accent-ink": "#6cb6ff", "accent-pale": "#0e3a5c",
        "ink": "#f1f1f1", "body": "#cccccc", "label": "#9d9d9d", "hair": "#3f3f46", "band": "#2d2d30",
        "ok": "#89d185", "ok-pale": "#1f3a1f", "warn": "#d7ba7d", "warn-pale": "#3d3420",
        "ng": "#f48771", "ng-pale": "#4b2420", "gray-pale": "#333337",
    },
}

_CSS = """
<style>
:root {{ {vars} }}
h1, h2, h3, h4 {{ letter-spacing: .02em; }}
h1 {{ font-size: 24px !important; font-weight: 600 !important; padding-bottom: 0 !important; }}
h2 {{ font-size: 17px !important; font-weight: 600 !important; }}
h3 {{ font-size: 15px !important; font-weight: 600 !important; }}
.apb-eyebrow {{ font-size: 12px; color: var(--apb-label); letter-spacing: .08em; margin-bottom: -6px; }}
.apb-lead {{ color: var(--apb-body); font-size: 14px; max-width: 72ch; margin: 2px 0 6px; }}
.apb-crumb {{ font-size: 12px; color: var(--apb-label); }}
.apb-crumb b {{ color: var(--apb-ink); font-weight: 500; }}
.apb-chip {{ display: inline-block; font-size: 12px; line-height: 1; padding: 4px 8px; border-radius: 3px;
  font-weight: 600; white-space: nowrap; }}
.apb-chip.ok {{ background: var(--apb-ok-pale); color: var(--apb-ok); }}
.apb-chip.warn {{ background: var(--apb-warn-pale); color: var(--apb-warn); }}
.apb-chip.ng {{ background: var(--apb-ng-pale); color: var(--apb-ng); }}
.apb-chip.blue {{ background: var(--apb-accent-pale); color: var(--apb-accent-ink); }}
.apb-chip.gray {{ background: var(--apb-gray-pale); color: var(--apb-body); }}
.apb-steps {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 2px 0 4px; }}
.apb-steps span.s {{ font-size: 13px; padding: 4px 10px; border-radius: 3px; border: 1px solid var(--apb-hair);
  color: var(--apb-label); white-space: nowrap; }}
.apb-steps span.s.done {{ background: var(--apb-gray-pale); color: var(--apb-body); border-color: transparent; }}
.apb-steps span.s.cur {{ border-color: var(--apb-accent); color: var(--apb-accent-ink); font-weight: 600;
  background: var(--apb-accent-pale); }}
.apb-steps span.sep {{ color: var(--apb-hair); }}
.apb-kv {{ display: flex; flex-wrap: wrap; gap: 4px 28px; font-size: 13px; color: var(--apb-label); }}
.apb-kv b {{ color: var(--apb-ink); font-weight: 600; }}
.apb-num {{ font-family: Consolas, "Cascadia Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }}
.apb-next-title {{ font-size: 15px; font-weight: 600; color: var(--apb-ink); margin: 0; }}
.apb-next-body {{ font-size: 14px; color: var(--apb-body); margin: 2px 0 0; }}
[data-testid="stMarkdownContainer"] table.apb-table {{ width: 100% !important; display: table !important; border-collapse: collapse; font-size: 14px; }}
[data-testid="stMarkdownContainer"] table.apb-table th {{ background: var(--apb-band); color: var(--apb-ink); font-weight: 600; font-size: 13px; text-align: left; padding: 7px 12px; border: 0; }}
[data-testid="stMarkdownContainer"] table.apb-table td {{ padding: 7px 12px; border: 0; border-bottom: 1px solid var(--apb-hair); vertical-align: top; color: var(--apb-ink); }}
[data-testid="stMarkdownContainer"] table.apb-table .r {{ text-align: right; }}
[data-testid="stMarkdownContainer"] table.apb-table .sub {{ display: block; font-size: 12px; color: var(--apb-label); }}
.apb-scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
[class*="wisumgrid"] {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
[class*="wisumgrid"] [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; min-width: 760px; }}
[class*="wisumgrid"] [data-testid="stColumn"] {{ min-width: 0 !important; }}
/* スマホ（縦）: 余白と文字を詰め、横にはみ出すものは各部品の中でスクロールさせる */
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding-top: 3.25rem !important; padding-left: 1rem !important;
    padding-right: 1rem !important; padding-bottom: 4rem !important; }}
  h1 {{ font-size: 20px !important; }}
  h2 {{ font-size: 16px !important; }}
  .apb-lead {{ font-size: 13px; }}
  .apb-kv {{ gap: 2px 16px; font-size: 12px; }}
  .apb-steps {{ gap: 4px; }}
  .apb-steps span.s {{ font-size: 12px; padding: 3px 8px; }}
  .apb-steps span.sep {{ display: none; }}
  [data-testid="stMarkdownContainer"] table.apb-table {{ font-size: 12px; }}
  [data-testid="stMarkdownContainer"] table.apb-table th,
  [data-testid="stMarkdownContainer"] table.apb-table td {{ padding: 6px 8px; }}
  [data-testid="stMarkdownContainer"] table.apb-table th {{ white-space: nowrap; }}
  [data-testid="stMarkdownContainer"] table.apb-table .opt {{ display: none; }}
}}
/* 下のスクリプト用の高さ 0 の iframe が余白を作らないように */
[data-testid="stElementContainer"]:has(> iframe[height="0"]),
[data-testid="stElementContainer"]:has(iframe[title="apb_mobile"]) {{ display: none; }}
</style>
"""

# スマホ・タブレット（幅 768px 未満）では、メニューを選んだ後もサイドバーが画面を覆ったままになるため閉じる
_MOBILE_JS = """
<script>
(function () {
  const w = window.parent, doc = w.document;
  if (w.__apbMobileNav) return;
  w.__apbMobileNav = true;
  doc.addEventListener("click", function (e) {
    if (!e.target.closest('[data-testid="stSidebarNavLink"]') || w.innerWidth >= 768) return;
    setTimeout(function () {
      const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
      const sb = doc.querySelector('[data-testid="stSidebar"]');
      if (btn && sb && sb.getAttribute("data-collapsed") === "false") btn.click();
    }, 200);
  }, true);
})();
</script>
"""


def theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def apply_theme():
    pal = _PALETTE[theme_type()]
    css_vars = " ".join(f"--apb-{k}: {v};" for k, v in pal.items())
    st.markdown(_CSS.format(vars=css_vars), unsafe_allow_html=True)
    components.html(_MOBILE_JS, height=0)


def page_header(title: str, lead: str = "", eyebrow: str = "", crumbs: list | None = None):
    if crumbs:
        parts = [html.escape(c) for c in crumbs[:-1]] + [f"<b>{html.escape(crumbs[-1])}</b>"]
        st.markdown(f"<div class='apb-crumb'>{' ＞ '.join(parts)}</div>", unsafe_allow_html=True)
    if eyebrow:
        st.markdown(f"<div class='apb-eyebrow'>{html.escape(eyebrow)}</div>", unsafe_allow_html=True)
    st.title(title, anchor=False)
    if lead:
        st.markdown(f"<p class='apb-lead'>{html.escape(lead)}</p>", unsafe_allow_html=True)


def chip(text: str, kind: str = "gray") -> str:
    """状態チップの HTML（kind: ok / warn / ng / blue / gray）"""
    return f"<span class='apb-chip {kind}'>{html.escape(text)}</span>"


def stepper(stages: list, done: dict, current: str | None):
    items = []
    for i, (key, label) in enumerate(stages, start=1):
        cls = "done" if done.get(key) else ("cur" if key == current else "")
        mark = "完了" if done.get(key) else ("次" if key == current else "")
        tail = f"　{mark}" if mark else ""
        items.append(f"<span class='s {cls}'>{i}. {html.escape(label)}{tail}</span>")
    st.markdown("<div class='apb-steps'>" + "<span class='sep'>―</span>".join(items) + "</div>",
                unsafe_allow_html=True)


# ── 点検画面の共通部品 ───────────────────────────────────────

# 判定色（ライト・ダークどちらの背景でも読める中間色）
STATUS_COLOR = {"ng": "#e5534b", "warn": "#d29922", "ok": "#3fb950", "nobase": "#8b949e", "missing": "#8b949e"}
BAND_FILL = "rgba(84, 140, 255, 0.07)"
REF_LINE = "#548cff"


def status_cell_style(label: str) -> str:
    """st.dataframe の判定列を色付けする（Styler.map 用）"""
    return {
        "異常": "background-color: rgba(229, 83, 75, 0.22)",
        "注意": "background-color: rgba(210, 153, 34, 0.22)",
        "正常": "background-color: rgba(63, 185, 80, 0.16)",
    }.get(label, "")


def rule_popover():
    """判定ルールの表示と調整（評価・ステップ詳細・推移で共通）"""
    from core.evaluate import DEFAULTS, rule_text

    ss = st.session_state
    with st.popover("判定ルール", icon=":material/rule:"):
        st.caption(rule_text())
        ss.setdefault("eval_nsig", DEFAULTS["nsig"])
        ss.setdefault("eval_min_tol", DEFAULTS["min_tol"])
        ss.setdefault("eval_ng_rate", DEFAULTS["ng_rate"])
        st.number_input("許容幅の σ 倍数 N", min_value=1.0, max_value=6.0, step=0.5, key="eval_nsig")
        st.number_input("最小許容幅 [ms]", min_value=0.0, max_value=100.0, step=1.0, key="eval_min_tol")
        st.number_input("異常とする NG率 [%]", min_value=1.0, max_value=100.0, step=1.0, key="eval_ng_rate")


def tolerance_band(fig, ref: float, tol: float, horizontal: bool = True, **rowcol):
    """基準値（点線）と許容幅（薄い帯＋細い境界線）を図に描く"""
    lo, hi = ref - tol, ref + tol
    if horizontal:
        fig.add_hrect(y0=lo, y1=hi, fillcolor=BAND_FILL, line_width=0, layer="below", **rowcol)
        for y in (lo, hi):
            fig.add_hline(y=y, line_color=REF_LINE, line_width=0.8, opacity=0.5, **rowcol)
        fig.add_hline(y=ref, line_color=REF_LINE, line_dash="dot", annotation_text="基準",
                      annotation_position="right", **rowcol)
    else:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=BAND_FILL, line_width=0, layer="below", **rowcol)
        for x in (lo, hi):
            fig.add_vline(x=x, line_color=REF_LINE, line_width=0.8, opacity=0.5, **rowcol)
        fig.add_vline(x=ref, line_color=REF_LINE, line_dash="dot", annotation_text="基準",
                      annotation_position="top", **rowcol)


def numeric(df, cols: list):
    """None を NaN にして数値列にそろえる（表示で「—」になるように）"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
