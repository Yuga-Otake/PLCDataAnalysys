"""
core/ui.py — 画面共通の見た目と部品

ライトは青と灰色のシンプルな配色、ダークは Sysmac Studio 風（Visual Studio 系の濃灰＋青）。
配色の大枠は .streamlit/config.toml の [theme.light] / [theme.dark]、
ここではチップ・段階表示・表など Streamlit 標準にない部品の色をモードごとに当てる。
"""
import html

import streamlit as st

LEGACY_PAGE = "legacy_app.py"

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
</style>
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


def open_legacy(view_mode: str | None = None, tab: str | None = None,
                wizard_for: str | None = None, dialog: tuple | None = None):
    """旧画面を指定の状態で開く（段階 2・3 で新画面に置き換わるまでの橋渡し）。

    dialog: ("add_process",) または ("bulk_baseline", 工程名)
    """
    ss = st.session_state
    if view_mode:
        ss["view_mode"] = view_mode
    if tab:
        ss["main_tab"] = tab
    if wizard_for:
        ss["wizard_step"] = 1
        ss["wizard_pname"] = wizard_for
    if dialog:
        ss["_legacy_open"] = dialog
    st.switch_page(LEGACY_PAGE)


def planned_page(title: str, lead: str, will_have: list, legacy_where: str,
                 legacy_kwargs: dict, eyebrow: str):
    """段階 2 以降で作る画面の仮ページ。目的と、今どこで操作できるかを示す。"""
    page_header(title, lead, eyebrow=eyebrow)
    with st.container(border=True):
        st.markdown(f"{chip('移行中', 'blue')}　この画面は次の段階で作成します。", unsafe_allow_html=True)
        st.markdown("**この画面でできるようになること**")
        st.markdown("\n".join(f"- {w}" for w in will_have))
        st.markdown(f"現在は **旧画面の「{legacy_where}」** で同じ操作ができます。")
        if st.button("旧画面で開く", type="primary", key=f"_legacy_go_{title}",
                     icon=":material/open_in_new:"):
            open_legacy(**legacy_kwargs)
