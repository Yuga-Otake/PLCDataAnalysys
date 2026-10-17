"""
core/ui.py — 画面共通の見た目と部品

OMRON IA Catalog Design System の製品 UI 規定に合わせる：
主色は OMRON Blue（#0068b8）1 色、1px の細罫（#dadbde）、操作部品の角丸 6px・パネル 8px、
状態は「色＋文字」で示し、絵文字は使わない。配色の大枠は .streamlit/config.toml の [theme]。
"""
import html

import streamlit as st

LEGACY_PAGE = "legacy_app.py"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=Roboto:wght@400;500;700&family=Roboto+Mono:wght@400;500&display=swap');
:root {
  --apb-blue: #0068b8; --apb-blue-deep: #0060b0; --apb-blue-chip: #c8e8f8; --apb-blue-tint: #d6e9f0;
  --apb-ink: #1d1d1b; --apb-body: #58585a; --apb-label: #808990; --apb-hair: #dadbde; --apb-band: #eaeef0;
  --apb-ok: #22a745; --apb-ok-pale: #e4f4e5; --apb-warn: #b85a00; --apb-warn-pale: #fbe3c7;
  --apb-ng: #d0121b; --apb-ng-pale: #f8d0d0;
}
h1, h2, h3, h4 { letter-spacing: .04em; color: var(--apb-ink); }
h1 { font-size: 26px !important; font-weight: 700 !important; padding-bottom: 0 !important; }
h2 { font-size: 18px !important; font-weight: 700 !important; }
h3 { font-size: 15px !important; font-weight: 700 !important; }
.apb-eyebrow { font-size: 12px; color: var(--apb-label); letter-spacing: .12em; margin-bottom: -6px; }
.apb-lead { color: var(--apb-body); font-size: 14px; max-width: 72ch; margin: 2px 0 6px; }
.apb-crumb { font-size: 12px; color: var(--apb-label); }
.apb-crumb b { color: var(--apb-ink); font-weight: 500; }
.apb-chip { display: inline-block; font-size: 12px; line-height: 1; padding: 5px 9px; border-radius: 999px;
  font-weight: 500; white-space: nowrap; letter-spacing: .04em; }
.apb-chip.ok { background: var(--apb-ok-pale); color: var(--apb-ok); }
.apb-chip.warn { background: var(--apb-warn-pale); color: var(--apb-warn); }
.apb-chip.ng { background: var(--apb-ng-pale); color: var(--apb-ng); }
.apb-chip.blue { background: var(--apb-blue-chip); color: var(--apb-blue-deep); }
.apb-chip.gray { background: var(--apb-band); color: var(--apb-body); }
.apb-steps { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 2px 0 4px; }
.apb-steps span.s { font-size: 13px; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--apb-hair);
  color: var(--apb-label); white-space: nowrap; }
.apb-steps span.s.done { background: var(--apb-ok-pale); color: var(--apb-ok); border-color: transparent; }
.apb-steps span.s.cur { border-color: var(--apb-blue); color: var(--apb-blue-deep); font-weight: 700;
  box-shadow: 0 0 0 2px var(--apb-blue-tint); }
.apb-steps span.sep { color: var(--apb-hair); }
.apb-kv { display: flex; flex-wrap: wrap; gap: 4px 28px; font-size: 13px; color: var(--apb-label); }
.apb-kv b { color: var(--apb-ink); font-weight: 500; font-family: "Roboto", "Noto Sans JP", sans-serif; }
.apb-num { font-family: "Roboto Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }
.apb-next-title { font-size: 16px; font-weight: 700; color: var(--apb-ink); margin: 0; }
.apb-next-body { font-size: 14px; color: var(--apb-body); margin: 2px 0 0; }
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] { letter-spacing: .14em; }
[data-testid="stMarkdownContainer"] table.apb-table { width: 100% !important; display: table !important; border-collapse: collapse; font-size: 14px; }
[data-testid="stMarkdownContainer"] table.apb-table th { background: var(--apb-band); color: var(--apb-ink); font-weight: 700; font-size: 13px; text-align: left; padding: 8px 12px; border: 0; }
[data-testid="stMarkdownContainer"] table.apb-table td { padding: 8px 12px; border: 0; border-bottom: 1px solid var(--apb-hair); vertical-align: top; }
[data-testid="stMarkdownContainer"] table.apb-table .r { text-align: right; }
[data-testid="stMarkdownContainer"] table.apb-table .sub { display: block; font-size: 12px; color: var(--apb-label); }
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


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
