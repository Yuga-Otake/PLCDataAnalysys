"""
views/home.py — ホーム

主な利用者は生産技術（設備立ち上げ時の準備担当）。準備の進み具合と、
工程ごとの設定状況を最初に示し、次にやる操作を 1 つだけ大きく出す。
"""
import html

import streamlit as st

from core.evaluate import STATUS, compare_keys_by_time, evaluate_process
from core.project import build_settings_json, mark_saved, settings_file_name
from core.state import STAGES, compare_keys, ref_entry, workflow
from core.ui import chip, page_header, stepper
from core.wave_inspect import inspection_points, process_defs

page_header(
    "ホーム",
    "設備立ち上げ時のデータから工程とステップを定義し、正常時のタイミングを基準値として登録します。"
    "登録した基準値は、その後の点検で比較の基準になります。",
    eyebrow="APB タイミング解析",
)

ss = st.session_state
wf = workflow()
ref_key, ref = ref_entry()
cmp_keys = compare_keys()

# ── データの概要 ─────────────────────────────────────────────
ref_label = ref["label"] if ref else "未読込"
sample_note = "（サンプルデータ）" if ss.get("use_sample", True) else ""
st.markdown(
    f"<div class='apb-kv'><span>基準データ <b>{html.escape(ref_label)}</b></span>"
    f"<span>比較データ <b class='apb-num'>{len(cmp_keys)}</b> 件</span>"
    f"<span>登録済みの工程 <b class='apb-num'>{len(wf['rows'])}</b></span>"
    f"<span>{sample_note}</span></div>",
    unsafe_allow_html=True,
)

# ── 準備の進み具合 ───────────────────────────────────────────
st.subheader("準備の進み具合", anchor=False)
with st.container(border=True):
    stepper(STAGES, wf["done"], wf["current"])
    cur, target = wf["current"], wf["target"]
    text_col, btn_col = st.columns([3, 1], vertical_alignment="center")

    if cur == "data":
        text_col.markdown("<p class='apb-next-title'>立ち上げ時のデータを読み込む</p>"
                          "<p class='apb-next-body'>APB で記録した CSV のうち、設備が正常に動いていたときの"
                          " 2〜10 サイクル分を基準データとして読み込みます。</p>", unsafe_allow_html=True)
        if btn_col.button("データを読み込む", type="primary", width="stretch"):
            st.switch_page("views/data.py")
    elif cur == "process":
        text_col.markdown("<p class='apb-next-title'>工程を登録する</p>"
                          "<p class='apb-next-body'>変数名のまとまりと、周期的に立ち上がる信号から工程を提案します。"
                          "内容を確認して、ワンクリックで作成できます。</p>", unsafe_allow_html=True)
        if btn_col.button("工程を登録", type="primary", width="stretch"):
            st.switch_page("views/process.py")
    elif cur == "steps":
        text_col.markdown(f"<p class='apb-next-title'>「{target}」のステップを定義する</p>"
                          "<p class='apb-next-body'>時間を測りたい動作（シリンダ前進、センサ検知など）を選びます。"
                          "変数が変化する順番の仮説から選ぶか、名前で検索して追加します。</p>", unsafe_allow_html=True)
        if btn_col.button("ステップを定義", type="primary", width="stretch"):
            ss["sel_process"] = target
            st.switch_page("views/process.py")
    elif cur == "baseline":
        text_col.markdown(f"<p class='apb-next-title'>「{target}」の基準値を登録する</p>"
                          "<p class='apb-next-body'>基準データの各ステップの平均タイミングを、正常時の値として記録します。"
                          "点検ではこの値との差を見ます。</p>", unsafe_allow_html=True)
        if btn_col.button("基準値を登録", type="primary", width="stretch"):
            ss["sel_process"] = target
            st.switch_page("views/baseline.py")
    elif cur == "save":
        text_col.markdown("<p class='apb-next-title'>設定を保存する</p>"
                          "<p class='apb-next-body'>工程・ステップ・基準値を設定ファイル（JSON）に保存します。"
                          "次回からはこのファイルを読み込むだけで点検を始められます。</p>", unsafe_allow_html=True)
        btn_col.download_button("設定を保存", data=build_settings_json(), file_name=settings_file_name(),
                                mime="application/json", type="primary", width="stretch",
                                on_click=mark_saved, icon=":material/download:")
    else:
        text_col.markdown("<p class='apb-next-title'>準備は完了しています</p>"
                          "<p class='apb-next-body'>比較データを追加して、基準値からの遅れやばらつきを点検できます。</p>",
                          unsafe_allow_html=True)
        if btn_col.button("最新データを評価", type="primary", width="stretch"):
            st.switch_page("views/evaluate.py")

# ── 工程の一覧 ───────────────────────────────────────────────
st.subheader("工程", anchor=False)
_STATUS = {
    "steps":    ("ステップ未定義", "warn"),
    "baseline": ("基準値未登録", "warn"),
    "ready":    ("準備完了", "ok"),
}
if not wf["rows"]:
    st.caption("工程はまだ登録されていません。")
else:
    body = []
    n_wave = {}
    for it in inspection_points({n: (d["trigger"], d["edge"]) for n, d in process_defs().items()}):
        n_wave[it["process"]] = n_wave.get(it["process"], 0) + 1
    for r in wf["rows"]:
        label, kind = _STATUS[r["status"]]
        edge = "立ち上がり" if r["edge"] == "RISE" else "立ち下がり"
        body.append(
            f"<tr><td><b>{html.escape(r['name'])}</b></td>"
            f"<td>{html.escape(r['trigger'])}<span class='sub'>{edge}</span></td>"
            f"<td class='r apb-num opt'>{r['cycles']}</td><td class='r apb-num'>{r['steps']}</td>"
            f"<td class='r apb-num opt'>{r['baseline']}</td>"
            f"<td class='r apb-num'>{n_wave.get(r['name'], 0) or '−'}</td><td>{chip(label, kind)}</td></tr>"
        )
    st.markdown(
        "<div class='apb-scroll'><table class='apb-table'><thead><tr><th>工程</th><th>サイクル開始信号</th>"
        "<th class='r opt'>サイクル数</th><th class='r'>ステップ</th><th class='r opt'>基準値</th>"
        "<th class='r'>波形検査点</th>"
        "<th>状態</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div>",
        unsafe_allow_html=True,
    )
    st.caption("サイクル数は基準データで検出した数です。波形検査点は、サイクル開始信号がその工程のものを数えています。")

# ── 点検 ─────────────────────────────────────────────────────
st.subheader("点検", anchor=False)
if not wf["done"]["baseline"]:
    st.caption("基準値を登録すると、比較データとの差をここから確認できます。")
elif not cmp_keys:
    st.caption("比較データがまだありません。データ画面から点検したい時期の CSV を追加してください。")
else:
    latest_key = compare_keys_by_time()[-1]
    latest = ss["csv_store"][latest_key]
    with st.container(border=True):
        c1, c2 = st.columns([3, 1], vertical_alignment="center")
        c1.markdown(f"最新の比較データ **{html.escape(latest['label'])}** の判定"
                    f"（比較データは全 {len(cmp_keys)} 件）")
        if c2.button("最新データを評価", width="stretch", key="_home_eval"):
            ss["eval_target"] = latest_key
            st.switch_page("views/evaluate.py")
        items = []
        for p in [r["name"] for r in wf["rows"] if r["status"] == "ready"]:
            ev = evaluate_process(latest["df"], p)
            label, kind, _ = STATUS[ev["status"]]
            items.append(f"<span>{html.escape(p)}　{chip(label, kind)}"
                         f"<span class='apb-crumb'>　異常 {ev['n_ng_steps']}・注意 {ev['n_warn_steps']} ステップ</span></span>")
        st.markdown("<div class='apb-kv' style='gap:8px 32px'>" + "".join(items) + "</div>",
                    unsafe_allow_html=True)
