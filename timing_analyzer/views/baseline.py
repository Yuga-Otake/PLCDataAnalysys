"""
views/baseline.py — 基準値

正常時の各ステップのタイミングを基準値として登録する。
旧画面の「基準値登録・編集」「複数 CSV 一括算出」「ウィザード手順 3」をこの 1 画面にまとめる。
"""
import pandas as pd
import streamlit as st

from core.analysis import MODE_LABELS, baseline_meta, compute_baseline
from core.profile import cycle_health, wave_candidates
from core.bundle import bundle_file_name, bundle_maker
from core.project import build_settings_json, mark_saved, settings_file_name
from core.state import pk
from core.ui import chip, page_header
from core.wave_inspect import registered_points

ss = st.session_state
page_header(
    "基準値",
    "正常に動いていたときの各ステップのタイミングを記録します。点検では、この基準値からの遅れやばらつきを見ます。",
    eyebrow="準備",
)
for msg in ss.pop("_bl_msgs", []):
    st.toast(msg)

processes = ss.get("processes", {})
store = ss.get("csv_store", {})
if not processes or not store:
    st.info("先にデータを読み込み、工程とステップを設定してください。")
    if st.button("工程とステップへ", type="primary"):
        st.switch_page("views/process.py")
    st.stop()

names = list(processes)
if ss.get("sel_process") not in names:
    ss["sel_process"] = names[0]
pname = st.selectbox("工程", names, key="sel_process")
steps = ss.get(pk(pname, "steps_list"), [])
trigger = ss.get(pk(pname, "trigger"), "")
edge = ss.get(pk(pname, "edge"), "RISE")
current = ss.get(pk(pname, "baseline"), {})
meta = ss.get(pk(pname, "baseline_meta"), {})

if not steps:
    st.info(f"「{pname}」にはまだステップがありません。")
    if st.button("ステップを設定する", type="primary"):
        st.switch_page("views/process.py")
    st.stop()


def _rows(bl: dict) -> list:
    rows = []
    for s in steps:
        b = bl.get(s["name"])
        if not b:
            rows.append({"ステップ": s["name"], "測り方": MODE_LABELS.get(s.get("mode"), ""),
                         "開始 [ms]": None, "長さ [ms]": None, "ばらつき σ [ms]": None})
        elif b["mode"] == "single":
            rows.append({"ステップ": s["name"], "測り方": MODE_LABELS["single"], "開始 [ms]": b["ref_ms"],
                         "長さ [ms]": None, "ばらつき σ [ms]": b.get("std_ms")})
        else:
            rows.append({"ステップ": s["name"], "測り方": MODE_LABELS.get(b["mode"], ""),
                         "開始 [ms]": b.get("ref_start_ms"), "長さ [ms]": b.get("ref_dur_ms"),
                         "ばらつき σ [ms]": b.get("std_dur_ms")})
    return rows


# ── 現在の基準値 ─────────────────────────────────────────────
st.subheader("登録済みの基準値", anchor=False)
with st.container(border=True):
    if current:
        n_reg = sum(1 for s in steps if s["name"] in current)
        status = chip("登録済み", "ok") if n_reg == len(steps) else chip(f"{len(steps) - n_reg} ステップ未登録", "warn")
        st.markdown(
            f"{status}　<span class='apb-kv'><span>算出元 <b>{meta.get('source', '—')}</b></span>"
            f"<span>サイクル数 <b class='apb-num'>{meta.get('n_cycles', '—')}</b></span>"
            f"<span>登録日時 <b>{meta.get('created_at', '—')}</b></span></span>",
            unsafe_allow_html=True,
        )
        st.caption("値を直接書き換えて「修正を保存」すると、その値が基準値になります。")
        edited = st.data_editor(
            pd.DataFrame(_rows(current)), hide_index=True, width="stretch",
            key=f"_bl_edit_{pname}_{meta.get('created_at', '')}",
            disabled=["ステップ", "測り方", "ばらつき σ [ms]"],
            column_config={c: st.column_config.NumberColumn(format="%.1f", min_value=0.0)
                           for c in ("開始 [ms]", "長さ [ms]", "ばらつき σ [ms]")},
        )
        e1, e2, _ = st.columns([1, 1, 3])
        if e1.button("修正を保存", key=f"_bl_save_edit_{pname}"):
            new_bl = dict(current)
            for s, row in zip(steps, edited.to_dict("records")):
                b = new_bl.get(s["name"])
                if not b or pd.isna(row["開始 [ms]"]):
                    continue
                if b["mode"] == "single":
                    new_bl[s["name"]] = {**b, "ref_ms": float(row["開始 [ms]"])}
                elif not pd.isna(row["長さ [ms]"]):
                    new_bl[s["name"]] = {**b, "ref_start_ms": float(row["開始 [ms]"]),
                                         "ref_dur_ms": float(row["長さ [ms]"])}
            ss[pk(pname, "baseline")] = new_bl
            ss[pk(pname, "baseline_meta")] = {**meta, "source": f"{meta.get('source', '')}（手動修正あり）"}
            ss["_bl_msgs"] = ["基準値の修正を保存しました"]
            st.rerun()
        if e2.button("基準値を消去", key=f"_bl_clear_{pname}"):
            ss.pop(pk(pname, "baseline"), None)
            ss.pop(pk(pname, "baseline_meta"), None)
            ss["_bl_msgs"] = ["基準値を消去しました"]
            st.rerun()
    else:
        st.markdown(chip("未登録", "warn") + "　下で基準にするデータを選び、登録してください。",
                    unsafe_allow_html=True)

# ── 基準値を算出して登録 ─────────────────────────────────────
st.subheader("基準値を算出する", anchor=False)
with st.container(border=True):
    keys = list(store)
    ref_keys = [k for k in keys if store[k].get("is_ref")]
    sel_key = f"_bl_src_{pname}"
    if sel_key not in ss or not all(k in keys for k in ss[sel_key]):
        ss[sel_key] = ref_keys or keys[:1]
    c1, c2 = st.columns([3, 1.3])
    chosen = c1.multiselect(
        "基準にするデータ", keys, key=sel_key, placeholder="データを選んでください",
        format_func=lambda k: ("基準｜" if store[k].get("is_ref") else "比較｜") + store[k]["label"],
        help="正常に動いていたときのデータを選びます。複数選ぶと全サイクルをまとめて算出します",
    )
    method = c2.radio("算出方法", ["mean", "min"], key=f"_bl_method_{pname}",
                      format_func=lambda m: {"mean": "平均値", "min": "最小値（最速）"}[m])
    if any(not store[k].get("is_ref") for k in chosen):
        st.caption("比較データを含めています。不具合が出ている時期のデータが混ざると、基準値がずれます。")
    if not chosen:
        st.caption("データを 1 件以上選んでください。")
    else:
        # 事前診断で見つけた外れサイクル（停止など）を除く
        excl = [set(cycle_health(store[k]["df"], trigger, edge)["outliers"]) for k in chosen]
        n_excl = sum(len(e) for e in excl)
        drop = st.checkbox(f"外れサイクル（停止など）を除いて算出する（{n_excl} サイクル）", value=True,
                           key=f"_bl_drop_{pname}", disabled=n_excl == 0,
                           help="サイクル長が他と大きく違うサイクルは、正常な動作を表していない可能性があります")
        if n_excl and drop:
            detail = "、".join(f"{store[k]['label']}: " + "・".join(f"#{c}" for c in sorted(e))
                               for k, e in zip(chosen, excl) if e)
            st.caption(f"除外するサイクル — {detail}")
        with st.spinner("算出中…"):
            preview, n_cyc = compute_baseline([store[k]["df"] for k in chosen], trigger, edge, steps, method,
                                              exclude=excl if drop else None)
        if not preview:
            st.warning("選んだデータではステップを検出できませんでした。サイクル開始信号とステップを確認してください。")
        else:
            st.markdown(f"<div class='apb-kv'><span>サイクル数 <b class='apb-num'>{n_cyc}</b></span>"
                        f"<span>データ <b class='apb-num'>{len(chosen)}</b> 件</span></div>",
                        unsafe_allow_html=True)
            prev_df = pd.DataFrame(_rows(preview))
            if current:
                prev_df["登録済みとの差 [ms]"] = [
                    None if (s["name"] not in current or s["name"] not in preview) else round(
                        (preview[s["name"]].get("ref_ms", 0) + preview[s["name"]].get("ref_start_ms", 0)
                         + preview[s["name"]].get("ref_dur_ms", 0))
                        - (current[s["name"]].get("ref_ms", 0) + current[s["name"]].get("ref_start_ms", 0)
                           + current[s["name"]].get("ref_dur_ms", 0)), 1)
                    for s in steps]
            st.dataframe(prev_df, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.1f")
                                        for c in prev_df.columns if "[ms]" in c})
            label = "この内容で登録" if not current else "この内容で登録し直す"
            if st.button(label, type="primary", key=f"_bl_register_{pname}"):
                ss[pk(pname, "baseline")] = preview
                src = "、".join(store[k]["label"] for k in chosen)
                ss[pk(pname, "baseline_meta")] = baseline_meta(
                    f"{src}の{'平均値' if method == 'mean' else '最小値'}"
                    + (f"（外れ {n_excl} サイクル除外）" if n_excl and drop else ""), n_cyc)
                ss["_bl_msgs"] = [f"「{pname}」の基準値を登録しました"]
                st.rerun()

# ── 次へ ─────────────────────────────────────────────────────
if current:
    st.divider()
    # 基準値を登録したところで、この工程のアナログ波形にも検出点の候補があれば知らせる（任意の詳細設定）
    ref = next((e for e in store.values() if e.get("is_ref")), None)
    if ref:
        procs = {n: {"trigger_col": ss.get(pk(n, "trigger"), ""), "edge": ss.get(pk(n, "edge"), "RISE")}
                 for n in names}
        hyp = {v: h for v, h in wave_candidates(ref, procs).items() if h["process"] == pname}
        pending = [(v, c) for v, h in hyp.items() for c in h["cands"]
                   if (c["dtype"], c["name"]) not in registered_points(v)]
        if pending:
            with st.container(border=True):
                eg = "、".join(f"{v.split('.')[-1]} {c['name']}" for v, c in pending[:3])
                st.markdown(
                    f"<p class='apb-next-title'>アナログ波形も点検しますか（任意）</p>"
                    f"<p class='apb-next-body'>「{pname}」にはアナログ波形が {len(hyp)} 本あり、"
                    f"検出点の候補が {len(pending)} 件あります（{eg} など）。"
                    "ピークの値や立ち上がりの時刻も推移で見たい場合は、候補から選んで登録できます。"
                    "登録した検出点も設定ファイルに保存されます。</p>", unsafe_allow_html=True)
                if st.button("波形検査の候補を見る", key="_bl_wave", icon=":material/show_chart:"):
                    st.switch_page("views/wave.py")
    others = [n for n in names if not ss.get(pk(n, "baseline"))]
    n1, n2, n3, _ = st.columns([1.3, 1.5, 1, 1.2])
    n1.download_button("データごと保存", data=bundle_maker(False), file_name=bundle_file_name(),
                       mime="application/zip", on_click=mark_saved, type="primary",
                       icon=":material/inventory_2:", key="_bl_dl_zip", width="stretch",
                       help="基準データと設定を 1 つのファイル（ZIP）にまとめます。読み込めばデータごと元に戻ります")
    n2.download_button("設定だけ保存（JSON）", data=build_settings_json(), file_name=settings_file_name(),
                       mime="application/json", on_click=mark_saved,
                       icon=":material/download:", key="_bl_dl", width="stretch")
    if n3.button("ホームへ戻る", key="_bl_home", width="stretch"):
        st.switch_page("views/home.py")
    if others:
        st.caption("基準値が未登録の工程: " + "、".join(others))
