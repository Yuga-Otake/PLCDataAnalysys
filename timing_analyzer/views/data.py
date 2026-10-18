"""
views/data.py — データ

基準データ（正常時）と比較データの読み込み・一覧、フォルダ読み込み、
設定ファイルの保存・読み込みをこの画面に集める。
"""
import json

import pandas as pd
import streamlit as st

from core.data import (add_uploaded, apply_settings_json, df_summary, remove_csv,
                       save_to_folder, scan_folder, set_use_sample, store)
from core.bundle import bundle_file_name, bundle_maker, is_bundle, load_bundle
from core.project import build_settings_json, mark_saved, settings_file_name
from core.profile import KIND_LABELS, compare_consistency, sampling_profile, variable_profile
from core.state import compare_keys, pk, ref_entry
from core.ui import chip, page_header

page_header(
    "データ",
    "設備が正常に動いていたときの基準データと、点検したい時期の比較データを読み込みます。"
    "基準データと設定（工程・ステップ・基準値）はプロジェクトファイル（ZIP）にまとめて保存でき、次回はそれを読み込むだけで再開できます。",
    eyebrow="準備",
)
ss = st.session_state

# 処理結果のメッセージは rerun 後に表示する
for msg in ss.pop("_data_msgs", []):
    st.toast(msg)


def _rerun_with(*msgs):
    ss["_data_msgs"] = list(msgs)
    st.rerun()


# ── 基準データ ───────────────────────────────────────────────
st.subheader("基準データ", anchor=False)
st.caption("設備立ち上げ時など、正常に動いていたときの CSV です。2〜10 サイクル分あれば十分です。"
           "工程・ステップの定義と基準値の算出に使います。")
ref_key, ref = ref_entry()
with st.container(border=True):
    if ref:
        info = df_summary(ref)
        st.markdown(
            f"<div class='apb-kv'><span><b>{ref['label']}</b></span>"
            f"<span>行数 <b class='apb-num'>{info['rows']:,}</b></span>"
            f"<span>期間 <b>{info['period']}</b></span>"
            f"<span>Bool 変数 <b class='apb-num'>{info['n_bool']}</b>"
            f"　アナログ変数 <b class='apb-num'>{info['n_num']}</b></span></div>",
            unsafe_allow_html=True,
        )
        # 事前診断: 記録の質と、実際に動いている変数
        sp = sampling_profile(ref["df"])
        prof = variable_profile(ref["df"], tuple(ref["col_types"].items()))
        n_const = int((prof["kind"] == "constant").sum())
        kinds = prof["kind"].value_counts()
        moving = "・".join(f"{KIND_LABELS[k]} {kinds[k]}" for k in ("bool", "state", "counter", "analog") if k in kinds)
        st.markdown(
            f"{chip('記録 ' + {'ok': '良好', 'warn': '注意', 'ng': '問題あり'}[sp['status']], {'ok': 'ok', 'warn': 'warn', 'ng': 'ng'}[sp['status']])}"
            f"　<span class='apb-crumb'>{sp['message']}</span><br>"
            f"{chip(f'動いている変数 {len(prof) - n_const}', 'blue')}"
            f"　<span class='apb-crumb'>{moving}"
            + (f"。ずっと一定の {n_const} 個はステップの候補・検索から外します" if n_const else "") + "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(chip("未読込", "warn") + "　基準データを読み込んでください。", unsafe_allow_html=True)

    up_col, sample_col = st.columns([3, 2], vertical_alignment="bottom")
    with up_col:
        ref_up = st.file_uploader("基準データを読み込む（CSV）", type=["csv"], key="_data_ref_up")
    with sample_col:
        # key を付けない: 他の画面での切り替えとずれないよう、毎回 use_sample の値から表示する
        use_sample = st.toggle("サンプルデータを使う", value=ss.get("use_sample", True),
                               help="オフにすると、サンプルの基準データ・比較データを取り除きます")
    if use_sample != ss.get("use_sample", True):
        set_use_sample(use_sample)
        _rerun_with("サンプルデータを使います" if use_sample else "サンプルデータを外しました")
    if ref_up is not None and ref_up.file_id != ss.get("_data_ref_loaded"):
        try:
            add_uploaded(ref_up, is_ref=True)
            ss["_data_ref_loaded"] = ref_up.file_id
            _rerun_with(f"基準データを読み込みました: {ref_up.name}")
        except Exception as e:
            st.error(f"{ref_up.name} を読み込めませんでした。APB の CSV 形式か確認してください。（{e}）")

# ── 比較データ ───────────────────────────────────────────────
st.subheader("比較データ", anchor=False)
st.caption("点検したい時期の CSV です。時期の違うものを複数入れると、推移で経時変化を確認できます。"
           "ラベルは推移グラフの横軸に使われます。")
cmp = compare_keys()
with st.container(border=True):
    cmp_up = st.file_uploader("比較データを追加（CSV・複数可）", type=["csv"],
                              accept_multiple_files=True, key="_data_cmp_up")
    # 一度取り込んだファイルは覚えておく（「外す」の後にアップロード欄から再追加しないため）
    seen = ss.setdefault("_data_cmp_seen", set())
    new_files = [f for f in (cmp_up or []) if f.file_id not in seen]
    seen.update(f.file_id for f in new_files)
    if new_files:
        done, errors = [], []
        for f in new_files:
            try:
                add_uploaded(f, is_ref=False)
                done.append(f.name)
            except Exception as e:
                errors.append(f"{f.name}（{e}）")
        if errors:
            st.error("読み込めなかったファイル: " + "、".join(errors))
        if done:
            _rerun_with(f"比較データを {len(done)} 件追加しました")

    if not cmp:
        st.caption("比較データはまだありません。")
    else:
        rows = []
        used = set()
        for p in ss.get("processes", {}):
            used.add(ss.get(pk(p, "trigger")))
            for st_ in ss.get(pk(p, "steps_list"), []):
                used.update(st_.get(x) for x in ("variable", "start_var", "end_var"))
        for k in cmp:
            e = store()[k]
            info = df_summary(e)
            cons = compare_consistency(ref, e, used) if ref else {"status": "ok", "message": "—"}
            rows.append({"外す": False, "ラベル": e["label"], "ファイル": k.replace("\\", "/").split("/")[-1],
                         "基準との整合": {"ok": "", "warn": "注意: ", "ng": "問題: "}[cons["status"]] + cons["message"],
                         "行数": info["rows"], "期間": info["period"],
                         "読み込み元": {"folder": "フォルダ", "upload": "アップロード"}.get(e.get("source"), "サンプル")})
        edited = st.data_editor(
            pd.DataFrame(rows), hide_index=True, width="stretch", key=f"_data_cmp_table_{len(cmp)}",
            disabled=["ファイル", "基準との整合", "行数", "期間", "読み込み元"],
            column_config={"外す": st.column_config.CheckboxColumn(width="small"),
                           "行数": st.column_config.NumberColumn(format="%d")},
        )
        # ラベルの変更はその場で反映
        for k, new_label in zip(cmp, edited["ラベル"]):
            if new_label and new_label != store()[k]["label"]:
                store()[k]["label"] = new_label
        to_remove = [k for k, flag in zip(cmp, edited["外す"]) if flag]
        if st.button(f"選んだ {len(to_remove)} 件を外す", disabled=not to_remove, key="_data_remove"):
            for k in to_remove:
                remove_csv(k)
            _rerun_with(f"比較データを {len(to_remove)} 件外しました")

# ── 保存・読み込み ───────────────────────────────────────────
st.subheader("保存・読み込み", anchor=False)
st.caption("プロジェクトファイル（ZIP）は、基準データ（CSV）と設定ファイル（JSON：工程・ステップ・基準値・波形検査条件）を"
           "1 つにまとめたものです。これを読み込めば、点検したい時期の CSV を追加するだけで点検を再開できます。")
with st.container(border=True):
    save_col, load_col = st.columns(2)
    with save_col:
        st.markdown("**保存**")
        if ss.get("processes"):
            st.download_button("基準データ＋設定を保存（ZIP）", data=bundle_maker(), file_name=bundle_file_name(),
                               mime="application/zip", on_click=mark_saved, icon=":material/inventory_2:",
                               type="primary", key="_data_save_bundle", width="stretch")
            st.download_button("設定ファイルだけ保存（JSON）", data=build_settings_json(),
                               file_name=settings_file_name(), mime="application/json", on_click=mark_saved,
                               icon=":material/download:", type="tertiary", key="_data_save_json")
        else:
            st.caption("工程を登録すると保存できます。")
    with load_col:
        st.markdown("**読み込み**")
        cfg_up = st.file_uploader("設定ファイル（JSON）またはプロジェクトファイル（ZIP）", type=["json", "zip"],
                                  key="_data_cfg_up", label_visibility="collapsed")
        st.caption("プロジェクトファイルを読み込むと、基準データと設定が置き換わります（追加済みの比較データはそのまま）。")
        if cfg_up is not None and cfg_up.file_id != ss.get("_data_cfg_loaded"):
            try:
                with st.spinner("読み込んでいます…"):
                    msg = load_bundle(cfg_up) if is_bundle(cfg_up.name) else apply_settings_json(json.load(cfg_up))
                ss["_data_cfg_loaded"] = cfg_up.file_id
                _rerun_with(msg)
            except Exception as e:
                st.error(f"{cfg_up.name} を読み込めませんでした。（{e}）")

# ── フォルダ ─────────────────────────────────────────────────
with st.expander("フォルダから読み込む・フォルダに保存する（ローカルで実行している場合）"):
    st.caption("ルートフォルダの下に、基準・比較・設定のサブフォルダを置いて使います。"
               "基準は最新の 1 件、比較は新しい順に指定件数を読み込みます。")
    root = st.text_input("ルートフォルダ", key="folder_root", placeholder=r"C:\APB\logs")
    c1, c2, c3, c4 = st.columns(4)
    sub_ref = c1.text_input("基準", value="baseline", key="folder_sub_ref")
    sub_cmp = c2.text_input("比較", value="data", key="folder_sub_cmp")
    sub_cfg = c3.text_input("設定", value="config", key="folder_sub_cfg")
    max_cmp = c4.number_input("比較の最大件数", min_value=1, max_value=500, value=20, step=10,
                              key="folder_max_cmp")
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("読み込む", disabled=not root, key="_data_folder_scan", icon=":material/folder_open:"):
        try:
            _rerun_with(*scan_folder(root, sub_ref, sub_cmp, sub_cfg, int(max_cmp)))
        except Exception as e:
            st.error(str(e))
    if b2.button("保存する", disabled=not root, key="_data_folder_save", icon=":material/save:"):
        try:
            st.success(save_to_folder(root, sub_ref, sub_cmp, sub_cfg))
        except Exception as e:
            st.error(f"保存できませんでした。（{e}）")
    jsons = ss.get("folder_jsons_found", [])
    if len(jsons) > 1:
        sel = st.selectbox("適用する設定ファイル", jsons, key="_data_folder_json",
                           format_func=lambda p: p.replace("\\", "/").split("/")[-1])
        if st.button("この設定を適用", key="_data_folder_apply"):
            with open(sel, encoding="utf-8") as f:
                _rerun_with(apply_settings_json(json.load(f)))

# ── 次へ ─────────────────────────────────────────────────────
st.divider()
if ref:
    if st.button("工程とステップの設定へ", type="primary", icon=":material/arrow_forward:", key="_data_next"):
        st.switch_page("views/process.py")
