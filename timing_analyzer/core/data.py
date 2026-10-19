"""
core/data.py — 読み込んだ CSV（csv_store）と設定ファイルの管理

csv_store の 1 件:
  {"df", "col_types", "label", "is_ref", "source"}
  source: なし=サンプル ／ "upload"=データ画面 ／ "folder"=フォルダ読み込み
"""
import glob
import json
import os
import re

import streamlit as st

from analyzer import detect_bool_columns, load_csv
from core.project import build_settings_json
from core.state import CMP_SAMPLE, REF_SAMPLE, pk


def parse_csv_label(filename: str) -> str:
    """ファイル名から表示ラベルを作る（例: 20260418_am_line1_good.csv → 2026-04-18 朝番 L1 良品）"""
    stem = re.sub(r"\.csv$", "", filename, flags=re.IGNORECASE)
    d = re.search(r"(\d{4})(\d{2})(\d{2})", stem)
    date_str = f"{d.group(1)}-{d.group(2)}-{d.group(3)}" if d else ""
    rest = re.sub(r"\d{8}", "", stem)
    shift_str = ""
    for k, v in {"am": "朝番", "pm": "夕番", "night": "夜番", "day": "日勤"}.items():
        if re.search(rf"\b{k}\b", rest, re.IGNORECASE):
            shift_str = v
            rest = re.sub(rf"\b{k}\b", "", rest, flags=re.IGNORECASE)
            break
    m = re.search(r"line(\d+)", rest, re.IGNORECASE)
    line_str = f"L{m.group(1)}" if m else ""
    qual_str = ""
    for k, v in {"good": "良品", "ng": "NG", "ok": "良品", "fail": "NG"}.items():
        if re.search(rf"\b{k}\b", rest, re.IGNORECASE):
            qual_str = v
            break
    parts = [p for p in [date_str, shift_str, line_str, qual_str] if p]
    return " ".join(parts) if parts else stem


@st.cache_data(show_spinner=False)
def _load_path(path: str, mtime: float):
    df = load_csv(path)
    return df, detect_bool_columns(df)


def store() -> dict:
    return st.session_state.setdefault("csv_store", {})


def add_csv(key: str, df, col_types: dict, label: str, is_ref: bool, source: str):
    ss = st.session_state
    if is_ref:
        # 基準データは 1 件。以前の基準は比較データとして残す（基準サンプルは取り除く）
        for k, v in list(store().items()):
            if v.get("is_ref") and k != key:
                if k == REF_SAMPLE:
                    store().pop(k)
                else:
                    v["is_ref"] = False
        ss["ref_csv_key"] = key
        ss["active_csv"] = key
    store()[key] = {"df": df, "col_types": col_types, "label": label,
                    "is_ref": is_ref, "source": source}


def add_uploaded(file, is_ref: bool) -> str:
    """アップロードされた CSV を追加する。戻り値はキー（ファイル名）"""
    df = load_csv(file)
    add_csv(file.name, df, detect_bool_columns(df), parse_csv_label(file.name), is_ref, "upload")
    return file.name


def remove_csv(key: str):
    ss = st.session_state
    store().pop(key, None)
    if ss.get("ref_csv_key") == key:
        ss["ref_csv_key"] = None
    if ss.get("active_csv") == key:
        ss.pop("active_csv", None)


def set_use_sample(use: bool):
    """サンプルデータの使用を切り替える。外すときはサンプルの 2 件を取り除く"""
    ss = st.session_state
    ss["use_sample"] = use
    if not use:
        for k in (REF_SAMPLE, CMP_SAMPLE):
            remove_csv(k)


def df_summary(entry: dict) -> dict:
    df = entry["df"]
    ct = entry["col_types"]
    period = ""
    if "Timestamp" in df.columns and len(df):
        period = f"{str(df['Timestamp'].iloc[0])[:19]} 〜 {str(df['Timestamp'].iloc[-1])[:19]}"
    return {"rows": len(df), "period": period,
            "n_bool": sum(1 for t in ct.values() if t == "bool"),
            "n_num": sum(1 for t in ct.values() if t == "numeric")}


# ═══════════════════════════════════════════════════════════════
# 設定ファイル
# ═══════════════════════════════════════════════════════════════

def apply_settings_json(loaded: dict) -> str:
    """設定 JSON を読み込み、工程・ステップ・基準値・波形検出条件を復元する。"""
    ss = st.session_state
    if "processes" not in loaded:
        raise ValueError("設定ファイルの形式が違います（processes が見つかりません）")
    new_procs = {}
    for pname, p in loaded["processes"].items():
        new_procs[pname] = {"trigger_col": p.get("trigger", ""), "edge": p.get("edge", "RISE"),
                            "takt_target_ms": p.get("takt_target_ms", 0), "steps": p.get("steps", [])}
        ss[pk(pname, "trigger")] = p.get("trigger", "")
        ss[pk(pname, "edge")] = p.get("edge", "RISE")
        ss[pk(pname, "takt")] = int(p.get("takt_target_ms", 0))
        ss[pk(pname, "steps_list")] = p.get("steps", [])
        for key in ("baseline", "baseline_meta", "wv_baseline", "wv_xy_baseline"):
            if p.get(key):
                ss[pk(pname, key)] = p[key]
        for k, v in p.get("det_conditions", {}).items():
            ss[k] = v
    ss["processes"] = new_procs
    ss["_expand_new"] = next(iter(new_procs), None)
    for k, v in loaded.get("global_det_conditions", {}).items():
        ss[k] = v
    return "設定を読み込みました（" + settings_summary(loaded) + "）"


def settings_summary(loaded: dict) -> str:
    """設定の中身を人が読める数で要約する（工程・ステップ・基準値・波形検査点）"""
    procs = loaded.get("processes", {})
    n_steps = sum(len(p.get("steps", [])) for p in procs.values())
    n_base = sum(1 for p in procs.values() if p.get("baseline"))
    n_pts = sum(len(v) for k, v in loaded.get("global_det_conditions", {}).items()
                if isinstance(v, list) and (k.endswith("_t_det_list") or k.endswith("_xy_det_list")))
    txt = f"工程 {len(procs)}・ステップ {n_steps}・基準値 {n_base} 工程"
    return txt + (f"・波形検査点 {n_pts}" if n_pts else "")


# ═══════════════════════════════════════════════════════════════
# フォルダ読み込み・保存（ローカル実行時）
# ═══════════════════════════════════════════════════════════════

def _csvs_newest_first(folder: str) -> list:
    return sorted(glob.glob(os.path.join(folder, "*.csv")), key=os.path.getmtime, reverse=True)


def scan_folder(root: str, sub_ref: str, sub_cmp: str, sub_cfg: str, max_cmp: int) -> list:
    """root/sub_ref の最新 CSV を基準、root/sub_cmp の新しい順 max_cmp 件を比較として読み込む。

    設定ファイルが 1 件ならそれも適用する。戻り値は結果メッセージのリスト。
    設定ファイルが複数ある場合は session_state["folder_jsons_found"] に一覧を残す。
    """
    root = os.path.normpath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"フォルダが見つかりません: {root}")
    msgs = []
    refs = _csvs_newest_first(os.path.join(root, sub_ref))
    if refs:
        p = refs[0]
        df, ct = _load_path(p, os.path.getmtime(p))
        add_csv(p, df, ct, f"基準（{os.path.basename(p)}）", True, "folder")
        msgs.append(f"基準データ: {os.path.basename(p)}")
    n_new = 0
    for p in _csvs_newest_first(os.path.join(root, sub_cmp))[:max_cmp]:
        if p in store():
            continue
        try:
            df, ct = _load_path(p, os.path.getmtime(p))
            add_csv(p, df, ct, parse_csv_label(os.path.basename(p)), False, "folder")
            n_new += 1
        except Exception:
            pass
    msgs.append(f"比較データ: {n_new} 件追加")
    jsons = sorted(glob.glob(os.path.join(root, sub_cfg, "*.json")), key=os.path.getmtime, reverse=True)
    st.session_state["folder_jsons_found"] = jsons
    if len(jsons) == 1:
        with open(jsons[0], encoding="utf-8") as f:
            apply_settings_json(json.load(f))
        msgs.append(f"設定ファイル: {os.path.basename(jsons[0])} を適用")
    elif len(jsons) > 1:
        msgs.append(f"設定ファイルが {len(jsons)} 件あります。適用するものを選んでください")
    st.session_state["folder_last_root"] = root
    return msgs


def save_to_folder(root: str, sub_ref: str, sub_cmp: str, sub_cfg: str) -> str:
    """読み込み中の全 CSV と設定ファイルをフォルダに書き出す。"""
    from datetime import datetime

    root = os.path.normpath(root)
    for sub in (sub_ref, sub_cmp, sub_cfg):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    n = 0
    for key, entry in store().items():
        name = os.path.basename(key)
        if not name.lower().endswith(".csv"):
            name += ".csv"
        entry["df"].to_csv(os.path.join(root, sub_ref if entry.get("is_ref") else sub_cmp, name), index=False)
        n += 1
    cfg_name = ""
    if st.session_state.get("processes"):
        cfg_name = f"apb_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(root, sub_cfg, cfg_name), "w", encoding="utf-8") as f:
            f.write(build_settings_json())
        st.session_state["_project_saved"] = True
    return f"CSV {n} 件" + (f"と設定ファイル {cfg_name}" if cfg_name else "") + f" を {root} に保存しました"
