"""
core/project.py — 設定ファイル（プロジェクト JSON）の書き出し

工程・ステップ・基準値・波形検出条件をまとめて 1 つの JSON にする。
読み込み側は legacy_app._apply_settings_json（段階 2 でデータ画面へ移設予定）。
"""
import json
from datetime import datetime

import streamlit as st

from core.state import pk

_SIMPLE = (bool, int, float, str, list, dict, type(None))


def build_settings_json() -> str:
    ss = st.session_state
    procs = {}
    for pname in ss.get("processes", {}):
        det_conds = {
            k: v for k, v in ss.items()
            if isinstance(k, str) and k.startswith(f"wvol_{pname}_") and isinstance(v, _SIMPLE)
        }
        procs[pname] = {
            "trigger":        ss.get(pk(pname, "trigger"), ""),
            "edge":           ss.get(pk(pname, "edge"), "RISE"),
            "takt_target_ms": ss.get(pk(pname, "takt"), 0),
            "steps":          ss.get(pk(pname, "steps_list"), []),
            "baseline":       ss.get(pk(pname, "baseline"), {}),
            "baseline_meta":  ss.get(pk(pname, "baseline_meta"), {}),
            "wv_baseline":    ss.get(pk(pname, "wv_baseline"), {}),
            "wv_xy_baseline": ss.get(pk(pname, "wv_xy_baseline"), {}),
            "det_conditions": det_conds,
        }
    glob_det = {
        k: v for k, v in ss.items()
        if isinstance(k, str) and k.startswith("wvol___global_") and isinstance(v, _SIMPLE)
    }
    return json.dumps(
        {"version": "1.2",
         "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
         "source_csv": ss.get("uploaded_filename", ""),
         "processes": procs,
         "global_det_conditions": glob_det},
        ensure_ascii=False, indent=2,
    )


def settings_file_name() -> str:
    return f"apb_settings_{datetime.now().strftime('%Y%m%d_%H%M')}.json"


def mark_saved():
    """download_button の on_click から呼ぶ（ワークフローの「設定の保存」完了判定）"""
    st.session_state["_project_saved"] = True
