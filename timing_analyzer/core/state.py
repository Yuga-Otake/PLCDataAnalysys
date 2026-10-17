"""
core/state.py — 全ページで共有するセッション状態

st.navigation によるマルチページ構成では、どのページから開いても
データと工程設定が揃っている必要がある。旧画面（legacy_app.py）と同じキー・
同じ規則で状態を初期化し、準備の進み具合（ワークフロー）もここで判定する。
"""
import functools
import hashlib
import inspect
import os

import streamlit as st

from analyzer import detect_bool_columns, detect_cycles, load_csv

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_SAMPLE = "sample_baseline_3cyc.csv"
CMP_SAMPLE = "sample_playback.csv"


def pk(pname: str, suffix: str) -> str:
    """工程ごとの session_state キー（legacy_app.pk と同一規則）"""
    safe = "".join(c if c.isalnum() else "_" for c in pname)
    return f"P_{safe}__{suffix}"


@st.cache_data(show_spinner=False)
def _load_csv_path(path: str):
    df = load_csv(path)
    return df, detect_bool_columns(df)


@st.cache_data(show_spinner=False)
def count_cycles(df, trigger_col: str, edge: str) -> int:
    try:
        return len(detect_cycles(df, trigger_col, edge))
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════
# ウィジェット値のページ間保持
# ═══════════════════════════════════════════════════════════════

_PERSIST_WIDGETS = (
    "checkbox", "toggle", "radio", "selectbox", "multiselect", "slider",
    "select_slider", "number_input", "text_input", "text_area",
    "date_input", "time_input", "color_picker", "segmented_control", "pills",
)


def enable_session_persistence():
    """key 付きウィジェットの値を既定でセッション全体に保持する。

    Streamlit は表示されなくなったウィジェットの値を破棄するため、ページを
    移ると旧画面の設定（タクト目標・波形検査条件など）が消えてしまう。
    旧画面のウィジェットは数百あり個別に persist_state を付けられないので、
    移行期間中はここで一括して既定値を "session" にする。
    """
    from streamlit.delta_generator import DeltaGenerator

    if getattr(DeltaGenerator, "_apb_persist_patched", False):
        return
    for name in _PERSIST_WIDGETS:
        fn = getattr(DeltaGenerator, name, None)
        if fn is None or "persist_state" not in inspect.signature(fn).parameters:
            continue

        def _wrap(f):
            @functools.wraps(f)
            def inner(self, *args, **kwargs):
                if kwargs.get("key") is not None:
                    kwargs.setdefault("persist_state", "session")
                return f(self, *args, **kwargs)
            return inner

        wrapped = _wrap(fn)
        setattr(DeltaGenerator, name, wrapped)
        # st.xxx はインポート時に束縛済みのメソッドなので差し替える
        main_dg = getattr(st, name, None)
        if main_dg is not None and hasattr(main_dg, "__self__"):
            setattr(st, name, wrapped.__get__(main_dg.__self__))
    DeltaGenerator._apb_persist_patched = True


# ═══════════════════════════════════════════════════════════════
# 初期化
# ═══════════════════════════════════════════════════════════════

def ensure_session():
    """どのページから開いても、データと工程の初期状態を揃える。"""
    ss = st.session_state
    ss.setdefault("csv_store", {})
    ss.setdefault("ref_csv_key", None)
    ss.setdefault("processes", {})

    if ss.get("use_sample", True):
        _load_samples()

    store = ss["csv_store"]
    if not store:
        return
    active = ss.get("active_csv")
    if active not in store:
        active = ss.get("ref_csv_key") if ss.get("ref_csv_key") in store else next(iter(store))
        ss["active_csv"] = active
    entry = store[active]
    ss["df"], ss["col_types"] = entry["df"], entry["col_types"]

    bool_cols = [c for c, t in entry["col_types"].items() if t == "bool"]
    # 初回のみ工程を自動登録する（列構成の変化は旧画面側の規則に任せる）
    if "_schema_hash" not in ss:
        ss["_schema_hash"] = hashlib.md5(
            str(sorted(entry["col_types"].keys())).encode()).hexdigest()[:8]
        _auto_register_processes(bool_cols)
    for pname, pinfo in ss["processes"].items():
        _init_process_defaults(pname, pinfo, bool_cols)


def _load_samples():
    ss = st.session_state
    store = ss["csv_store"]
    ref_path = os.path.join(APP_DIR, REF_SAMPLE)
    if REF_SAMPLE not in store and os.path.exists(ref_path):
        df, ct = _load_csv_path(ref_path)
        store[REF_SAMPLE] = {"df": df, "col_types": ct,
                             "label": f"基準サンプル（{REF_SAMPLE}）", "is_ref": True}
        ss["ref_csv_key"] = REF_SAMPLE
    cmp_path = os.path.join(APP_DIR, CMP_SAMPLE)
    if CMP_SAMPLE not in store and os.path.exists(cmp_path) and not ss.get("_cmp_sample_loaded"):
        df, ct = _load_csv_path(cmp_path)
        store[CMP_SAMPLE] = {"df": df, "col_types": ct,
                             "label": f"比較サンプル（{CMP_SAMPLE}）", "is_ref": False}
    # ユーザーが比較サンプルを外した後に復活させない
    ss["_cmp_sample_loaded"] = True


def _auto_register_processes(bool_cols: list):
    ss = st.session_state
    if ss["processes"] or not bool_cols:
        return
    starts = [c for c in bool_cols
              if any(kw in c.lower() for kw in ["start", "begin", "trigger", "trig"])]
    for var in starts or bool_cols[:1]:
        name = var
        for sfx in ["_Start", "_start", "_Begin", "_begin", "_Trigger", "_trigger"]:
            name = name.replace(sfx, "")
        ss["processes"][name] = {"trigger_col": var, "edge": "RISE",
                                 "takt_target_ms": 0, "steps": []}
    ss["_expand_new"] = next(iter(ss["processes"]), None)


def _init_process_defaults(pname: str, pinfo: dict, bool_cols: list):
    ss = st.session_state
    trigger = pinfo.get("trigger_col", bool_cols[0] if bool_cols else "")
    if trigger not in bool_cols and bool_cols:
        trigger = bool_cols[0]
    ss.setdefault(pk(pname, "trigger"), trigger)
    ss.setdefault(pk(pname, "edge"), pinfo.get("edge", "RISE"))
    ss.setdefault(pk(pname, "takt"), int(pinfo.get("takt_target_ms", 0)))
    if pk(pname, "steps_list") not in ss:
        ss[pk(pname, "steps_list")] = [
            {"name": s.get("label", s["variable"]), "color": s.get("color", "#4472C4"),
             "mode": "single", "variable": s["variable"], "edge": "RISE"}
            for s in pinfo.get("steps", [])
        ]


# ═══════════════════════════════════════════════════════════════
# 参照用ヘルパー
# ═══════════════════════════════════════════════════════════════

def ref_entry():
    store = st.session_state.get("csv_store", {})
    key = st.session_state.get("ref_csv_key")
    if key in store:
        return key, store[key]
    for k, v in store.items():
        if v.get("is_ref"):
            return k, v
    return (next(iter(store.items())) if store else (None, None))


def compare_keys() -> list:
    return [k for k, v in st.session_state.get("csv_store", {}).items() if not v.get("is_ref")]


def process_rows() -> list:
    """工程ごとの準備状況（ホームの一覧・ワークフロー判定に使う）"""
    ss = st.session_state
    _, ref = ref_entry()
    rows = []
    for pname in ss.get("processes", {}):
        trigger = ss.get(pk(pname, "trigger"), "")
        edge = ss.get(pk(pname, "edge"), "RISE")
        steps = ss.get(pk(pname, "steps_list"), [])
        baseline = ss.get(pk(pname, "baseline"), {})
        n_cyc = count_cycles(ref["df"], trigger, edge) if ref and trigger in ref["df"].columns else 0
        if not steps:
            status = "steps"
        elif not baseline:
            status = "baseline"
        else:
            status = "ready"
        rows.append({"name": pname, "trigger": trigger, "edge": edge, "cycles": n_cyc,
                     "steps": len(steps), "baseline": len(baseline), "status": status})
    return rows


STAGES = [
    ("data",     "データ読込"),
    ("process",  "工程登録"),
    ("steps",    "ステップ定義"),
    ("baseline", "基準値登録"),
    ("save",     "設定の保存"),
]


def workflow():
    """準備の各段階の完了状況と、次にやるべき段階を返す。"""
    ss = st.session_state
    rows = process_rows()
    no_steps = [r["name"] for r in rows if r["status"] == "steps"]
    no_bl = [r["name"] for r in rows if r["status"] == "baseline"]
    done = {
        "data":     bool(ss.get("csv_store")),
        "process":  bool(rows),
        "steps":    bool(rows) and not no_steps,
        "baseline": bool(rows) and not no_steps and not no_bl,
        "save":     bool(ss.get("_project_saved")),
    }
    current = next((k for k, _ in STAGES if not done[k]), None)
    target = (no_steps[0] if current == "steps" and no_steps
              else no_bl[0] if current == "baseline" and no_bl else None)
    return {"done": done, "current": current, "target": target, "rows": rows}


# ═══════════════════════════════════════════════════════════════
# 工程の追加・名前変更・削除
# ═══════════════════════════════════════════════════════════════

def add_process(name: str, trigger: str, edge: str = "RISE", takt_ms: int = 0):
    ss = st.session_state
    ss["processes"][name] = {"trigger_col": trigger, "edge": edge,
                             "takt_target_ms": takt_ms, "steps": []}
    ss[pk(name, "trigger")] = trigger
    ss[pk(name, "edge")] = edge
    ss[pk(name, "takt")] = int(takt_ms)
    ss[pk(name, "steps_list")] = []
    ss["_expand_new"] = name


def rename_process(old: str, new: str):
    """工程名を変え、工程に紐づく session_state キー（P_<工程>__*）をすべて移す"""
    ss = st.session_state
    ss["processes"] = {(new if k == old else k): v for k, v in ss["processes"].items()}
    old_pfx, new_pfx = pk(old, ""), pk(new, "")
    for key in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(old_pfx)]:
        ss[new_pfx + key[len(old_pfx):]] = ss.pop(key)
    if ss.get("_expand_new") == old:
        ss["_expand_new"] = new


def delete_process(name: str):
    ss = st.session_state
    ss["processes"].pop(name, None)
    pfx = pk(name, "")
    for key in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(pfx)]:
        ss.pop(key, None)
    if ss.get("_expand_new") == name:
        ss["_expand_new"] = None
