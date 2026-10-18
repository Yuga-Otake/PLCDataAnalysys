"""
views/capability.py — 工程能力

ステップごとの Cpk（基準値の許容幅を規格としたときの工程能力）と、
工程ごとのタクト目標に対する工程能力、登録済みの基準値（良品条件）を一覧する。
旧画面の品質分析モード（工程横断 Cpk サマリー・良品条件一覧）を置き換える。
"""
import numpy as np
import pandas as pd
import streamlit as st

from core.analysis import MODE_LABELS
from core.evaluate import compare_keys_by_time, cpk, evaluate_process, rule
from core.state import pk
from core.ui import numeric, page_header, rule_popover

ss = st.session_state
page_header(
    "工程能力",
    "基準値の許容幅に対して、各ステップがどれだけ余裕をもって収まっているか（Cpk）を一覧します。",
    eyebrow="点検",
)
store = ss.get("csv_store", {})
processes = ss.get("processes", {})
ready = [p for p in processes if ss.get(pk(p, "baseline"))]
cmp_keys = compare_keys_by_time()
if not ready or not cmp_keys:
    st.info("基準値の登録と比較データが必要です。")
    if st.button("ホームへ", type="primary"):
        st.switch_page("views/home.py")
    st.stop()

c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
if ss.get("eval_target") not in cmp_keys:
    ss["eval_target"] = cmp_keys[-1]
target = c1.selectbox("対象データ", cmp_keys, key="eval_target",
                      format_func=lambda k: store[k]["label"] + ("（最新）" if k == cmp_keys[-1] else ""))
with c2:
    rule_popover()


def _grade(v):
    if v is None:
        return "—"
    return "十分（1.33 以上）" if v >= 1.33 else ("やや不足（1.00〜1.33）" if v >= 1.0 else "不足（1.00 未満）")


def _grade_style(v: str) -> str:
    if v.startswith("十分"):
        return "background-color: rgba(63, 185, 80, 0.16)"
    if v.startswith("やや"):
        return "background-color: rgba(210, 153, 34, 0.22)"
    if v.startswith("不足"):
        return "background-color: rgba(229, 83, 75, 0.22)"
    return ""


r = rule()
evals = {p: evaluate_process(store[target]["df"], p, r) for p in ready}

# ── ステップごとの Cpk ───────────────────────────────────────
st.subheader("ステップごとの Cpk", anchor=False)
rows = []
for p, ev in evals.items():
    for s in ev["steps"]:
        if s["ref"] is None or s["n"] < 2:
            continue
        v = s["series"]["value"].to_numpy(dtype=float)
        lsl = max(0.0, s["ref"] - s["tol"])  # 時間は 0 未満にならない
        c = cpk(v, s["ref"] + s["tol"], lsl if lsl > 0 else None)
        rows.append({"評価": _grade(c), "工程": p, "ステップ": s["name"], "Cpk": c,
                     "平均 [ms]": float(np.mean(v)), "σ [ms]": float(np.std(v, ddof=1)),
                     "下限 [ms]": lsl, "上限 [ms]": s["ref"] + s["tol"], "サイクル数": s["n"]})
if rows:
    tbl = numeric(pd.DataFrame(rows), ["Cpk"]).sort_values("Cpk", na_position="last")
    st.dataframe(tbl.style.map(_grade_style, subset=["評価"]).format(
        {"Cpk": "{:.2f}", "平均 [ms]": "{:.1f}", "σ [ms]": "{:.2f}", "下限 [ms]": "{:.1f}", "上限 [ms]": "{:.1f}"},
        na_rep="—"), hide_index=True, width="stretch")
    st.caption("規格は 基準値 ± 許容幅（判定ルールと同じ、下限は 0 ms 以上）。Cpk の小さい順に並べています。"
               "1.33 以上で十分、1.00 未満は許容幅に対してばらつきや偏りが大きい状態です。")
else:
    st.caption("Cpk を計算できるステップがありません。")

# ── タクト目標に対する工程能力 ───────────────────────────────
st.subheader("タクト目標に対する工程能力", anchor=False)
trows = []
for p, ev in evals.items():
    takt = ev["takt"]
    res = ev["result"]
    if not takt or res is None or len(res) == 0:
        trows.append({"工程": p, "タクト目標 [ms]": takt or None, "最終ステップ終了 平均 [ms]": ev["last_end"],
                      "Cpk": None, "評価": "タクト目標が未設定" if not takt else "—"})
        continue
    # サイクルごとの「最後のステップの終わり」
    ends = []
    for s in ev["steps"]:
        ser = s["series"]
        if len(ser):
            e = ser["start"] + (ser["value"] if s["mode"] != "single" else 0)
            ends.append(pd.Series(e.to_numpy(), index=ser["サイクル#"].to_numpy()))
    cycle_end = pd.concat(ends, axis=1).max(axis=1).to_numpy() if ends else np.array([])
    c = cpk(cycle_end, float(takt))
    trows.append({"工程": p, "タクト目標 [ms]": takt, "最終ステップ終了 平均 [ms]": float(np.mean(cycle_end)),
                  "Cpk": c, "評価": _grade(c)})
st.dataframe(numeric(pd.DataFrame(trows), ["タクト目標 [ms]", "最終ステップ終了 平均 [ms]", "Cpk"]).style.map(_grade_style, subset=["評価"]).format(
    {"Cpk": "{:.2f}", "最終ステップ終了 平均 [ms]": "{:.1f}", "タクト目標 [ms]": "{:.0f}"}, na_rep="—"),
    hide_index=True, width="stretch")
st.caption("サイクルごとに、最後のステップが終わる時刻をタクト目標（上限）と比べています。タクト目標は工程とステップ画面で設定します。")

# ── 良品条件（登録済みの基準値）─────────────────────────────
with st.expander("良品条件（登録済みの基準値）の一覧"):
    brows = []
    for p in ready:
        meta = ss.get(pk(p, "baseline_meta"), {})
        for s in evals[p]["steps"]:
            if s["ref"] is None:
                continue
            brows.append({"工程": p, "ステップ": s["name"], "測り方": MODE_LABELS.get(s["mode"], ""),
                          "基準 [ms]": s["ref"], "許容幅 ± [ms]": s["tol"],
                          "算出元": meta.get("source", ""), "登録日時": meta.get("created_at", "")})
    st.dataframe(pd.DataFrame(brows).style.format({"基準 [ms]": "{:.1f}", "許容幅 ± [ms]": "{:.1f}"}),
                 hide_index=True, width="stretch")
