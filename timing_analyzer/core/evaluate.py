"""
core/evaluate.py — 点検（基準値との比較）の判定ロジック

判定ルール（画面にもそのまま表示する）:
  計測値   時刻ステップ＝サイクル開始からの発生時刻 ／ 期間ステップ＝長さ
  許容幅   基準値 ± max(N × 基準のσ, 最小許容幅)        既定 N=3, 最小許容幅=2 ms
           基準のσは、基準データのサイクル数が少ないほど過小に出るため、
           σ の 95% 上側信頼限界を使う（3 サイクルで約 5.4 倍、40 サイクルで約 1.2 倍）
  NG       許容幅の外に出たサイクル
  判定     NG率 ≥ 異常のしきい値（既定 10%）→ 異常 ／ NG が 1 サイクル以上 → 注意 ／ それ以外 → 正常
"""
import numpy as np
import pandas as pd
import streamlit as st

from core.analysis import analyze, cycle_starts, mean_cycle_ms
from core.state import pk

DEFAULTS = {"nsig": 3.0, "min_tol": 2.0, "ng_rate": 10.0}

STATUS = {
    "ng":      ("異常", "ng", 3),
    "warn":    ("注意", "warn", 2),
    "ok":      ("正常", "ok", 1),
    "nobase":  ("基準値なし", "gray", 0),
    "missing": ("未検出", "gray", 0),
}


def rule() -> dict:
    ss = st.session_state
    return {k: float(ss.get(f"eval_{k}", v)) for k, v in DEFAULTS.items()}


def rule_text(r: dict | None = None) -> str:
    r = r or rule()
    return (f"許容幅は 基準値 ± max({r['nsig']:g}σ, {r['min_tol']:g} ms)。"
            f"許容幅の外に出たサイクルを NG とし、NG率 {r['ng_rate']:g}% 以上で異常、"
            "NG が 1 サイクル以上で注意としています。時刻ステップは発生時刻、期間ステップは長さで判定します。"
            "基準のσは、基準データのサイクル数が少ないほど大きめに見積もります（95% 上側信頼限界）。")


# χ²(k) の下側 5% 点（k = 1〜10）。k > 10 は Wilson–Hilferty 近似
_CHI2_05 = [0.00393, 0.1026, 0.3518, 0.7107, 1.1455, 1.6354, 2.1673, 2.7326, 3.3251, 3.9403]


def sigma_upper(sd: float, n: int) -> float:
    """n サイクルから求めた σ（母標準偏差）の 95% 上側信頼限界"""
    if not n or n < 2:
        return sd
    k = n - 1
    q = _CHI2_05[k - 1] if k <= 10 else k * (1 - 2 / (9 * k) - 1.6449 * np.sqrt(2 / (9 * k))) ** 3
    return float(sd * np.sqrt(n / k) * np.sqrt(k / q))


def data_start(entry: dict):
    df = entry["df"]
    try:
        return pd.Timestamp(df["Timestamp"].iloc[0])
    except Exception:
        return pd.Timestamp.min


def compare_keys_by_time() -> list:
    """比較データを、データの取得開始時刻が古い順に並べたキー一覧"""
    store = st.session_state.get("csv_store", {})
    keys = [k for k, v in store.items() if not v.get("is_ref")]
    return sorted(keys, key=lambda k: data_start(store[k]))


def _ref_of(bl: dict) -> tuple:
    """基準値の (計測値の基準, σ, 開始の基準)"""
    if bl.get("mode", "single") == "single":
        return float(bl.get("ref_ms", 0)), float(bl.get("std_ms", 0)), float(bl.get("ref_ms", 0))
    return (float(bl.get("ref_dur_ms", 0)), float(bl.get("std_dur_ms", 0)),
            float(bl.get("ref_start_ms", 0)))


def step_series(result_df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """1 ステップの計測値をサイクルごとに: サイクル#, 開始時刻, start, value"""
    name = step["name"]
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame(columns=["サイクル#", "開始時刻", "start", "value"])
    if step.get("mode", "single") == "single":
        col = f"{name}_遅れ[ms]"
        if col not in result_df.columns:
            return pd.DataFrame(columns=["サイクル#", "開始時刻", "start", "value"])
        out = result_df[["サイクル#", "開始時刻", col]].rename(columns={col: "value"})
        out["start"] = out["value"]
    else:
        sc, dc = f"{name}_start[ms]", f"{name}_dur[ms]"
        if dc not in result_df.columns:
            return pd.DataFrame(columns=["サイクル#", "開始時刻", "start", "value"])
        out = result_df[["サイクル#", "開始時刻", sc, dc]].rename(columns={sc: "start", dc: "value"})
    return out.dropna(subset=["value"]).reset_index(drop=True)


def judge_step(series: pd.DataFrame, bl: dict | None, r: dict, n_base: int = 0) -> dict:
    vals = series["value"].to_numpy(dtype=float) if len(series) else np.array([])
    res = {"n": len(vals), "mean": float(np.mean(vals)) if len(vals) else None,
           "std": float(np.std(vals)) if len(vals) else None,
           "start_mean": float(series["start"].mean()) if len(series) else None,
           "ref": None, "ref_std": None, "ref_start": None, "tol": None,
           "delta": None, "ng_rate": None, "ng_mask": np.zeros(len(vals), dtype=bool)}
    if not len(vals):
        res["status"] = "missing"
        return res
    if not bl:
        res["status"] = "nobase"
        return res
    ref, ref_std, ref_start = _ref_of(bl)
    tol = max(r["nsig"] * sigma_upper(ref_std, n_base), r["min_tol"])
    mask = np.abs(vals - ref) > tol
    rate = float(mask.mean() * 100)
    res.update(ref=ref, ref_std=ref_std, ref_start=ref_start, tol=tol, delta=res["mean"] - ref,
               ng_rate=rate, ng_mask=mask)
    res["status"] = "ng" if rate >= r["ng_rate"] else ("warn" if mask.any() else "ok")
    return res


def evaluate_process(df: pd.DataFrame, pname: str, r: dict | None = None) -> dict:
    """1 工程を 1 つのデータで評価する"""
    ss = st.session_state
    r = r or rule()
    steps = ss.get(pk(pname, "steps_list"), [])
    trigger = ss.get(pk(pname, "trigger"), "")
    edge = ss.get(pk(pname, "edge"), "RISE")
    baseline = ss.get(pk(pname, "baseline"), {})
    n_base = int(ss.get(pk(pname, "baseline_meta"), {}).get("n_cycles", 0) or 0)
    result = analyze(df, trigger, edge, steps) if trigger in df.columns else pd.DataFrame()
    rows = []
    for i, s in enumerate(steps):
        series = step_series(result, s)
        j = judge_step(series, baseline.get(s["name"]), r, n_base)
        rows.append({"name": s["name"], "mode": s.get("mode", "single"),
                     "color": s.get("color", "#4472C4"), "index": i, "series": series, **j})
    statuses = [x["status"] for x in rows]
    worst = max(statuses, key=lambda st_: STATUS[st_][2]) if statuses else "nobase"
    if worst in ("nobase", "missing") and any(x in ("ok", "warn", "ng") for x in statuses):
        worst = max((x for x in statuses if x in ("ok", "warn", "ng")), key=lambda st_: STATUS[st_][2])
    starts = cycle_starts(df, trigger, edge) if trigger in df.columns else []
    takt = int(ss.get(pk(pname, "takt"), 0))
    ends = [x["start_mean"] + (x["mean"] if x["mode"] != "single" else 0)
            for x in rows if x["mean"] is not None]
    return {"name": pname, "status": worst if rows else "nobase", "steps": rows, "result": result,
            "n_cycles": len(result), "n_base": n_base, "cycle_ms": mean_cycle_ms(df, starts), "takt": takt,
            "last_end": max(ends) if ends else None,
            "n_ng_steps": sum(1 for x in statuses if x == "ng"),
            "n_warn_steps": sum(1 for x in statuses if x == "warn")}


# ═══════════════════════════════════════════════════════════════
# 管理図・工程能力
# ═══════════════════════════════════════════════════════════════

_SPC_CONSTS = {  # n: (A2, D3, D4, c4, B3, B4)
    2: (1.880, 0, 3.267, 0.7979, 0, 3.267), 3: (1.023, 0, 2.574, 0.8862, 0, 2.568),
    4: (0.729, 0, 2.282, 0.9213, 0, 2.266), 5: (0.577, 0, 2.114, 0.9400, 0, 2.089),
    6: (0.483, 0, 2.004, 0.9515, 0.030, 1.970), 7: (0.419, 0.076, 1.924, 0.9594, 0.118, 1.882),
    8: (0.373, 0.136, 1.864, 0.9650, 0.185, 1.815), 9: (0.337, 0.184, 1.816, 0.9693, 0.239, 1.761),
    10: (0.308, 0.223, 1.777, 0.9727, 0.284, 1.716), 11: (0.285, 0.256, 1.744, 0.9754, 0.321, 1.679),
    12: (0.266, 0.284, 1.717, 0.9776, 0.354, 1.646), 13: (0.249, 0.308, 1.692, 0.9794, 0.382, 1.618),
    14: (0.235, 0.329, 1.671, 0.9810, 0.406, 1.594), 15: (0.223, 0.347, 1.653, 0.9823, 0.428, 1.572),
    16: (0.212, 0.363, 1.637, 0.9835, 0.448, 1.552), 17: (0.203, 0.378, 1.622, 0.9845, 0.466, 1.534),
    18: (0.194, 0.391, 1.608, 0.9854, 0.482, 1.518), 19: (0.187, 0.403, 1.597, 0.9862, 0.497, 1.503),
    20: (0.180, 0.415, 1.585, 0.9869, 0.510, 1.490), 21: (0.173, 0.425, 1.575, 0.9876, 0.523, 1.477),
    22: (0.167, 0.434, 1.566, 0.9882, 0.534, 1.466), 23: (0.162, 0.443, 1.557, 0.9887, 0.545, 1.455),
    24: (0.157, 0.452, 1.548, 0.9892, 0.555, 1.445), 25: (0.153, 0.459, 1.541, 0.9896, 0.565, 1.435),
}


def control_limits(groups: list) -> dict | None:
    """サブグループ（各時期のサイクル値の配列）から X̄ と R（n>25 は S）の管理限界を求める"""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return None
    xbar = [float(g.mean()) for g in groups]
    rng = [float(g.max() - g.min()) for g in groups]
    sd = [float(g.std(ddof=1)) for g in groups]
    n = int(round(np.mean([len(g) for g in groups])))
    a2, d3, d4, c4, b3, b4 = _SPC_CONSTS[min(_SPC_CONSTS, key=lambda k: abs(k - max(2, n)))]
    cl = float(np.mean(xbar))
    if n > 25:
        s_bar = float(np.mean(sd))
        half = 3 * (s_bar / c4) / np.sqrt(n)
        return {"kind": "S", "xbar": xbar, "sub": sd, "cl": cl, "ucl": cl + half, "lcl": cl - half,
                "sub_cl": s_bar, "sub_ucl": b4 * s_bar, "sub_lcl": b3 * s_bar, "n": n}
    r_bar = float(np.mean(rng))
    return {"kind": "R", "xbar": xbar, "sub": rng, "cl": cl, "ucl": cl + a2 * r_bar, "lcl": cl - a2 * r_bar,
            "sub_cl": r_bar, "sub_ucl": d4 * r_bar, "sub_lcl": d3 * r_bar, "n": n}


def cpk(values, usl: float | None, lsl: float | None = None) -> float | None:
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or (usl is None and lsl is None):
        return None
    m, s = float(v.mean()), float(v.std(ddof=1))
    if s <= 0:
        return None
    sides = []
    if usl is not None:
        sides.append((usl - m) / (3 * s))
    if lsl is not None:
        sides.append((m - lsl) / (3 * s))
    return round(min(sides), 2)
