"""
core/profile.py — CSV の事前診断（準備の補助）

読み込んだ CSV を設定の前に調べ、準備画面の入力のそばに出す提案の材料を作る。
  sampling_profile   サンプリング周期（欠落・重複・逆行・ばらつき）
  variable_profile   変数の種類（Bool／状態番号／カウンタ／アナログ／一定）と変化回数
  compare_consistency 基準データと比較データの列構成・周期の違い
  cycle_health       サイクル長の外れ（停止の可能性など）と、基準に使えるきれいな範囲
  group_hypotheses   変数名の接頭辞から工程をまとめ、サイクル開始信号を推定
  sequence_hypothesis 1 サイクル内の動作の順番（仮説）と、その確からしさ
大量の変数（数万）でも動くよう、信号処理は列単位でまとめて行う。
"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from analyzer import normalize_bool_series

KIND_LABELS = {"bool": "Bool", "state": "状態番号", "counter": "カウンタ", "analog": "アナログ", "constant": "一定"}
_START_WORDS = re.compile(r"start|begin|trig|開始|起動|スタート", re.IGNORECASE)


def _ts_ms(df: pd.DataFrame) -> np.ndarray:
    return df["Timestamp"].to_numpy("datetime64[ns]").astype(np.int64) / 1e6


def _bool_array(s: pd.Series) -> np.ndarray:
    return normalize_bool_series(s).to_numpy().astype(np.int8)


# ═══════════════════════════════════════════════════════════════
# データの健全性
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def sampling_profile(df: pd.DataFrame) -> dict:
    """タイムスタンプの間隔を調べる。status: ok / warn / ng"""
    t = _ts_ms(df)
    if len(t) < 3:
        return {"status": "ng", "message": "行数が少なすぎます", "dt_ms": 0.0}
    dt = np.diff(t)
    med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    n_dup = int(np.sum(dt == 0))
    n_back = int(np.sum(dt < 0))
    gaps = dt[dt > 3 * med] if med > 0 else np.array([])
    irregular = float(np.mean(np.abs(dt - med) > 0.5 * med) * 100) if med > 0 else 100.0
    res = {"dt_ms": med, "irregular_pct": irregular, "n_gaps": int(len(gaps)),
           "gap_total_ms": float(np.sum(gaps - med)) if len(gaps) else 0.0,
           "max_gap_ms": float(np.max(gaps)) if len(gaps) else 0.0,
           "n_dup": n_dup, "n_back": n_back, "duration_s": float((t[-1] - t[0]) / 1000)}
    if n_back or n_dup > len(dt) * 0.01:
        res["status"], res["message"] = "ng", (
            f"時刻の逆行 {n_back} 件・重複 {n_dup} 件があります。ロガーの記録方法を確認してください")
    elif len(gaps) or irregular > 1:
        msg = []
        if len(gaps):
            msg.append(f"記録の抜け {len(gaps)} か所（最大 {res['max_gap_ms']:.0f} ms）")
        if irregular > 1:
            msg.append(f"間隔のばらつき {irregular:.1f}%")
        res["status"], res["message"] = "warn", "、".join(msg) + "。抜けをまたぐサイクルは時間がずれます"
    else:
        res["status"], res["message"] = "ok", f"{med:g} ms 周期で一定"
    return res


@st.cache_data(show_spinner=False)
def variable_profile(df: pd.DataFrame, col_types: tuple) -> pd.DataFrame:
    """変数ごとの種類と変化回数。col_types は (列名, "bool"|"numeric") のタプル"""
    rows = []
    for col, t in col_types:
        s = df[col]
        missing = float(s.isna().mean() * 100)
        v = _bool_array(s) if t == "bool" else s.to_numpy(dtype=float)
        ok = ~np.isnan(v) if t != "bool" else np.ones(len(v), dtype=bool)
        vv = v[ok]
        d = np.diff(vv) if len(vv) > 1 else np.array([])
        changes = int(np.count_nonzero(d))
        if changes == 0:
            kind = "constant"
        elif t == "bool":
            kind = "bool"
        else:
            is_int = bool(np.all(np.mod(vv, 1) == 0))
            nz = d[d != 0]
            if is_int and len(nz) and np.mean(nz > 0) >= 0.95 and len(np.unique(vv)) > 32:
                kind = "counter"
            elif is_int and len(np.unique(vv)) <= 32:
                kind = "state"
            else:
                kind = "analog"
        rows.append({"var": col, "kind": kind, "changes": changes, "missing_pct": missing})
    return pd.DataFrame(rows).set_index("var") if rows else pd.DataFrame(
        columns=["kind", "changes", "missing_pct"])


def active_bool_vars(entry: dict) -> list:
    """変化のある Bool 変数（一定の変数は候補から外す）"""
    prof = variable_profile(entry["df"], tuple(entry["col_types"].items()))
    return [v for v in prof.index if prof.at[v, "kind"] == "bool"]


def compare_consistency(ref: dict, cmp: dict, used_vars: set) -> dict:
    """基準と比較データの違い。status: ok / warn / ng"""
    rc, cc = set(ref["col_types"]), set(cmp["col_types"])
    missing_used = sorted(v for v in used_vars if v and v in rc and v not in cc)
    missing_other = sorted(rc - cc - set(missing_used))
    extra = sorted(cc - rc)
    dt_r, dt_c = sampling_profile(ref["df"])["dt_ms"], sampling_profile(cmp["df"])["dt_ms"]
    dt_diff = abs(dt_c - dt_r) > 0.1 * dt_r if dt_r else False
    if missing_used:
        status, msg = "ng", f"設定で使う変数が {len(missing_used)} 個ありません（{', '.join(missing_used[:3])}…）"
    elif dt_diff or missing_other:
        parts = []
        if dt_diff:
            parts.append(f"周期が違います（基準 {dt_r:g} ms / 比較 {dt_c:g} ms）")
        if missing_other:
            parts.append(f"基準にある変数が {len(missing_other)} 個ありません")
        status, msg = "warn", "。".join(parts)
    else:
        status, msg = "ok", "一致" + (f"（比較側に追加の変数 {len(extra)} 個）" if extra else "")
    return {"status": status, "message": msg, "missing_used": missing_used,
            "missing_other": missing_other, "extra": extra}


# ═══════════════════════════════════════════════════════════════
# サイクルの健全性
# ═══════════════════════════════════════════════════════════════

def _rise_index(b: np.ndarray, edge: str) -> np.ndarray:
    prev = np.concatenate([[0], b[:-1]])
    mask = (prev == 0) & (b == 1) if edge == "RISE" else (prev == 1) & (b == 0)
    return np.nonzero(mask)[0]


@st.cache_data(show_spinner=False)
def cycle_health(df: pd.DataFrame, trigger: str, edge: str) -> dict:
    """サイクル長の外れを調べる。サイクル番号は解析結果のサイクル# と同じ 1 始まり。

    最後のサイクルは終わりが分からないため判定しない。
    """
    if trigger not in df.columns:
        return {"n": 0, "outliers": {}, "clean": None, "median_ms": 0.0, "lengths": []}
    idx = _rise_index(_bool_array(df[trigger]), edge)
    t = _ts_ms(df)
    if len(idx) < 3:
        return {"n": len(idx), "outliers": {}, "clean": (1, len(idx)) if len(idx) else None,
                "median_ms": 0.0, "lengths": []}
    lengths = np.diff(t[idx])
    med = float(np.median(lengths))
    mad = float(np.median(np.abs(lengths - med))) or 0.01 * med
    z = 0.6745 * (lengths - med) / mad
    outliers = {}
    for i, (x, zz) in enumerate(zip(lengths, z), start=1):
        if x > 1.5 * med or zz > 3.5:
            outliers[i] = f"長い（{x:.0f} ms、停止の可能性）"
        elif x < 0.5 * med or zz < -3.5:
            outliers[i] = f"短い（{x:.0f} ms、途中で始まり直した可能性）"
    # 外れを含まない最長の連続範囲
    best, cur = (0, 0), None
    for i in range(1, len(lengths) + 1):
        if i in outliers:
            cur = None
            continue
        cur = (cur[0], i) if cur else (i, i)
        if cur[1] - cur[0] > best[1] - best[0]:
            best = cur
    return {"n": len(idx), "outliers": outliers, "clean": best if best != (0, 0) else None,
            "median_ms": med, "lengths": lengths.tolist(),
            "cv_pct": float(np.std(lengths) / med * 100) if med else 0.0}


# ═══════════════════════════════════════════════════════════════
# 工程・動作シーケンスの仮説
# ═══════════════════════════════════════════════════════════════

def _separator(names: list) -> str | None:
    for sep in (".", "/", ":", "_"):
        if names and sum(sep in n for n in names) >= 0.5 * len(names):
            return sep
    return None


@st.cache_data(show_spinner=False)
def group_hypotheses(df: pd.DataFrame, bool_vars: tuple) -> list:
    """変数名の接頭辞で工程候補をまとめ、各グループのサイクル開始信号を推定する。

    返り値: [{"name", "vars", "trigger", "period_ms", "cv_pct", "n_cycles", "reason"}]
    """
    sep = _separator(list(bool_vars))
    groups: dict = {}
    for v in bool_vars:
        key = v.split(sep)[0] if sep and sep in v else "（その他）"
        groups.setdefault(key, []).append(v)
    t = _ts_ms(df)
    out = []
    for name, vs in groups.items():
        best = None
        for v in vs:
            b = _bool_array(df[v])
            idx = _rise_index(b, "RISE")
            if len(idx) < 3:
                continue
            iv = np.diff(t[idx])
            cv = float(np.std(iv) / np.mean(iv)) if np.mean(iv) > 0 else 9.9
            on_ratio = float(b.mean())
            # 周期のそろい方を第一に、名前（開始・start など）とパルスらしさで順位を付ける
            score = cv - (0.05 if _START_WORDS.search(v) else 0) + 0.01 * on_ratio
            if best is None or score < best[0]:
                best = (score, v, float(np.median(iv)), cv, len(idx))
        if best is None:
            out.append({"name": name, "vars": vs, "trigger": None, "period_ms": 0.0, "cv_pct": 0.0,
                        "n_cycles": 0, "reason": "周期的に立ち上がる信号が見つかりません"})
            continue
        _, trig, period, cv, n = best
        reason = "名前と周期性から推定" if _START_WORDS.search(trig) else "周期性から推定"
        out.append({"name": name, "vars": vs, "trigger": trig, "period_ms": period, "cv_pct": cv * 100,
                    "n_cycles": n, "reason": reason})
    return sorted(out, key=lambda g: (g["trigger"] is None, g["name"]))


@st.cache_data(show_spinner=False)
def sequence_hypothesis(df: pd.DataFrame, trigger: str, edge: str, candidates: tuple,
                        prefix: str = "") -> list:
    """1 サイクル内の動作の順番の仮説を作る。

    各変数について、サイクルごとの最初の変化（立ち上がり／開始時点で ON なら立ち下がり／立ち下がり）を集め、
    中央値の時刻で並べる。確からしさは、検出できたサイクルの割合と、隣の動作と順番が入れ替わった割合で決める。
    返り値（時刻順）: [{"var", "label", "kind", "t_ms", "spread_ms", "det_rate", "swap_rate", "confidence",
                        "mode", "edge", "include"}]
    """
    if trigger not in df.columns:
        return []
    starts = _rise_index(_bool_array(df[trigger]), edge)
    if len(starts) == 0:
        return []
    t = _ts_ms(df)
    ends = np.append(starts[1:], len(df))
    n_cyc = len(starts)
    per_var = {}
    for v in candidates:
        if v == trigger or v not in df.columns:
            continue
        b = _bool_array(df[v])
        rises, falls = _rise_index(b, "RISE"), _rise_index(b, "FALL")
        kinds, times = [], []
        for s, e in zip(starts, ends):
            if b[s] == 1:
                j = np.searchsorted(falls, s, side="right")
                kinds.append("ON")
                times.append(t[falls[j]] - t[s] if j < len(falls) and falls[j] < e else np.nan)
            else:
                j = np.searchsorted(rises, s, side="left")
                if j < len(rises) and rises[j] < e:
                    kinds.append("RISE")
                    times.append(t[rises[j]] - t[s])
                else:
                    j = np.searchsorted(falls, s, side="left")
                    kinds.append("FALL")
                    times.append(t[falls[j]] - t[s] if j < len(falls) and falls[j] < e else np.nan)
        kinds, times = np.array(kinds), np.array(times, dtype=float)
        vals, counts = np.unique(kinds, return_counts=True)
        kind = vals[np.argmax(counts)]
        tk = np.where(kinds == kind, times, np.nan)
        det = float(np.mean(~np.isnan(tk)))
        if det == 0:
            continue
        per_var[v] = {"kind": kind, "times": tk, "det": det}
    order = sorted(per_var, key=lambda v: np.nanmedian(per_var[v]["times"]))
    rows = []
    for i, v in enumerate(order):
        d = per_var[v]
        tk = d["times"]
        swap = 0.0
        if i + 1 < len(order):
            nxt = per_var[order[i + 1]]["times"]
            both = ~np.isnan(tk) & ~np.isnan(nxt)
            swap = float(np.mean(tk[both] > nxt[both])) if both.any() else 0.0
        q75, q25 = np.nanpercentile(tk, [75, 25])
        conf = ("高" if d["det"] >= 0.95 and swap == 0 else
                "中" if d["det"] >= 0.7 and swap <= 0.1 else "低")
        mode, sedge = ("on_period", "RISE") if d["kind"] == "RISE" else ("single", "FALL")
        label = v[len(prefix):].lstrip("._/:") if prefix and v.startswith(prefix) else v
        rows.append({"var": v, "label": label, "kind": d["kind"], "t_ms": float(np.nanmedian(tk)),
                     "spread_ms": float(q75 - q25), "det_rate": d["det"], "swap_rate": swap,
                     "confidence": conf, "mode": mode, "edge": sedge,
                     "include": conf != "低" and n_cyc >= 1})
    return rows
