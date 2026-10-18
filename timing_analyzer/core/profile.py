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


# ═══════════════════════════════════════════════════════════════
# アナログ波形の検出点の仮説（波形検査条件の候補）
# ═══════════════════════════════════════════════════════════════

WAVE_POINT_TYPES = ["最大値点", "最小値点", "閾値超え検出", "傾き変化点"]


def _nice(x: float) -> float:
    """しきい値などを 3 桁に丸める（人が読み直しやすいように）"""
    if not np.isfinite(x) or x == 0:
        return 0.0
    return float(np.round(x, 2 - int(np.floor(np.log10(abs(x))))))


def _conf(det: float, spread_pct: float) -> str:
    if det >= 0.95 and spread_pct <= 10:
        return "高"
    if det >= 0.8 and spread_pct <= 25:
        return "中"
    return "低"


@st.cache_data(show_spinner=False)
def wave_hypotheses_all(df: pd.DataFrame, pairs: tuple) -> dict:
    """wave_hypotheses を (変数, トリガー, エッジ) の組ごとにまとめて計算する（df のハッシュは 1 回だけ）"""
    from analyzer import detect_cycles, get_cycle_waveforms
    cycles, out = {}, {}
    for var, trigger, edge in pairs:
        if (trigger, edge) not in cycles:
            try:
                cycles[(trigger, edge)] = pd.Index(list(detect_cycles(df, trigger, edge)))
            except Exception:
                cycles[(trigger, edge)] = None
        idx = cycles[(trigger, edge)]
        try:
            waves = get_cycle_waveforms(df, idx, [var]) if idx is not None and var in df.columns else []
        except Exception:
            waves = []
        out[var] = wave_hypotheses(waves, var)
    return out


def wave_hypotheses(waves: list, var: str) -> dict:
    """1 つのアナログ変数について、波形の形と検出点の候補を出す。waves はサイクルごとの波形（get_cycle_waveforms）

    サイクルごとの波形（トリガー基準）を重ねた平均の形から、
      山形 → 最大値点、谷形 → 最小値点、
      台形（上がって保持して戻る）→ 閾値超え検出（立ち上がり・立ち下がり）、
      途中で傾きが急に変わる → 傾き変化点
    を候補にする。検出は波形検査と同じ関数で全サイクルに試し、検出できた割合とサイクル間のばらつきで
    確からしさ（高・中・低）を付ける。設定値（しきい値・平滑化など）は、そのまま検出点に入れられる形で返す。
    返り値: {"shape", "cycle_ms", "wpost", "spark", "cands": [{"dtype", "name", "params", "t_ms", "v",
             "det_rate", "confidence", "include", "note"}], "defaults": {dtype: params}}
    """
    from core.wave_inspect import (_detect_inflections, _detect_threshold_crossings, _select_nth_pts,
                                   _slope_diff_t)
    empty = {"shape": "", "cands": [], "defaults": {}, "spark": [], "cycle_ms": 0.0, "wpost": 300}
    if len(waves) > 2:
        waves = waves[:-1]          # 最後のサイクルは途中で切れている
    sw = []
    for c in waves[:60]:
        if var not in c.columns:
            continue
        t = c["time_offset_ms"].to_numpy(dtype=float)
        v = c[var].to_numpy(dtype=float)
        ok = np.isfinite(v)
        if ok.sum() >= 10:
            sw.append((t[ok], v[ok]))
    if len(sw) < 2:
        return empty
    allv = np.concatenate([v for _, v in sw])
    p1, p99 = np.percentile(allv, [1, 99])
    rng = float(p99 - p1)
    if not np.isfinite(rng) or rng <= 0:
        return empty
    cyc_ms = float(np.median([t[-1] for t, _ in sw]))
    dt = float(np.median(np.concatenate([np.diff(t) for t, _ in sw]))) or 1.0
    grid = np.linspace(0, cyc_ms, 200)
    mat = np.array([np.where(grid <= t[-1], np.interp(grid, t, v), np.nan) for t, v in sw])
    mean = np.nanmean(mat, axis=0)
    # 形はサイクルごとに判定して多数決（タイミングがずれると平均の形が崩れるため）
    votes = []
    for _, v in sw:
        r = (v - p1) / rng
        k = max(3, len(r) // 30)
        st_, en_ = float(np.mean(r[:k])), float(np.mean(r[-k:]))
        if r.max() >= 0.7 and st_ < 0.3 and en_ < 0.3 and np.mean(r > 0.9) < 0.08:
            votes.append("山形")
        elif r.min() <= 0.3 and st_ > 0.7 and en_ > 0.7 and np.mean(r < 0.1) < 0.08:
            votes.append("谷形")
        elif st_ < 0.3 and np.mean(r > 0.8) >= 0.08:
            votes.append("台形")
        elif st_ > 0.7 and np.mean(r < 0.2) >= 0.08:
            votes.append("くぼみ")
        else:
            votes.append("その他")
    kinds, counts = np.unique(votes, return_counts=True)
    shape = str(kinds[np.argmax(counts)])
    if counts.max() < 0.6 * len(sw):
        shape = "その他"
    n = len(sw)
    wpost = int(min(2000, max(300, np.ceil(cyc_ms * 1.05 / 50) * 50)))

    def summarize(pts):
        got = [p for p in pts if p is not None]
        det = len(got) / n
        if not got:
            return det, np.nan, np.nan, 100.0
        ts = np.array([p[0] for p in got])
        q75, q25 = np.percentile(ts, [75, 25])
        return det, float(np.median(ts)), float(np.median([p[1] for p in got])), float((q75 - q25) / cyc_ms * 100)

    # 波形検査の各タイプの設定値（タイプを変えても使えるように全タイプ分を作る）
    level = _nice(p1 + 0.5 * rng)
    sm = int(min(50, max(3, round(5 / dt))))
    nlr = int(min(100, max(3, round(15 / dt))))
    defaults = {
        "最大値点": {},
        "最小値点": {},
        "閾値超え検出": {"tv": level, "tdir": "下降 ↓" if shape in ("谷形", "くぼみ") else "上昇 ↑", "nth": 1},
        "傾き変化点": {"smooth": sm, "nleft": nlr, "nright": nlr, "thresh": 0.0, "nth": 1,
                     "dir_inc": True, "dir_dec": False, "use_vrange": True,
                     "vrange_lo": _nice(p1 + 0.1 * rng), "vrange_hi": _nice(p1 + 0.9 * rng)},
    }
    cands = []

    def add(dtype, name, params, pts, note, conf=None):
        det, tm, vm, spread = summarize(pts)
        c = conf or _conf(det, spread)
        cands.append({"dtype": dtype, "name": name, "params": params, "t_ms": tm, "v": vm,
                      "det_rate": det, "confidence": c, "include": c != "低", "note": note})

    # 山形・谷形
    if shape in ("山形", "谷形"):
        peak = shape == "山形"
        pts, prom = [], []
        for t, v in sw:
            i = int(np.argmax(v) if peak else np.argmin(v))
            pts.append((float(t[i]), float(v[i])))
            prom.append(abs(v[i] - np.median(v[: max(3, len(v) // 20)])) >= 0.5 * rng)
        share = float(np.mean(prom))
        conf = "高" if share >= 0.95 else "中" if share >= 0.8 else "低"
        add("最大値点" if peak else "最小値点", "ピーク" if peak else "谷", {}, pts,
            "山形の波形。ピークの値と時刻を追います" if peak else "谷形の波形。谷の値と時刻を追います", conf)
    elif shape in ("台形", "くぼみ"):
        up = shape == "台形"
        for d, nm in (("上昇 ↑", "立ち上がり" if up else "復帰"), ("下降 ↓", "立ち下がり" if up else "低下")):
            pts = []
            for t, v in sw:
                cr = _detect_threshold_crossings(t, v, level, direction="rise" if "上昇" in d else "fall")
                sel = _select_nth_pts(cr, 1)
                pts.append(sel[0] if sel else None)
            add("閾値超え検出", nm, {"tv": level, "tdir": d, "nth": 1}, pts,
                f"値が {level:g} を{'上回る' if '上昇' in d else '下回る'}時刻を追います")
        if not up:
            cands.reverse()

    # 途中で傾きが急に変わる点（立ち上がり・立ち下がりの途中の折れ曲がり。始まりと頂上は除く）
    if shape in ("山形", "台形", "谷形", "くぼみ"):
        rising = shape in ("山形", "台形")
        lo_v, hi_v = p1 + 0.15 * rng, p1 + 0.85 * rng
        peaks, noise, signs = [], [], []
        for t, v in sw:
            w = max(1, min(sm, len(v) // 2))
            vs = np.convolve(v, np.ones(w) / w, mode="same")
            # 最初に 15% を越えてから 85% に届くまで（谷形は逆向き）を 1 本の変化区間とみる
            a_ = np.argmax(vs >= lo_v) if rising else np.argmax(vs <= hi_v)
            b_ = (a_ + np.argmax(vs[a_:] >= hi_v)) if rising else (a_ + np.argmax(vs[a_:] <= lo_v))
            if b_ - a_ < 6 * nlr:
                continue
            tm_, _, _, d_ = _slope_diff_t(t, vs, nlr, nlr)
            seg = d_[max(0, a_ + nlr):max(0, b_ - 3 * nlr)]
            if len(seg) < 5:
                continue
            k = int(np.argmax(np.abs(seg)))
            peaks.append(abs(float(seg[k])))
            signs.append(np.sign(seg[k]))
            noise.append(float(np.median(np.abs(seg))) + 1e-12)
        if len(peaks) >= 0.8 * n and np.median(np.array(peaks) / np.array(noise)) >= 6:
            inc = bool(np.median(signs) > 0)
            th = _nice(0.5 * float(np.median(peaks)))
            params = dict(defaults["傾き変化点"], thresh=th, dir_inc=inc, dir_dec=not inc,
                          vrange_lo=_nice(lo_v), vrange_hi=_nice(hi_v))
            defaults["傾き変化点"] = params
            pts = []
            for t, v in sw:
                ts = _detect_inflections(t, v, smooth_w=sm, n_left=nlr, n_right=nlr, threshold=th,
                                         detect_increase=inc, detect_decrease=not inc)
                pp = [(float(x), float(np.interp(x, t, v))) for x in ts]
                pp = [q for q in pp if lo_v <= q[1] <= hi_v]
                sel = _select_nth_pts(pp, 1)
                pts.append(sel[0] if sel else None)
            add("傾き変化点", "傾き変化", params, pts,
                "変化の途中で傾きが急に" + ("大きくなる" if inc else "小さくなる") + "点（接触・切り替わりなど）を追います")

    spark = [float(x) for x in mean[:: max(1, len(mean) // 40)] if np.isfinite(x)]
    return {"shape": shape, "cycle_ms": cyc_ms, "wpost": wpost, "spark": spark, "cands": cands,
            "defaults": defaults}


def analog_vars(entry: dict) -> list:
    """変化のあるアナログ変数（状態番号・カウンタ・一定の変数は外す）"""
    prof = variable_profile(entry["df"], tuple(entry["col_types"].items()))
    return [v for v in prof.index if prof.loc[v, "kind"] == "analog"]


def trigger_for_var(var: str, processes: dict) -> tuple:
    """変数の工程を名前の先頭で推定し、(工程名, トリガー, エッジ) を返す。分からなければ最初の工程"""
    for name, p in processes.items():
        if var.startswith(name) and p.get("trigger_col"):
            return name, p["trigger_col"], p.get("edge", "RISE")
    for name, p in processes.items():
        if p.get("trigger_col"):
            return name, p["trigger_col"], p.get("edge", "RISE")
    return "", "", "RISE"


def wave_candidates(entry: dict, processes: dict) -> dict:
    """基準データのアナログ変数ごとに、工程のサイクル開始信号を基準にした波形の仮説を返す。
    processes: {工程名: {"trigger_col", "edge"}}。返り値: {変数: {"process", "trigger", "edge", **wave_hypotheses}}
    """
    pairs, owner = [], {}
    for v in analog_vars(entry):
        pname, trig, edge = trigger_for_var(v, processes)
        if trig:
            pairs.append((v, trig, edge))
            owner[v] = pname
    res = wave_hypotheses_all(entry["df"], tuple(pairs))
    return {v: dict(res[v], process=owner[v], trigger=t, edge=e) for v, t, e in pairs}
