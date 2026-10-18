"""
core/wave_inspect.py — 波形検査（アナログ波形の検出点・良品範囲・検査条件）

旧メイン画面の波形検査タブの補助関数・ダイアログをそのまま移したもの（画面は views/wave.py）。
検査ロジックの互換性（設定ファイル・session_state のキー wvol_*／wi_*）を保つため、中身は変更していない。
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analyzer import detect_cycles, get_cycle_waveforms, normalize_bool_series
from core.state import pk

# 検査ダイアログが参照する列一覧（ページ側が set_columns() で設定する）
bool_cols: list = []
num_cols: list = []


def set_columns(bools: list, nums: list):
    global bool_cols, num_cols
    bool_cols, num_cols = list(bools), list(nums)

_WI_SAVE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wi_saved_setups.json")
_WI_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wi_current_config.json"
)

def _wi_save_to_file(setups: list) -> None:
    try:
        import json as _json
        with open(_WI_SAVE_FILE, "w", encoding="utf-8") as f:
            _json.dump(setups, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _wi_load_from_file() -> list:
    try:
        import json as _json
        if os.path.isfile(_WI_SAVE_FILE):
            with open(_WI_SAVE_FILE, "r", encoding="utf-8") as f:
                return _json.load(f)
    except Exception:
        pass
    return []


_WI_SKIP_KEY_SUFFIXES = (
    "_t_det_add",   # ＋ 検出点を追加 ボタン（時間軸）
    "_xy_det_add",  # ＋ 検出点を追加 ボタン（XY）
    "_del",         # 🗑 削除ボタン
    "_bl_reg",      # 📌 現データを基準として登録 ボタン（時間軸）
    "_bl_del",      # 基準削除 ボタン（時間軸）
    "_xy_bl_reg",   # 📌 現データを基準として登録 ボタン（XY）
    "_xy_bl_del",   # 基準削除 ボタン（XY）
)


def _wi_skip_key(k: str) -> bool:
    """ウィジェット（ボタン等）として使われるキーは session_state への
    直接書き込みが Streamlit に禁止されているためスキップする。"""
    return any(k.endswith(s) for s in _WI_SKIP_KEY_SUFFIXES)


def _wi_save_config(ss, num_cols_list: list) -> None:
    """現在の波形検査設定を wi_current_config.json に保存する。"""
    snap: dict = {}
    for _v in num_cols_list:
        _vk = f"wvol___global_{_v}"
        for _k, _val in ss.items():
            if isinstance(_k, str) and _k.startswith(_vk) and not _wi_skip_key(_k):
                snap[_k] = _val
    try:
        import json as _j
        with open(_WI_CONFIG_FILE, "w", encoding="utf-8") as _f:
            _j.dump(snap, _f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _wi_restore_config(ss) -> bool:
    """wi_current_config.json から波形検査設定を session_state へ復元する。
    復元できたら True を返す。"""
    try:
        import json as _j
        if not os.path.isfile(_WI_CONFIG_FILE):
            return False
        with open(_WI_CONFIG_FILE, "r", encoding="utf-8") as _f:
            snap = _j.load(_f)
        for _k, _v in snap.items():
            if _wi_skip_key(_k):
                continue  # ボタンキーは書き込み禁止のためスキップ
            ss[_k] = _v
        return True
    except Exception:
        return False

def mean_waveform(waves: list, var: str):
    if not waves:
        return [], []
    mt  = max(float(c["time_offset_ms"].iloc[-1]) for c in waves)
    ta  = np.linspace(0, mt, 300)
    mat = np.zeros((len(waves), len(ta)), dtype=np.float32)
    for j, c in enumerate(waves):
        t_arr = c["time_offset_ms"].values
        v_arr = c[var].values.astype(np.float32)
        idx   = np.searchsorted(t_arr, ta).clip(0, len(t_arr) - 1)
        mat[j] = v_arr[idx]
    return ta, mat.mean(axis=0)

@st.cache_data
def cached_detect_cycles(df: pd.DataFrame, trigger_col: str, edge: str) -> list:
    return list(detect_cycles(df, trigger_col, edge))

@st.cache_data
def cached_waveforms(df: pd.DataFrame, trigger_col: str, edge: str,
                     target_vars: tuple) -> list:
    idx = pd.Index(cached_detect_cycles(df, trigger_col, edge))
    return get_cycle_waveforms(df, idx, list(target_vars))

@st.cache_data
def make_mini_chart(df: pd.DataFrame, var: str, height: int = 65) -> go.Figure:
    sample = df.head(2000)
    try:
        vals    = normalize_bool_series(sample[var])
        y_range = [-0.1, 1.3]
        color   = "#4472C4"
    except Exception:
        vals    = sample[var]
        y_range = None
        color   = "#ED7D31"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(vals))), y=vals, fill="tozeroy",
                             mode="lines", line=dict(width=1, color=color)))
    lo = dict(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
              plot_bgcolor="#f5f7fa", paper_bgcolor="rgba(0,0,0,0)",
              xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
              yaxis=dict(showticklabels=False, showgrid=False, zeroline=False))
    if y_range:
        lo["yaxis"]["range"] = y_range
    fig.update_layout(**lo)
    return fig

def calc_nice_bins(data: np.ndarray, key: str = "") -> int:
    """Freedman-Diaconis ルールで bin 数を計算（見た目がきれいな刻み幅）"""
    # まずセッション上書きを確認
    override = st.session_state.get(f"_bins_{key}") if key else None
    if override and override > 0:
        return int(override)
    n = len(data)
    if n <= 4:
        return 5
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    data_range = float(np.max(data) - np.min(data))
    if iqr <= 0 or data_range <= 0:
        return max(5, min(30, int(1 + 3.32 * np.log10(max(n, 2)))))
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
    # nice な幅に丸める
    mag = 10.0 ** np.floor(np.log10(bin_width))
    for f in [1, 2, 5, 10]:
        nw = f * mag
        if nw >= bin_width * 0.8:
            bin_width = nw
            break
    return max(5, min(50, int(np.ceil(data_range / bin_width))))

def _get_wv_cfg(pname: str, step_name: str, var: str) -> dict:
    """_render_waveform_overlay が使うウィジェットキーから設定を読み取る（デフォルト付き）。
    スタンドアロン波形検査タブ（wvol_{pname}_{var}）を優先し、
    なければステップ別キー（wvol_{pname}_{step_name}_{var}）を使う。
    """
    _sa_vkey   = f"wvol_{pname}_{var}"           # スタンドアロン（波形検査タブ）
    _step_vkey = f"wvol_{pname}_{step_name}_{var}"
    # スタンドアロンキーに検出点が登録されていればそちらを優先
    _vkey = _sa_vkey if st.session_state.get(f"{_sa_vkey}_t_det_list") else _step_vkey
    return {
        "win_pre":        int(st.session_state.get(f"{_vkey}_wpre",   50)),
        "win_post":       int(st.session_state.get(f"{_vkey}_wpost",  300)),
        "insp_type":      st.session_state.get(f"{_vkey}_itype",  "時間軸"),
        "insp_s":         float(st.session_state.get(f"{_vkey}_is",     0)),
        "insp_e":         float(st.session_state.get(f"{_vkey}_ie",   200)),
        "insp_trig_val":  float(st.session_state.get(f"{_vkey}_itv",   1.0)),
        "insp_trig_pre":  float(st.session_state.get(f"{_vkey}_itpre", 10)),
        "insp_trig_post": float(st.session_state.get(f"{_vkey}_itpost",150)),
        "good_mode":      st.session_state.get(f"{_vkey}_gmode", "手動入力"),
        "good_lo":        float(st.session_state.get(f"{_vkey}_glo",   0.0)),
        "good_hi":        float(st.session_state.get(f"{_vkey}_ghi",   0.0)),
        "good_nsig":      float(st.session_state.get(f"{_vkey}_nsig",  3.0)),
    }


def _compute_wv_ng(df: pd.DataFrame, trigger_col: str, edge: str,
                   step: dict, pname: str, result_df: pd.DataFrame) -> dict:
    """
    波形NG判定をUIなしで実行。新データ評価・傾向解析から呼び出す。

    Returns:
        {var: {"ng_count": int, "total": int, "ng_flags": list[bool], "peaks": list[float]}}
        peaks = 各サイクルの検査ウィンドウ内最大値
    """
    mode          = step.get("mode", "single")
    name          = step.get("name", "")
    waveform_vars = [v for v in step.get("waveform_vars", []) if v in df.columns]
    if not waveform_vars:
        return {}

    start_col = f"{name}_遅れ[ms]" if mode == "single" else f"{name}_start[ms]"
    if start_col not in result_df.columns:
        return {}

    start_offsets = result_df[start_col].values
    try:
        waveforms = cached_waveforms(df, trigger_col, edge, tuple(waveform_vars))
    except Exception:
        return {}

    n_cyc = min(len(waveforms), len(start_offsets))
    if n_cyc == 0:
        return {}

    _wv_bl = st.session_state.get(pk(pname, "wv_baseline"), {})
    out    = {}

    for var in waveform_vars:
        cfg      = _get_wv_cfg(pname, name, var)
        win_pre  = cfg["win_pre"]
        win_post = cfg["win_post"]

        # 波形データ抽出（ステップ開始基準）
        step_waves = []
        for i in range(n_cyc):
            cyc      = waveforms[i]
            step_off = float(start_offsets[i]) if not np.isnan(start_offsets[i]) else None
            if step_off is None or var not in cyc.columns:
                continue
            t_step = cyc["time_offset_ms"].values - step_off
            v_arr  = cyc[var].values.astype(np.float64)
            mask   = (t_step >= -win_pre) & (t_step <= win_post)
            if mask.sum() < 2:
                continue
            step_waves.append((t_step[mask], v_arr[mask]))

        if not step_waves:
            continue

        # エンベロープ計算
        t_common = np.linspace(-win_pre, win_post, 500)
        mat = np.full((len(step_waves), len(t_common)), np.nan)
        for j, (t_sw, v_sw) in enumerate(step_waves):
            idx      = np.searchsorted(t_sw, t_common).clip(0, len(t_sw) - 1)
            in_range = (t_common >= t_sw[0]) & (t_common <= t_sw[-1])
            mat[j, in_range] = v_sw[idx[in_range]]
        mean_v = np.nanmean(mat, axis=0)
        std_v  = np.nanstd(mat,  axis=0)

        # 良品範囲
        good_mode = cfg["good_mode"]
        if good_mode == "自動（基準±Nσ）":
            good_nsig = cfg["good_nsig"]
            bl        = _wv_bl.get(f"{pname}_{name}_{var}", {})
            if bl:
                _t_bl   = np.array(bl["t"])
                _m_bl   = np.interp(t_common, _t_bl, np.array(bl["mean"]))
                _s_bl   = np.interp(t_common, _t_bl, np.array(bl["std"]))
                env_hi  = _m_bl + good_nsig * _s_bl
                env_lo  = _m_bl - good_nsig * _s_bl
            else:
                env_hi  = mean_v + good_nsig * std_v
                env_lo  = mean_v - good_nsig * std_v
            has_envelope = True
        else:
            glo = cfg["good_lo"]
            ghi = cfg["good_hi"]
            has_envelope = glo < ghi
            env_hi = np.full_like(t_common, ghi) if has_envelope else None
            env_lo = np.full_like(t_common, glo) if has_envelope else None

        # 検査ウィンドウ・NG判定・ピーク収集
        insp_type = cfg["insp_type"]
        ng_flags  = []
        peaks     = []

        for _j, (t_sw, v_sw) in enumerate(step_waves):
            # 検査ウィンドウ確定
            if insp_type == "時間軸":
                _win = (cfg["insp_s"], cfg["insp_e"])
            else:
                _over = np.where(v_sw >= cfg["insp_trig_val"])[0]
                if len(_over) > 0:
                    _t0  = float(t_sw[_over[0]])
                    _win = (_t0 - cfg["insp_trig_pre"], _t0 + cfg["insp_trig_post"])
                else:
                    _win = None

            # ピーク収集（検査ウィンドウ内最大値）
            if _win is not None:
                _ws, _we  = _win
                _mask_i   = (t_sw >= _ws) & (t_sw <= _we)
                peaks.append(float(np.max(v_sw[_mask_i])) if _mask_i.sum() > 0 else float(np.max(v_sw)))
            else:
                peaks.append(float(np.max(v_sw)))

            # NG判定
            if not has_envelope or _win is None:
                ng_flags.append(False)
                continue
            _ws, _we = _win
            _mask_i  = (t_sw >= _ws) & (t_sw <= _we)
            if _mask_i.sum() == 0:
                ng_flags.append(False)
                continue
            _t_i  = t_sw[_mask_i]
            _v_i  = v_sw[_mask_i]
            _hi_i = np.interp(_t_i, t_common, env_hi)
            _lo_i = np.interp(_t_i, t_common, env_lo)
            ng_flags.append(bool(np.any(_v_i > _hi_i) or np.any(_v_i < _lo_i)))

        out[var] = {
            "ng_count": sum(ng_flags),
            "total":    len(ng_flags),
            "ng_flags": ng_flags,
            "peaks":    peaks,
        }

    return out


# ═══════════════════════════════════════════════════════════════
# 波形特徴量検出ヘルパー
# ═══════════════════════════════════════════════════════════════

def _slope_diff_t(t: np.ndarray, vs: np.ndarray,
                  n_left: int, n_right: int) -> tuple:
    """
    時間軸波形の左右傾き差を計算（平滑化済み配列を受け取る）。
    各点 i について（有効範囲: n_left ≤ i < len-n_right）:
        左傾き L[i] = (vs[i] - vs[i-n_left])  / (t[i] - t[i-n_left])
        右傾き R[i] = (vs[i+n_right] - vs[i])  / (t[i+n_right] - t[i])
        差 D[i] = R[i] - L[i]
    Returns: (t_mid, L_arr, R_arr, D_arr)
    """
    idx  = np.arange(n_left, len(t) - n_right)
    dt_l = t[idx] - t[idx - n_left];   dt_l = np.where(dt_l != 0, dt_l, 1e-10)
    dt_r = t[idx + n_right] - t[idx];  dt_r = np.where(dt_r != 0, dt_r, 1e-10)
    L = (vs[idx] - vs[idx - n_left])  / dt_l
    R = (vs[idx + n_right] - vs[idx]) / dt_r
    return t[idx], L, R, R - L


def _slope_diff_xy(xs: np.ndarray, ys: np.ndarray,
                   n_left: int, n_right: int) -> tuple:
    """
    XY軸（X昇順ソート済み）の左右傾き差を計算（平滑化済み配列を受け取る）。
    Returns: (x_mid, L_arr, R_arr, D_arr)
    """
    idx  = np.arange(n_left, len(xs) - n_right)
    dx_l = xs[idx] - xs[idx - n_left];   dx_l = np.where(dx_l != 0, dx_l, 1e-10)
    dx_r = xs[idx + n_right] - xs[idx];  dx_r = np.where(dx_r != 0, dx_r, 1e-10)
    L = (ys[idx] - ys[idx - n_left])  / dx_l
    R = (ys[idx + n_right] - ys[idx]) / dx_r
    return xs[idx], L, R, R - L


def _detect_inflections(t: np.ndarray, v: np.ndarray,
                        smooth_w: int = 5,
                        n_left: int = 3, n_right: int = 3,
                        threshold: float = 0.0,
                        range_s: float = None, range_e: float = None,
                        detect_increase: bool = True,
                        detect_decrease: bool = True) -> np.ndarray:
    """
    左右傾き差による傾き変化点検出（時間軸）。
    detect_increase=True: D = R-L > +threshold (傾きが増加する点)
    detect_decrease=True: D = R-L < -threshold (傾きが減少する点)
    両方 True のとき |D| > threshold（従来動作）。
    threshold = 0 または方向フラグ両方 False の場合は空配列を返す。
    """
    if len(t) < n_left + n_right + 1 or threshold <= 0:
        return np.array([])
    if not detect_increase and not detect_decrease:
        return np.array([])
    w  = max(1, min(smooth_w, len(v) // 2))
    vs = np.convolve(v, np.ones(w) / w, mode="same") if w > 1 else v.copy()
    t_mid, _, _, D = _slope_diff_t(t, vs, n_left, n_right)
    if detect_increase and detect_decrease:
        mask = np.abs(D) > threshold
    elif detect_increase:
        mask = D > threshold
    else:
        mask = D < -threshold
    if range_s is not None:
        mask &= (t_mid >= range_s)
    if range_e is not None:
        mask &= (t_mid <= range_e)
    return t_mid[mask]


def _detect_xy_inflections(x: np.ndarray, y: np.ndarray,
                            smooth_w: int = 5,
                            n_left: int = 3, n_right: int = 3,
                            threshold: float = 0.0,
                            range_s: float = None, range_e: float = None,
                            detect_increase: bool = True,
                            detect_decrease: bool = True):
    """
    左右傾き差による傾き変化点検出（XY軸）。
    detect_increase=True: D = R-L > +threshold (傾きが増加する点)
    detect_decrease=True: D = R-L < -threshold (傾きが減少する点)
    Returns: (x_pts, diff_vals)
    """
    if len(x) < n_left + n_right + 1 or threshold <= 0:
        return np.array([]), np.array([])
    if not detect_increase and not detect_decrease:
        return np.array([]), np.array([])
    si = np.argsort(x)
    xs, ys = x[si], y[si]
    w  = max(1, min(smooth_w, len(ys) // 2))
    ys = np.convolve(ys, np.ones(w) / w, mode="same") if w > 1 else ys.copy()
    x_mid, _, _, D = _slope_diff_xy(xs, ys, n_left, n_right)
    if detect_increase and detect_decrease:
        mask = np.abs(D) > threshold
    elif detect_increase:
        mask = D > threshold
    else:
        mask = D < -threshold
    if range_s is not None:
        mask &= (x_mid >= range_s)
    if range_e is not None:
        mask &= (x_mid <= range_e)
    return x_mid[mask], D[mask]


def _slope_diff_max_ref_t(waves_sample, smooth_w: int = 5,
                           n_left: int = 3, n_right: int = 3) -> float:
    """時間軸波形サンプルの最大 |傾き差| を推定（閾値設定の参考値）"""
    ref = 0.0
    for (t_r, v_r) in waves_sample:
        if len(t_r) < n_left + n_right + 1:
            continue
        w  = max(1, min(smooth_w, len(v_r) // 2))
        vs = np.convolve(v_r, np.ones(w) / w, mode="same") if w > 1 else v_r.copy()
        _, _, _, D = _slope_diff_t(t_r, vs, n_left, n_right)
        if len(D):
            ref = max(ref, float(np.nanmax(np.abs(D))))
    return ref


def _slope_diff_max_ref_xy(xy_waves_sample, smooth_w: int = 5,
                            n_left: int = 3, n_right: int = 3) -> float:
    """XY波形サンプルの最大 |傾き差| を推定（閾値設定の参考値）"""
    ref = 0.0
    for (xw, yw) in xy_waves_sample:
        if len(xw) < n_left + n_right + 1:
            continue
        si = np.argsort(xw)
        xs, ys = xw[si], yw[si]
        w  = max(1, min(smooth_w, len(ys) // 2))
        ys = np.convolve(ys, np.ones(w) / w, mode="same") if w > 1 else ys.copy()
        _, _, _, D = _slope_diff_xy(xs, ys, n_left, n_right)
        if len(D):
            ref = max(ref, float(np.nanmax(np.abs(D))))
    return ref


# 旧関数の別名（_compute_wv_ng などから参照されている場合のフォールバック）
_d2v_max_ref = _slope_diff_max_ref_t
_d2y_max_ref = _slope_diff_max_ref_xy


def _select_nth_pts(pts: list, nth: int) -> list:
    """
    pts: list of (coord, value) tuples, coord 昇順ソート済み。
    nth: 0=全て, 1=最初, 2=2番目, ..., -1=最後, -2=後ろから2番目。
    IndexError の場合は空リストを返す。
    """
    if nth == 0 or not pts:
        return pts
    try:
        return [pts[nth - 1 if nth > 0 else nth]]
    except IndexError:
        return []


def _detect_threshold_crossings(coord: np.ndarray, v: np.ndarray,
                                 threshold: float,
                                 direction: str = "rise",
                                 range_s=None, range_e=None) -> list:
    """
    coord/v が同長の配列（coord は昇順）でスカラー閾値 threshold を
    v が超える（上昇）または下回る（下降）座標を返す。
    direction: "rise" | "fall" | "both"
    Returns: list of (coord_cross, threshold) tuples
    """
    if len(coord) < 2:
        return []
    mask = np.ones(len(coord), dtype=bool)
    if range_s is not None:
        mask &= coord >= range_s
    if range_e is not None:
        mask &= coord <= range_e
    coord, v = coord[mask], v[mask]
    if len(coord) < 2:
        return []
    crossings = []
    for i in range(1, len(v)):
        dv = float(v[i] - v[i - 1])
        dc = float(coord[i] - coord[i - 1])
        if direction in ("rise", "both") and v[i - 1] < threshold <= v[i]:
            c = (float(coord[i - 1]) + (threshold - float(v[i - 1])) / dv * dc
                 if dv != 0 else float(coord[i]))
            crossings.append((c, float(threshold)))
        elif direction in ("fall", "both") and v[i - 1] >= threshold > v[i]:
            c = (float(coord[i - 1]) + (float(v[i - 1]) - threshold) / (-dv) * dc
                 if dv != 0 else float(coord[i]))
            crossings.append((c, float(threshold)))
    return crossings


def _render_item_insp_win_t(dkey: str, step_waves: list,
                             insp_windows_global: list) -> list:
    """
    検出アイテム単位の検査ウィンドウ UI を描画し、有効な insp_windows リストを返す。
    「個別設定を使う」が OFF の場合はグローバルウィンドウをそのまま返す。
    時間軸タブ用。
    """
    import streamlit as _st
    _st.markdown("---")
    _iw_hd, _iw_cb = _st.columns([3, 2])
    with _iw_hd:
        _st.markdown("**🔍 検査ウィンドウ（このアイテム専用）**")
    with _iw_cb:
        _own = _st.checkbox("個別設定を使う", key=f"{dkey}_own_win")
    if not _own:
        return insp_windows_global

    _iw_type = _st.radio("種別", ["時間軸", "値トリガ"],
                          horizontal=True, key=f"{dkey}_iw_type")
    if _iw_type == "時間軸":
        _c1, _c2 = _st.columns(2)
        with _c1:
            _st.number_input("開始 [ms]", value=0.0, step=5.0, key=f"{dkey}_iw_s")
        with _c2:
            _st.number_input("終了 [ms]", value=200.0, step=5.0, key=f"{dkey}_iw_e")
        _s = float(_st.session_state.get(f"{dkey}_iw_s", 0.0))
        _e = float(_st.session_state.get(f"{dkey}_iw_e", 200.0))
        return [(_s, _e)] * len(step_waves)
    else:
        _c1, _c2, _c3 = _st.columns(3)
        with _c1:
            _st.number_input("トリガ閾値", value=1.0, step=0.5, key=f"{dkey}_iw_tv")
        with _c2:
            _st.number_input("前 [ms]", value=10.0, step=5.0, key=f"{dkey}_iw_pre")
        with _c3:
            _st.number_input("後 [ms]", value=150.0, step=5.0, key=f"{dkey}_iw_post")
        _tv  = float(_st.session_state.get(f"{dkey}_iw_tv",   1.0))
        _pre = float(_st.session_state.get(f"{dkey}_iw_pre",  10.0))
        _pst = float(_st.session_state.get(f"{dkey}_iw_post", 150.0))
        result = []
        for (t_sw, v_sw) in step_waves:
            over = np.where(v_sw >= _tv)[0]
            if len(over) > 0:
                t0 = float(t_sw[over[0]])
                result.append((t0 - _pre, t0 + _pst))
            else:
                result.append(None)
        return result


def _render_item_insp_win_xy(dkey: str, x_min: float, x_max: float,
                              xs_global: float, xe_global: float,
                              use_global: bool) -> tuple:
    """
    検出アイテム単位の検査ウィンドウ UI を描画し、(eff_xs, eff_xe, eff_use) を返す。
    XY タブ用。
    """
    import streamlit as _st
    _st.markdown("---")
    _iw_hd, _iw_cb = _st.columns([3, 2])
    with _iw_hd:
        _st.markdown("**🔍 検査ウィンドウ（このアイテム専用）**")
    with _iw_cb:
        _own = _st.checkbox("個別設定を使う", key=f"{dkey}_own_win")
    if not _own:
        return xs_global, xe_global, use_global

    _c1, _c2 = _st.columns(2)
    with _c1:
        _st.number_input("X 開始", value=float(x_min), step=0.5, key=f"{dkey}_iw_xs")
    with _c2:
        _st.number_input("X 終了", value=float(x_max), step=0.5, key=f"{dkey}_iw_xe")
    _xs = float(_st.session_state.get(f"{dkey}_iw_xs", x_min))
    _xe = float(_st.session_state.get(f"{dkey}_iw_xe", x_max))
    return _xs, _xe, (_xs < _xe)


# ═══════════════════════════════════════════════════════════════
# 波形検査 検出点 — 傾向解析用ヘルパー関数
# ═══════════════════════════════════════════════════════════════

def _detect_point_for_trend(step_waves, dtype, dkey, ss):
    """
    各サイクルの波形から検出点 (t, v) を取得する（傾向解析用）。
    step_waves : list of (t_arr: np.ndarray, v_arr: np.ndarray)
    dtype      : "傾き変化点" | "閾値超え検出" | "最大値点" | "最小値点"
    dkey       : session_state prefix (e.g. "wvol___global_Var_td0")
    ss         : st.session_state (dict-like)
    Returns    : list of (t, v) | None, length == len(step_waves)
    """
    result = []

    if dtype == "傾き変化点":
        # ※ trend_on=True で呼ばれる前提なので _on チェックは行わない
        _sm    = int(ss.get(f"{dkey}_smooth", 5))
        _nl    = int(ss.get(f"{dkey}_nleft",  3))
        _nr    = int(ss.get(f"{dkey}_nright", 3))
        _th    = float(ss.get(f"{dkey}_thresh", 0.0))
        _nth   = int(ss.get(f"{dkey}_nth", 0))
        _dinc  = bool(ss.get(f"{dkey}_dir_inc", True))
        _ddec  = bool(ss.get(f"{dkey}_dir_dec", True))
        _use_r = bool(ss.get(f"{dkey}_use_range", False))
        _rs    = float(ss.get(f"{dkey}_range_s",   0.0)) if _use_r else None
        _re    = float(ss.get(f"{dkey}_range_e", 200.0)) if _use_r else None
        _use_v = bool(ss.get(f"{dkey}_use_vrange", False))
        _rvlo  = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_v else None
        _rvhi  = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_v else None
        if _th <= 0.0:
            return [None] * len(step_waves)
        for (t_sw, v_sw) in step_waves:
            try:
                _ts = _detect_inflections(
                    t_sw, v_sw, smooth_w=_sm, n_left=_nl, n_right=_nr,
                    threshold=_th, range_s=_rs, range_e=_re,
                    detect_increase=_dinc, detect_decrease=_ddec)
                _pts = [(float(ti), float(np.interp(ti, t_sw, v_sw))) for ti in _ts]
                if _rvlo is not None and _rvhi is not None:
                    _pts = [(ti, vi) for ti, vi in _pts if _rvlo <= vi <= _rvhi]
                _sel = _select_nth_pts(_pts, _nth)
                result.append(_sel[0] if _sel else None)
            except Exception:
                result.append(None)
        return result

    elif dtype == "閾値超え検出":
        # ※ trend_on=True で呼ばれる前提なので _on チェックは行わない
        _tv       = float(ss.get(f"{dkey}_tv", 1.0))
        _tdir_raw = str(ss.get(f"{dkey}_tdir", "上昇 ↑"))
        _tdir     = ("rise" if "上昇" in _tdir_raw
                     else "fall" if "下降" in _tdir_raw else "both")
        _nth   = int(ss.get(f"{dkey}_nth", 1))
        _use_r = bool(ss.get(f"{dkey}_use_range", False))
        _rs    = float(ss.get(f"{dkey}_range_s",   0.0)) if _use_r else None
        _re    = float(ss.get(f"{dkey}_range_e", 200.0)) if _use_r else None
        _use_v = bool(ss.get(f"{dkey}_use_vrange", False))
        _rvlo  = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_v else None
        _rvhi  = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_v else None
        for (t_sw, v_sw) in step_waves:
            try:
                _crs = _detect_threshold_crossings(
                    t_sw, v_sw, _tv, direction=_tdir, range_s=_rs, range_e=_re)
                if _rvlo is not None and _rvhi is not None:
                    _crs = [(tc, vc) for tc, vc in _crs if _rvlo <= vc <= _rvhi]
                _sel = _select_nth_pts(_crs, _nth)
                result.append(_sel[0] if _sel else None)
            except Exception:
                result.append(None)
        return result

    elif dtype in ("最大値点", "最小値点"):
        # ※ trend_on=True で呼ばれる前提なので _on チェックは行わない
        _is_max   = (dtype == "最大値点")
        _use_st   = bool(ss.get(f"{dkey}_use_range", False))
        _srng_s   = float(ss.get(f"{dkey}_range_s",   0.0)) if _use_st else None
        _srng_e   = float(ss.get(f"{dkey}_range_e", 200.0)) if _use_st else None
        _use_sv   = bool(ss.get(f"{dkey}_use_vrange", False))
        _srng_vlo = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_sv else None
        _srng_vhi = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_sv else None
        for (t_sw, v_sw) in step_waves:
            try:
                _tp_w, _vp_w = t_sw, v_sw
                if _srng_s is not None and _srng_e is not None:
                    _mp_t = (t_sw >= _srng_s) & (t_sw <= _srng_e)
                    if _mp_t.sum() > 0:
                        _tp_w = t_sw[_mp_t]; _vp_w = v_sw[_mp_t]
                if _srng_vlo is not None and _srng_vhi is not None:
                    _mp_v = (_vp_w >= _srng_vlo) & (_vp_w <= _srng_vhi)
                    if _mp_v.sum() > 0:
                        _tp_w = _tp_w[_mp_v]; _vp_w = _vp_w[_mp_v]
                if len(_vp_w) == 0:
                    result.append(None); continue
                _idx = int(np.nanargmax(_vp_w) if _is_max else np.nanargmin(_vp_w))
                result.append((float(_tp_w[_idx]), float(_vp_w[_idx])))
            except Exception:
                result.append(None)
        return result

    return [None] * len(step_waves)


def _detect_xy_point_for_trend(x_sw, y_sw, dtype, dkey, ss):
    """
    XYグラフ検出点を1サイクル分検出して (x, y) | None を返す（傾向解析用）。
    x_sw, y_sw: 1サイクルの X/Y 波形配列
    dtype: "傾き変化点"|"閾値超え検出"|"Y最大値点"|"Y最小値点"
    """
    if dtype == "傾き変化点":
        _sm   = int(ss.get(f"{dkey}_smooth", 5))
        _nl   = int(ss.get(f"{dkey}_nleft",  3))
        _nr   = int(ss.get(f"{dkey}_nright", 3))
        _th   = float(ss.get(f"{dkey}_thresh", 0.0))
        _nth  = int(ss.get(f"{dkey}_nth", 0))
        _dinc = bool(ss.get(f"{dkey}_dir_inc", True))
        _ddec = bool(ss.get(f"{dkey}_dir_dec", True))
        _use_r = bool(ss.get(f"{dkey}_use_range", False))
        _rs    = float(ss.get(f"{dkey}_range_s", 0.0))  if _use_r else None
        _re    = float(ss.get(f"{dkey}_range_e", 1.0))  if _use_r else None
        _use_v = bool(ss.get(f"{dkey}_use_vrange", False))
        _rvlo  = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_v else None
        _rvhi  = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_v else None
        if _th <= 0.0:
            return None
        try:
            _xs  = _detect_inflections(x_sw, y_sw, smooth_w=_sm, n_left=_nl, n_right=_nr,
                                       threshold=_th, range_s=_rs, range_e=_re,
                                       detect_increase=_dinc, detect_decrease=_ddec)
            _pts = [(float(xi), float(np.interp(xi, x_sw, y_sw))) for xi in _xs]
            if _rvlo is not None and _rvhi is not None:
                _pts = [(xi, yi) for xi, yi in _pts if _rvlo <= yi <= _rvhi]
            _sel = _select_nth_pts(_pts, _nth)
            return _sel[0] if _sel else None
        except Exception:
            return None

    elif dtype == "閾値超え検出":
        _tv      = float(ss.get(f"{dkey}_tv", 1.0))
        _tdir_r  = str(ss.get(f"{dkey}_tdir", "上昇 ↑"))
        _tdir    = "rise" if "上昇" in _tdir_r else "fall" if "下降" in _tdir_r else "both"
        _nth     = int(ss.get(f"{dkey}_nth", 1))
        _use_r   = bool(ss.get(f"{dkey}_use_range", False))
        _rs      = float(ss.get(f"{dkey}_range_s", 0.0)) if _use_r else None
        _re      = float(ss.get(f"{dkey}_range_e", 1.0)) if _use_r else None
        _use_v   = bool(ss.get(f"{dkey}_use_vrange", False))
        _rvlo    = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_v else None
        _rvhi    = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_v else None
        try:
            _crs = _detect_threshold_crossings(x_sw, y_sw, _tv, direction=_tdir,
                                               range_s=_rs, range_e=_re)
            if _rvlo is not None and _rvhi is not None:
                _crs = [(xc, yc) for xc, yc in _crs if _rvlo <= yc <= _rvhi]
            _sel = _select_nth_pts(_crs, _nth)
            return _sel[0] if _sel else None
        except Exception:
            return None

    elif dtype in ("Y最大値点", "Y最小値点"):
        _is_max = (dtype == "Y最大値点")
        _use_xr = bool(ss.get(f"{dkey}_use_range",  False))
        _xrs    = float(ss.get(f"{dkey}_range_s",  0.0)) if _use_xr else None
        _xre    = float(ss.get(f"{dkey}_range_e",  1.0)) if _use_xr else None
        _use_yr = bool(ss.get(f"{dkey}_use_vrange", False))
        _yrlo   = float(ss.get(f"{dkey}_vrange_lo", 0.0)) if _use_yr else None
        _yrhi   = float(ss.get(f"{dkey}_vrange_hi", 1.0)) if _use_yr else None
        try:
            _xw, _yw = x_sw, y_sw
            if _xrs is not None and _xre is not None:
                _mx = (x_sw >= _xrs) & (x_sw <= _xre)
                if _mx.sum() > 0:
                    _xw = x_sw[_mx]; _yw = y_sw[_mx]
            if _yrlo is not None and _yrhi is not None:
                _my = (_yw >= _yrlo) & (_yw <= _yrhi)
                if _my.sum() > 0:
                    _xw = _xw[_my]; _yw = _yw[_my]
            if len(_yw) == 0:
                return None
            _idx = int(np.nanargmax(_yw) if _is_max else np.nanargmin(_yw))
            return (float(_xw[_idx]), float(_yw[_idx]))
        except Exception:
            return None

    return None


def _compute_wi_det_stats_for_csv(df, var_list, ss, trigger_col="", edge="RISE"):
    """
    CSV 1 ファイル分の波形検査検出点を解析し、各検出点の統計 (平均・σ) を返す。
    df          : DataFrame (1 CSV)
    var_list    : アナログ変数名リスト
    ss          : st.session_state
    trigger_col : グローバルフォールバックトリガー（省略可）
    edge        : グローバルフォールバックエッジ（省略可）
    Returns     : dict[det_key -> {label, color, t_mean, t_std, v_mean, v_std, n}]
    """
    _POINT_TYPES = ["傾き変化点", "閾値超え検出", "最大値点", "最小値点"]
    _DET_COLORS  = ["darkorange", "deeppink", "limegreen", "dodgerblue",
                    "gold", "orchid", "coral", "steelblue"]
    stats = {}
    for var in var_list:
        if var not in df.columns:
            continue
        _vkey      = f"wvol___global_{var}"
        _tdet_list = ss.get(f"{_vkey}_t_det_list", [])
        _trend_dets = [
            (_di, _det["id"], _det["type"], f"{_vkey}_{_det['id']}")
            for _di, _det in enumerate(_tdet_list)
            if _det.get("type", "") in _POINT_TYPES
            and bool(ss.get(f"{_vkey}_{_det['id']}_trend_on", False))
        ]
        if not _trend_dets:
            continue
        _wpre  = int(ss.get(f"{_vkey}_wpre",   50))
        _wpost = int(ss.get(f"{_vkey}_wpost", 300))
        # 変数ごとのトリガー/エッジ（未設定ならフォールバック）
        _var_trig = ss.get(f"{_vkey}_trigger", trigger_col) or trigger_col
        _var_edge = ss.get(f"{_vkey}_edge", edge)
        if not _var_trig or _var_trig not in df.columns:
            continue
        try:
            _waveforms = cached_waveforms(df, _var_trig, _var_edge, (var,))
        except Exception:
            continue
        step_waves = []
        for _cyc_df in _waveforms:
            _t_all = _cyc_df["time_offset_ms"].values.astype(float)
            if var not in _cyc_df.columns:
                continue
            _v_all = _cyc_df[var].values.astype(float)
            _mask  = (_t_all >= -_wpre) & (_t_all <= _wpost)
            step_waves.append((_t_all[_mask], _v_all[_mask]))
        if not step_waves:
            continue
        for (_di, _did, _dtype, _dkey) in _trend_dets:
            _color  = _DET_COLORS[_di % len(_DET_COLORS)]
            _pts    = _detect_point_for_trend(step_waves, _dtype, _dkey, ss)
            _t_vals = [p[0] for p in _pts if p is not None]
            _v_vals = [p[1] for p in _pts if p is not None]
            _n = len(_t_vals)
            if _n == 0:
                continue
            _type_lbl = {"傾き変化点": "傾き変化点", "閾値超え検出": "閾値超え検出",
                         "最大値点": "最大値点", "最小値点": "最小値点"}.get(_dtype, _dtype)
            _det_name_lbl = ss.get(f"{_dkey}_name", "").strip()
            stats[f"{_vkey}_{_did}"] = {
                "label":   f"{var} #{_di+1} {_det_name_lbl or _type_lbl}",
                "color":   _color,
                "t_mean":  float(np.mean(_t_vals)),
                "t_std":   float(np.std(_t_vals)) if _n > 1 else 0.0,
                "t_range": float(np.max(_t_vals) - np.min(_t_vals)) if _n > 1 else 0.0,
                "v_mean":  float(np.mean(_v_vals)),
                "v_std":   float(np.std(_v_vals)) if _n > 1 else 0.0,
                "v_range": float(np.max(_v_vals) - np.min(_v_vals)) if _n > 1 else 0.0,
                "n":       _n,
            }

    # ── XY グラフ検出点 ───────────────────────────────────────────
    _XY_POINT_TYPES = ["傾き変化点", "閾値超え検出", "Y最大値点", "Y最小値点"]
    for var in var_list:
        if var not in df.columns:
            continue
        _vkey   = f"wvol___global_{var}"
        _xvar   = ss.get(f"{_vkey}_xy_xvar", "")
        if not _xvar or _xvar not in df.columns:
            continue
        _xydet_list = ss.get(f"{_vkey}_xy_det_list", [])
        _xy_trend_dets = [
            (_di, _det["id"], _det["type"], f"{_vkey}_{_det['id']}")
            for _di, _det in enumerate(_xydet_list)
            if _det.get("type", "") in _XY_POINT_TYPES
            and bool(ss.get(f"{_vkey}_{_det['id']}_trend_on", False))
        ]
        if not _xy_trend_dets:
            continue
        _wpre  = int(ss.get(f"{_vkey}_wpre",   50))
        _wpost = int(ss.get(f"{_vkey}_wpost", 300))
        _var_trig_xy = ss.get(f"{_vkey}_trigger", trigger_col) or trigger_col
        _var_edge_xy = ss.get(f"{_vkey}_edge", edge)
        if not _var_trig_xy or _var_trig_xy not in df.columns:
            continue
        try:
            _wf_y = cached_waveforms(df, _var_trig_xy, _var_edge_xy, (var,))
            _wf_x = cached_waveforms(df, _var_trig_xy, _var_edge_xy, (_xvar,))
        except Exception:
            continue
        step_waves_xy = []
        for _cy, _cx in zip(_wf_y, _wf_x):
            _t_all = _cy["time_offset_ms"].values.astype(float)
            if var not in _cy.columns or _xvar not in _cx.columns:
                continue
            _y_all = _cy[var].values.astype(float)
            _x_all = _cx[_xvar].values.astype(float)
            _mask  = (_t_all >= -_wpre) & (_t_all <= _wpost)
            step_waves_xy.append((_x_all[_mask], _y_all[_mask]))
        if not step_waves_xy:
            continue
        for (_di, _did, _dtype, _dkey) in _xy_trend_dets:
            _color = _DET_COLORS[_di % len(_DET_COLORS)]
            _x_vals, _y_vals = [], []
            for (x_sw, y_sw) in step_waves_xy:
                _pt = _detect_xy_point_for_trend(x_sw, y_sw, _dtype, _dkey, ss)
                if _pt is not None:
                    _x_vals.append(_pt[0])
                    _y_vals.append(_pt[1])
            _n = len(_x_vals)
            if _n == 0:
                continue
            _type_lbl = _dtype
            _xy_det_name_lbl = ss.get(f"{_dkey}_name", "").strip()
            stats[f"{_vkey}_{_did}_xy"] = {
                "label":   f"{_xvar}→{var} #{_di+1} {_xy_det_name_lbl or _type_lbl} [XY]",
                "color":   _color,
                "t_mean":  float(np.mean(_x_vals)),
                "t_std":   float(np.std(_x_vals)) if _n > 1 else 0.0,
                "t_range": float(np.max(_x_vals) - np.min(_x_vals)) if _n > 1 else 0.0,
                "v_mean":  float(np.mean(_y_vals)),
                "v_std":   float(np.std(_y_vals)) if _n > 1 else 0.0,
                "v_range": float(np.max(_y_vals) - np.min(_y_vals)) if _n > 1 else 0.0,
                "n":       _n,
                "is_xy":   True,
                "x_label": _xvar,
                "y_label": var,
            }

    # ── 数式検出点（他の検出点を参照する計算式）──────────────────────
    for var in var_list:
        if var not in df.columns:
            continue
        _vkey = f"wvol___global_{var}"
        _tdet_list = ss.get(f"{_vkey}_t_det_list", [])
        # 数式型で傾向解析 ON のものを収集
        _fm_trend_dets = [
            (_di, _det["id"], f"{_vkey}_{_det['id']}")
            for _di, _det in enumerate(_tdet_list)
            if _det.get("type", "") == "数式"
            and bool(ss.get(f"{_vkey}_{_det['id']}_trend_on", False))
        ]
        if not _fm_trend_dets:
            continue
        _wpre  = int(ss.get(f"{_vkey}_wpre",   50))
        _wpost = int(ss.get(f"{_vkey}_wpost", 300))
        _var_trig_fm = ss.get(f"{_vkey}_trigger", trigger_col) or trigger_col
        _var_edge_fm = ss.get(f"{_vkey}_edge", edge)
        if not _var_trig_fm or _var_trig_fm not in df.columns:
            continue
        try:
            _wf_fm = cached_waveforms(df, _var_trig_fm, _var_edge_fm, (var,))
        except Exception:
            continue
        _sw_fm = []
        for _cyc_df in _wf_fm:
            _t_all = _cyc_df["time_offset_ms"].values.astype(float)
            if var not in _cyc_df.columns:
                continue
            _v_all = _cyc_df[var].values.astype(float)
            _mask  = (_t_all >= -_wpre) & (_t_all <= _wpost)
            _sw_fm.append((_t_all[_mask], _v_all[_mask]))
        if not _sw_fm:
            continue
        # 数式が参照する全非数式検出点をサイクルごとに計算（trend_on 不問）
        _ref_fm: dict = {}  # {di: [(t,v)|None, ...]}
        for _rdi, _rdet in enumerate(_tdet_list):
            _rtype = _rdet.get("type", "")
            _rdkey = f"{_vkey}_{_rdet['id']}"
            if _rtype in _POINT_TYPES:
                _ref_fm[_rdi] = _detect_point_for_trend(_sw_fm, _rtype, _rdkey, ss)
        # 各数式検出点を評価
        for (_fdi, _fdid, _fdkey) in _fm_trend_dets:
            _color    = _DET_COLORS[_fdi % len(_DET_COLORS)]
            _fm_expr  = str(ss.get(f"{_fdkey}_expr", ""))
            _fm_name_lbl = ss.get(f"{_fdkey}_name", "").strip()
            _fm_label = f"{var} #{_fdi + 1} {_fm_name_lbl or '数式'}"
            if not _fm_expr.strip():
                stats.setdefault("__formula_warns__", []).append(
                    f"⚠️ {_fm_label}: 数式が未入力です"
                )
                continue
            _fm_py    = _translate_formula(_fm_expr)
            _fm_vals: list = []
            for _ci in range(len(_sw_fm)):
                _vd: dict = {}
                for _rdi, _rpts in _ref_fm.items():
                    if _ci < len(_rpts) and _rpts[_ci] is not None:
                        _rc, _rv = _rpts[_ci]
                        _vd[f"p{_rdi + 1}t"] = _rc
                        _vd[f"p{_rdi + 1}v"] = _rv
                _res = _safe_eval_expr(_fm_py, _vd)
                if _res is not None:
                    _fm_vals.append(float(_res))
            _n = len(_fm_vals)
            if _n == 0:
                if not _ref_fm:
                    _why = "参照先の検出点がありません（傾き変化点・閾値超え検出等を先に追加してください）"
                elif all(all(p is None for p in pts) for pts in _ref_fm.values()):
                    _why = (
                        f"参照先の検出点が全サイクルで未検出です"
                        f"（傾き変化点なら閾値 > 0 が必要です）"
                    )
                else:
                    _why = f"数式 `{_fm_expr}` の評価が全サイクルで None でした（変数名 #N.t / #N.v を確認してください）"
                stats.setdefault("__formula_warns__", []).append(
                    f"⚠️ {_fm_label} [{_fm_expr}]: {_why}"
                )
                continue
            _short_expr = _fm_expr if len(_fm_expr) <= 20 else _fm_expr[:17] + "..."
            stats[f"{_vkey}_{_fdid}_formula"] = {
                "label":      f"{var} #{_fdi + 1} {_fm_name_lbl or ('数式 [' + _short_expr + ']')}",
                "color":      _color,
                "t_mean":     float(np.mean(_fm_vals)),
                "t_std":      float(np.std(_fm_vals)) if _n > 1 else 0.0,
                "t_range":    float(np.max(_fm_vals) - np.min(_fm_vals)) if _n > 1 else 0.0,
                "v_mean":     0.0,
                "v_std":      0.0,
                "v_range":    0.0,
                "n":          _n,
                "is_formula": True,
                "expr":       _fm_expr,
            }

    return stats


import ast as _ast_mod
import re as _re_mod


def _safe_eval_expr(expr_str: str, var_dict: dict):
    """
    AST-based safe expression evaluator for 数式 detection type.
    expr_str: Python式文字列（#N.t/#N.v → pNt/pNv 変換済み）
    var_dict: {"p1t": 12.3, "p1v": 5.6, ...}
    Returns: float | None（エラー・未解決変数時）
    """
    allowed_nodes = (
        _ast_mod.Expression, _ast_mod.BoolOp, _ast_mod.And, _ast_mod.Or,
        _ast_mod.Compare, _ast_mod.Lt, _ast_mod.Gt, _ast_mod.LtE, _ast_mod.GtE,
        _ast_mod.Eq, _ast_mod.NotEq,
        _ast_mod.BinOp, _ast_mod.Add, _ast_mod.Sub, _ast_mod.Mult, _ast_mod.Div,
        _ast_mod.UnaryOp, _ast_mod.USub, _ast_mod.UAdd,
        _ast_mod.Constant, _ast_mod.Name, _ast_mod.Load,
    )
    try:
        tree = _ast_mod.parse(expr_str, mode="eval")
        for node in _ast_mod.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
        result = eval(compile(tree, "<expr>", "eval"),
                      {"__builtins__": {}}, var_dict)
        return float(result)
    except Exception:
        return None


def _translate_formula(expr_str: str) -> str:
    """#N.t → pNt、#N.v → pNv、AND/OR → and/or に変換。"""
    s = expr_str.replace("AND", " and ").replace("OR", " or ")
    s = _re_mod.sub(r"#(\d+)\.t", r"p\1t", s)
    s = _re_mod.sub(r"#(\d+)\.v", r"p\1v", s)
    return s


# ═══════════════════════════════════════════════════════════════
# 傾き変化点検出 — 3段階アルゴリズム可視化
# ═══════════════════════════════════════════════════════════════

def _render_inflection_debug_t(t_arr, v_arr, smooth_w: int,
                                n_left: int, n_right: int,
                                threshold: float,
                                range_s: float, range_e: float,
                                y_label: str, chart_key: str):
    """
    時間軸波形の傾き変化点検出 3段階プレビュー（左右傾き差方式）
      ① 平滑化（移動平均）
      ② 左傾き L（n_left 離れた点）と右傾き R（n_right 離れた点）
      ③ |R−L| + 閾値 + 検出範囲 + 検出マーカー
    """
    need = n_left + n_right + 1
    if len(t_arr) < need:
        st.caption(f"データ不足でプレビュー不可（必要点数 {need} に対し {len(t_arr)} 点）")
        return

    w  = max(1, min(smooth_w, len(v_arr) // 2))
    vs = np.convolve(v_arr, np.ones(w) / w, mode="same") if w > 1 else v_arr.copy()
    t_mid, L, R, D = _slope_diff_t(t_arr, vs, n_left, n_right)
    aD = np.abs(D)

    use_range = (range_s is not None and range_e is not None and range_s < range_e)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "① 平滑化（移動平均）— 元データとの比較",
            f"② 左傾き L（{n_left}サンプル前）・右傾き R（{n_right}サンプル後）",
            "③ |傾き差 R−L| — 閾値・検出範囲・検出点",
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.32, 0.33],
    )

    # ① 元データ vs 平滑化
    fig.add_trace(go.Scatter(
        x=t_arr, y=v_arr, mode="lines",
        line=dict(color="rgba(150,150,150,0.45)", width=1),
        name="元データ",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=t_arr, y=vs, mode="lines",
        line=dict(color="royalblue", width=2),
        name=f"平滑化（幅 {w} サンプル）",
    ), row=1, col=1)
    if use_range:
        for row_ in (1, 2, 3):
            fig.add_vrect(x0=range_s, x1=range_e,
                          fillcolor="rgba(100,200,255,0.07)",
                          line_width=1, line_color="steelblue",
                          row=row_, col=1)

    # ② 左傾き L・右傾き R
    fig.add_trace(go.Scatter(
        x=t_mid, y=L, mode="lines",
        line=dict(color="steelblue", width=1.5, dash="dot"),
        name=f"左傾き L = (V[i]−V[i−{n_left}]) / Δt",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=t_mid, y=R, mode="lines",
        line=dict(color="darkorange", width=1.5),
        name=f"右傾き R = (V[i+{n_right}]−V[i]) / Δt",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)

    # ③ |傾き差| + 閾値 + 検出マーカー
    fig.add_trace(go.Scatter(
        x=t_mid, y=aD, mode="lines",
        line=dict(color="mediumpurple", width=1.5),
        name="|R − L|",
    ), row=3, col=1)
    if threshold > 0:
        fig.add_hline(
            y=threshold, line_color="red", line_dash="dash",
            row=3, col=1,
            annotation_text=f"閾値 {threshold:.4f}",
            annotation_position="bottom right",
        )
        hit = aD > threshold
        if use_range:
            hit = hit & (t_mid >= range_s) & (t_mid <= range_e)
        if hit.any():
            fig.add_trace(go.Scatter(
                x=t_mid[hit], y=aD[hit], mode="markers",
                marker=dict(symbol="diamond", color="darkorange",
                            size=9, line=dict(color="white", width=1)),
                name=f"検出点 ({int(hit.sum())} 点)",
            ), row=3, col=1)

    fig.update_layout(
        height=490,
        margin=dict(t=55, b=30, l=65, r=20),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0, font_size=11),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=y_label,   row=1, col=1)
    fig.update_yaxes(title_text="傾き",     row=2, col=1)
    fig.update_yaxes(title_text="|R − L|", row=3, col=1)
    fig.update_xaxes(title_text="ステップ開始からの時間 [ms]", row=3, col=1)
    st.plotly_chart(fig, width="stretch", key=chart_key)


def _render_inflection_debug_xy(x_arr, y_arr, smooth_w: int,
                                 n_left: int, n_right: int,
                                 threshold: float,
                                 range_s: float, range_e: float,
                                 x_label: str, y_label: str, chart_key: str):
    """
    XY波形の傾き変化点検出 3段階プレビュー（左右傾き差方式、X昇順ソート後）
      ① 平滑化（移動平均）
      ② 左傾き L（n_left 離れた点）と右傾き R（n_right 離れた点）
      ③ |R−L| + 閾値 + 検出範囲 + 検出マーカー
    """
    need = n_left + n_right + 1
    if len(x_arr) < need:
        st.caption(f"データ不足でプレビュー不可（必要点数 {need} に対し {len(x_arr)} 点）")
        return

    si = np.argsort(x_arr)
    xs, ys_raw = x_arr[si], y_arr[si]
    w  = max(1, min(smooth_w, len(ys_raw) // 2))
    ys = np.convolve(ys_raw, np.ones(w) / w, mode="same") if w > 1 else ys_raw.copy()
    x_mid, L, R, D = _slope_diff_xy(xs, ys, n_left, n_right)
    aD = np.abs(D)

    use_range = (range_s is not None and range_e is not None and range_s < range_e)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[
            "① 平滑化（移動平均）— 元データとの比較",
            f"② 左傾き L（{n_left}サンプル前）・右傾き R（{n_right}サンプル後）",
            "③ |傾き差 R−L| — 閾値・検出範囲・検出点",
        ],
        vertical_spacing=0.10,
        row_heights=[0.35, 0.32, 0.33],
    )

    # ① 元データ vs 平滑化
    fig.add_trace(go.Scatter(
        x=xs, y=ys_raw, mode="lines",
        line=dict(color="rgba(150,150,150,0.45)", width=1),
        name="元データ（X昇順）",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines",
        line=dict(color="royalblue", width=2),
        name=f"平滑化（幅 {w} サンプル）",
    ), row=1, col=1)
    if use_range:
        for row_ in (1, 2, 3):
            fig.add_vrect(x0=range_s, x1=range_e,
                          fillcolor="rgba(100,200,255,0.07)",
                          line_width=1, line_color="steelblue",
                          row=row_, col=1)

    # ② 左傾き L・右傾き R
    fig.add_trace(go.Scatter(
        x=x_mid, y=L, mode="lines",
        line=dict(color="steelblue", width=1.5, dash="dot"),
        name=f"左傾き L = (Y[i]−Y[i−{n_left}]) / ΔX",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_mid, y=R, mode="lines",
        line=dict(color="darkorange", width=1.5),
        name=f"右傾き R = (Y[i+{n_right}]−Y[i]) / ΔX",
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="gray", line_dash="dot", row=2, col=1)

    # ③ |傾き差| + 閾値 + 検出マーカー
    fig.add_trace(go.Scatter(
        x=x_mid, y=aD, mode="lines",
        line=dict(color="mediumpurple", width=1.5),
        name="|R − L|",
    ), row=3, col=1)
    if threshold > 0:
        fig.add_hline(
            y=threshold, line_color="red", line_dash="dash",
            row=3, col=1,
            annotation_text=f"閾値 {threshold:.4f}",
            annotation_position="bottom right",
        )
        hit = aD > threshold
        if use_range:
            hit = hit & (x_mid >= range_s) & (x_mid <= range_e)
        if hit.any():
            fig.add_trace(go.Scatter(
                x=x_mid[hit], y=aD[hit], mode="markers",
                marker=dict(symbol="diamond", color="darkorange",
                            size=9, line=dict(color="white", width=1)),
                name=f"検出点 ({int(hit.sum())} 点)",
            ), row=3, col=1)

    fig.update_layout(
        height=490,
        margin=dict(t=55, b=30, l=65, r=20),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0, font_size=11),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=y_label,   row=1, col=1)
    fig.update_yaxes(title_text="傾き",     row=2, col=1)
    fig.update_yaxes(title_text="|R − L|", row=3, col=1)
    fig.update_xaxes(title_text=x_label,   row=3, col=1)
    st.plotly_chart(fig, width="stretch", key=chart_key)

def _render_waveform_overlay(df: pd.DataFrame, trigger_col: str, edge: str,
                              step_stat: dict, step: dict, pname: str,
                              result_df: pd.DataFrame,
                              _sa_vars: list = None,
                              _ref_df: pd.DataFrame = None,
                              _compare_entries: list = None):
    """アナログ波形重ね合わせ表示（ステップ開始基準 or スタンドアロン）

    _sa_vars が指定された場合はスタンドアロンモード:
      - waveform_vars = _sa_vars
      - _vkey = wvol_{pname}_{var}  (ステップ名なし)
      - start_offsets = ゼロ (トリガー開始基準)
    """
    _standalone = _sa_vars is not None

    if _standalone:
        name  = ""          # bl_key 等で参照されるためデフォルト設定
        mode  = "single"
        waveform_vars = [v for v in _sa_vars if v in df.columns]
    else:
        mode  = step.get("mode", "single")
        name  = step_stat["name"]
        waveform_vars = [v for v in step.get("waveform_vars", []) if v in df.columns]

    if not waveform_vars:
        if _standalone:
            st.info("変数を選択してください")
        else:
            st.info("波形監視変数が未設定です。ステップ設定の「📈 波形監視変数」でアナログ変数を選択してください。")
        return

    # start_offsets: スタンドアロン=ゼロ、通常=ステップ開始時刻
    if _standalone:
        try:
            _n_cycles_sa = len(cached_detect_cycles(df, trigger_col, edge))
        except Exception:
            _n_cycles_sa = 1
        start_offsets = np.zeros(_n_cycles_sa)
    else:
        if mode == "single":
            start_col = f"{name}_遅れ[ms]"
        else:
            start_col = f"{name}_start[ms]"
        if start_col not in result_df.columns:
            st.warning(f"ステップ開始時刻列 `{start_col}` が見つかりません（解析を実行してください）")
            return
        start_offsets = result_df[start_col].values  # ms, per cycle

    # 非スタンドアロン: 全変数まとめて波形取得
    if not _standalone:
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, tuple(waveform_vars))
        except Exception as e:
            st.error(f"波形データ取得エラー: {e}")
            return
        n_cyc = min(len(waveforms), len(start_offsets))
        if n_cyc == 0:
            st.warning("波形データがありません")
            return

    # 基準登録データ取得
    _baseline = st.session_state.get(pk(pname, "baseline"), {})
    _wv_bl    = st.session_state.get(pk(pname, "wv_baseline"), {})  # 波形専用基準

    for var in waveform_vars:
        st.markdown(f"---\n#### 📈 {var}")
        # スタンドアロン: wvol_{pname}_{var} / 通常: wvol_{pname}_{step}_{var}
        _vkey = f"wvol_{pname}_{var}" if _standalone else f"wvol_{pname}_{name}_{var}"

        # スタンドアロン: 変数ごとにトリガーを持つ（未設定は trigger_col にフォールバック）
        if _standalone:
            _var_trig = st.session_state.get(
                f"{_vkey}_trigger",
                trigger_col,
            )
            _var_edge = st.session_state.get(
                f"{_vkey}_edge",
                edge,
            )
            try:
                _var_n_cyc = len(cached_detect_cycles(df, _var_trig, _var_edge))
            except Exception:
                _var_n_cyc = 1
            _var_start = np.zeros(_var_n_cyc)
            try:
                _var_waves = cached_waveforms(df, _var_trig, _var_edge, (var,))
            except Exception:
                _var_waves = []
            _var_n_cyc = min(len(_var_waves), _var_n_cyc)

        _tab_time, _tab_xy = st.tabs(["⏱ 時間軸", "📊 XY グラフ"])

        with _tab_time:
            # ── コントロールパネル ─────────────────────────────────
            _wwin_c1, _wwin_c2 = st.columns([2, 5])
            with _wwin_c1:
                st.markdown("**表示ウィンドウ**")
                win_pre  = st.number_input("開始前 [ms]", min_value=0, max_value=500,
                                            value=50, step=10, key=f"{_vkey}_wpre")
                win_post = st.number_input("終了後 [ms]", min_value=10, max_value=2000,
                                            value=300, step=10, key=f"{_vkey}_wpost")

            # ── 波形データ抽出（ステップ開始基準に変換）─────────────
            step_waves = []   # list of (t_step_rel, v_arr)
            if _standalone:
                for i in range(_var_n_cyc):
                    cyc = _var_waves[i]
                    if var not in cyc.columns:
                        continue
                    t_abs  = cyc["time_offset_ms"].values
                    t_step = t_abs  # start_offset = 0 (trigger-relative)
                    v_arr  = cyc[var].values.astype(np.float64)
                    mask = (t_step >= -win_pre) & (t_step <= win_post)
                    if mask.sum() < 2:
                        continue
                    step_waves.append((t_step[mask], v_arr[mask]))
            else:
                for i in range(n_cyc):
                    cyc        = waveforms[i]
                    step_off   = float(start_offsets[i]) if not np.isnan(start_offsets[i]) else None
                    if step_off is None or var not in cyc.columns:
                        continue
                    t_abs  = cyc["time_offset_ms"].values          # cycle-relative [ms]
                    t_step = t_abs - step_off                       # step-relative [ms]
                    v_arr  = cyc[var].values.astype(np.float64)
                    mask = (t_step >= -win_pre) & (t_step <= win_post)
                    if mask.sum() < 2:
                        continue
                    step_waves.append((t_step[mask], v_arr[mask]))

            # ダイアログ用にキャッシュ（スタンドアロンのみ）
            if _standalone and step_waves:
                st.session_state[f"{_vkey}_step_waves_cache"] = [
                    (t.tolist(), v.tolist()) for t, v in step_waves
                ]

            if not step_waves:
                st.warning(f"{var}: 有効な波形データがありません")
                continue

            # ── 基準エンベロープ計算 ─────────────────────────────────
            t_common = np.linspace(-win_pre, win_post, 500)
            mat = np.full((len(step_waves), len(t_common)), np.nan)
            for j, (t_sw, v_sw) in enumerate(step_waves):
                idx = np.searchsorted(t_sw, t_common).clip(0, len(t_sw) - 1)
                in_range = (t_common >= t_sw[0]) & (t_common <= t_sw[-1])
                mat[j, in_range] = v_sw[idx[in_range]]

            mean_v = np.nanmean(mat, axis=0)
            std_v  = np.nanstd(mat, axis=0)

            # ── 基準CSV 参照波形の計算 ────────────────────────────────
            _ref_mean_v = None
            _ref_std_v  = None
            if _ref_df is not None and len(_ref_df) > 0:
                try:
                    _rw_all = cached_waveforms(_ref_df, trigger_col, edge,
                                               tuple(waveform_vars))
                    _ref_sw = []
                    for _ri in range(len(_rw_all)):
                        _rcyc = _rw_all[_ri]
                        if var not in _rcyc.columns:
                            continue
                        _rt = _rcyc["time_offset_ms"].values
                        _rv = _rcyc[var].values.astype(np.float64)
                        _rm = (_rt >= -win_pre) & (_rt <= win_post)
                        if _rm.sum() >= 2:
                            _ref_sw.append((_rt[_rm], _rv[_rm]))
                    if _ref_sw:
                        _ref_mat2 = np.full((len(_ref_sw), len(t_common)), np.nan)
                        for _rj, (_rts, _rvs) in enumerate(_ref_sw):
                            _ridx = np.searchsorted(_rts, t_common).clip(
                                0, len(_rts) - 1)
                            _rin  = (t_common >= _rts[0]) & (t_common <= _rts[-1])
                            _ref_mat2[_rj, _rin] = _rvs[_ridx[_rin]]
                        _ref_mean_v = np.nanmean(_ref_mat2, axis=0)
                        _ref_std_v  = np.nanstd(_ref_mat2,  axis=0)
                except Exception:
                    pass

            ng_flags = [False] * len(step_waves)

            # ── 検出点リスト ──────────────────────────────────────
            _tdet_list_key = f"{_vkey}_t_det_list"
            _tdet_cnt_key  = f"{_vkey}_t_det_cnt"
            if _tdet_list_key not in st.session_state:
                st.session_state[_tdet_list_key] = []
            if _tdet_cnt_key not in st.session_state:
                st.session_state[_tdet_cnt_key] = 0
            _tdet_list = st.session_state[_tdet_list_key]

            # 点取得系: (t,v) を記録してマーカー表示 + OK/NG判定 + 傾向解析に出す
            # 判定系: NG フラグのみ（サイクル単位の合否）
            _DET_TYPES_POINT  = ["傾き変化点", "閾値超え検出", "最大値点", "最小値点"]
            _DET_TYPES_JUDGE  = ["上下判定比較", "最大値判定", "最小値判定", "検出点比較", "数式"]
            _DET_TYPES_T      = _DET_TYPES_POINT + _DET_TYPES_JUDGE
            _DET_COLORS       = ["darkorange", "deeppink", "limegreen", "dodgerblue",
                                 "gold", "orchid", "coral", "steelblue"]

            # 結果収集用
            all_inf_markers = []   # [{"t":[], "v":[], "label":str, "color":str}]
            peak_ng_flags   = [False] * len(step_waves)
            cyc_max_vals    = [float(np.nanmax(v)) for _, v in step_waves]
            cyc_min_vals    = [float(np.nanmin(v)) for _, v in step_waves]
            # 数式タイプが参照するサイクルごとの検出点 {di: [(coord, val)|None, ...]}
            _det_pts_per_cycle = {}

            if _tdet_list:
                _h1, _h2, _h3, _h4, _h5 = st.columns([0.5, 3.5, 1.8, 0.8, 0.5])
                _h1.caption("種別")
                _h2.caption("名前 / タイプ")
                _h3.caption("")
                _h4.caption("有効")
            _t_del_idx = None
            for _di, _det in enumerate(_tdet_list):
                _did   = _det["id"]
                _dtype = _det["type"]
                _dkey  = f"{_vkey}_{_did}"
                _color = _DET_COLORS[_di % len(_DET_COLORS)]
                _icon  = {"傾き変化点": "📐", "閾値超え検出": "🎯",
                          "最大値点": "⬆️", "最小値点": "⬇️",
                          "上下判定比較": "📊", "最大値判定": "🔺", "最小値判定": "🔻",
                          "検出点比較": "🔲", "数式": "🧮"}.get(_dtype, "📏")

                _det_disp_name = st.session_state.get(f"{_dkey}_name", "").strip()
                _is_on = bool(st.session_state.get(f"{_dkey}_on", False))
                _ri1, _ri2, _ri3, _ri4, _ri5 = st.columns([0.5, 3.5, 1.8, 0.8, 0.5])
                with _ri1: st.markdown(f"{_icon} **{_di+1}**")
                with _ri2: st.markdown(_det_disp_name or f"*{_dtype}*")
                with _ri3: st.caption(_dtype if _det_disp_name else "")
                with _ri4: st.markdown("✅" if _is_on else "⬜")
                with _ri5:
                    if st.button("🗑", key=f"{_dkey}_del", help="削除"):
                        _t_del_idx = _di
                with st.popover(f"⚙️ #{_di+1} {_det_disp_name or _dtype}"):
                    st.text_input("名前（任意）", key=f"{_dkey}_name",
                                  placeholder=f"例: {_dtype}①",
                                  help="傾向解析ラベル・サマリーに表示されます")

                    if _dtype == "傾き変化点":
                        st.caption(
                            "① 移動平均でノイズ除去 → "
                            "② 左右 n サンプル離れた点で傾き L・R を計算 → "
                            "③ |R − L| > 閾値 かつ 検出範囲内 の点を検出"
                        )
                        _pa, _pb, _pc, _pd, _pe = st.columns([2, 2, 2, 3, 1])
                        with _pa:
                            st.markdown("**① 平滑化幅**")
                            st.number_input("サンプル数", min_value=1, max_value=50,
                                            value=5, step=1, key=f"{_dkey}_smooth")
                        with _pb:
                            st.markdown("**n_L（左）**")
                            st.number_input("左へ何サンプル", min_value=1, max_value=100,
                                            value=3, step=1, key=f"{_dkey}_nleft",
                                            help="L = (V[i]−V[i−n_L]) / Δt")
                        with _pc:
                            st.markdown("**n_R（右）**")
                            st.number_input("右へ何サンプル", min_value=1, max_value=100,
                                            value=3, step=1, key=f"{_dkey}_nright",
                                            help="R = (V[i+n_R]−V[i]) / Δt")
                        with _pd:
                            _sm = int(st.session_state.get(f"{_dkey}_smooth", 5))
                            _nl = int(st.session_state.get(f"{_dkey}_nleft",  3))
                            _nr = int(st.session_state.get(f"{_dkey}_nright", 3))
                            _ref = _slope_diff_max_ref_t(
                                step_waves[:min(5, len(step_waves))],
                                smooth_w=_sm, n_left=_nl, n_right=_nr)
                            _stp = max(0.0001, round(_ref / 20, 4))
                            st.markdown("**③ 閾値 |R − L|**")
                            st.number_input(f"（参考最大 ≈ {_ref:.4f}）",
                                            min_value=0.0, value=0.0,
                                            step=_stp, format="%.4f",
                                            key=f"{_dkey}_thresh")
                        with _pe:
                            st.markdown("**有効**")
                            st.checkbox("ON", key=f"{_dkey}_on")

                        # ── 方向指定 ─────────────────────────────
                        _dir_ca, _dir_cb = st.columns(2)
                        with _dir_ca:
                            st.checkbox("📈 増加方向を検出 (R > L)",
                                        value=True, key=f"{_dkey}_dir_inc")
                        with _dir_cb:
                            st.checkbox("📉 減少方向を検出 (L > R)",
                                        value=True, key=f"{_dkey}_dir_dec")

                        _rng_ca, _rng_cb, _rng_cc = st.columns([1, 2, 2])
                        with _rng_ca:
                            st.markdown("**🔍 t 検索範囲 [ms]**")
                            st.checkbox("指定する", key=f"{_dkey}_use_range")
                        _use_rng = bool(st.session_state.get(f"{_dkey}_use_range", False))
                        with _rng_cb:
                            st.number_input("開始", value=0.0, step=5.0,
                                            key=f"{_dkey}_range_s",
                                            disabled=not _use_rng)
                        with _rng_cc:
                            st.number_input("終了", value=200.0, step=5.0,
                                            key=f"{_dkey}_range_e",
                                            disabled=not _use_rng)
                        _rng_va, _rng_vb, _rng_vc = st.columns([1, 2, 2])
                        with _rng_va:
                            st.markdown("**🔍 v 検索範囲**")
                            st.checkbox("指定する", key=f"{_dkey}_use_vrange")
                        _use_vrng = bool(st.session_state.get(f"{_dkey}_use_vrange", False))
                        with _rng_vb:
                            st.number_input("v 下限", value=0.0, step=0.1,
                                            key=f"{_dkey}_vrange_lo",
                                            disabled=not _use_vrng)
                        with _rng_vc:
                            st.number_input("v 上限", value=1.0, step=0.1,
                                            key=f"{_dkey}_vrange_hi",
                                            disabled=not _use_vrng)

                        _sm_d  = int(st.session_state.get(f"{_dkey}_smooth", 5))
                        _nl_d  = int(st.session_state.get(f"{_dkey}_nleft",  3))
                        _nr_d  = int(st.session_state.get(f"{_dkey}_nright", 3))
                        _th_d  = float(st.session_state.get(f"{_dkey}_thresh", 0.0))
                        _rs_d  = float(st.session_state.get(f"{_dkey}_range_s", 0.0)) \
                                 if _use_rng else None
                        _re_d  = float(st.session_state.get(f"{_dkey}_range_e", 200.0)) \
                                 if _use_rng else None
                        _rvlo_d = float(st.session_state.get(f"{_dkey}_vrange_lo", 0.0)) \
                                  if _use_vrng else None
                        _rvhi_d = float(st.session_state.get(f"{_dkey}_vrange_hi", 1.0)) \
                                  if _use_vrng else None
                        _dinc  = bool(st.session_state.get(f"{_dkey}_dir_inc", True))
                        _ddec  = bool(st.session_state.get(f"{_dkey}_dir_dec", True))
                        st.markdown("---")
                        st.markdown("**📊 3段階プレビュー**（サンプル波形 #1）")
                        _prev_t0, _prev_v0 = step_waves[0]
                        _render_inflection_debug_t(
                            _prev_t0, _prev_v0,
                            smooth_w=_sm_d, n_left=_nl_d, n_right=_nr_d,
                            threshold=_th_d, range_s=_rs_d, range_e=_re_d,
                            y_label=var, chart_key=f"{_dkey}_preview",
                        )

                        # N番目の点
                        _nth_c1, _nth_c2 = st.columns([2, 4])
                        with _nth_c1:
                            st.markdown("**🎯 使用する点**")
                        with _nth_c2:
                            st.number_input(
                                "N番目（0=全て, 1=最初, -1=最後）",
                                value=0, step=1, key=f"{_dkey}_nth",
                                help="サイクルごとにN番目の検出点のみ使用。0=全点。")

                        # ── OK/NG 判定（基準±Δ）& 傾向解析 ──────────
                        st.markdown("**✅ OK/NG 判定**")
                        _pm_c1, _pm_c2, _pm_c3 = st.columns([1.5, 2, 2])
                        with _pm_c1:
                            st.checkbox("基準±Δで判定", key=f"{_dkey}_pm_on",
                                        help="現データの平均検出値を基準に±許容差を超えたサイクルをNG")
                        _pm_on_inf = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                        with _pm_c2:
                            st.number_input("t 許容差 [ms]（0=無効）", min_value=0.0,
                                            value=0.0, step=1.0, key=f"{_dkey}_pm_dt",
                                            disabled=not _pm_on_inf)
                        with _pm_c3:
                            st.number_input("v 許容差（0=無効）", min_value=0.0,
                                            value=0.0, step=0.01, key=f"{_dkey}_pm_dv",
                                            disabled=not _pm_on_inf)
                        st.checkbox("📈 傾向解析に出す", key=f"{_dkey}_trend_on",
                                    help="検出点の t・v 値をサイクルごとに 📈 傾向解析タブへ送ります")

                        # 全サイクルから検出点収集
                        _det_on  = bool(st.session_state.get(f"{_dkey}_on", False))
                        _det_thr = float(st.session_state.get(f"{_dkey}_thresh", 0.0))
                        _det_nth = int(st.session_state.get(f"{_dkey}_nth", 0))
                        if _det_on and _det_thr > 0:
                            _mkt, _mkv = [], []
                            _cyc_pts_inf = []
                            for (t_sw, v_sw) in step_waves:
                                _ts = _detect_inflections(
                                    t_sw, v_sw,
                                    smooth_w=_sm_d, n_left=_nl_d, n_right=_nr_d,
                                    threshold=_det_thr,
                                    range_s=_rs_d, range_e=_re_d,
                                    detect_increase=_dinc, detect_decrease=_ddec)
                                _pts = [(float(ti), float(np.interp(ti, t_sw, v_sw)))
                                        for ti in _ts]
                                # v 検索範囲フィルター
                                if _rvlo_d is not None and _rvhi_d is not None:
                                    _pts = [(ti, vi) for ti, vi in _pts
                                            if _rvlo_d <= vi <= _rvhi_d]
                                _sel = _select_nth_pts(_pts, _det_nth)
                                for ti, vi in _sel:
                                    _mkt.append(ti); _mkv.append(vi)
                                _cyc_pts_inf.append(_sel[0] if _sel else None)
                            _det_pts_per_cycle[_di] = _cyc_pts_inf
                            if _mkt:
                                all_inf_markers.append({
                                    "t": _mkt, "v": _mkv,
                                    "label": f"#{_di+1} {_det_disp_name or '傾き変化点'} ({len(_mkt)}点)",
                                    "color": _color,
                                })
                            # OK/NG 判定（基準±Δ）
                            _pm_on = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                            _pm_dt = float(st.session_state.get(f"{_dkey}_pm_dt", 0.0))
                            _pm_dv = float(st.session_state.get(f"{_dkey}_pm_dv", 0.0))
                            if _pm_on and (_pm_dt > 0 or _pm_dv > 0):
                                _valid_inf = [p for p in _cyc_pts_inf if p is not None]
                                if _valid_inf:
                                    _ref_t_m = float(np.mean([p[0] for p in _valid_inf]))
                                    _ref_v_m = float(np.mean([p[1] for p in _valid_inf]))
                                    for _ji, _pts_ji in enumerate(_cyc_pts_inf):
                                        if _pts_ji is None:
                                            peak_ng_flags[_ji] = True; continue
                                        _ti_c, _vi_c = _pts_ji
                                        if _pm_dt > 0 and abs(_ti_c - _ref_t_m) > _pm_dt:
                                            peak_ng_flags[_ji] = True
                                        if _pm_dv > 0 and abs(_vi_c - _ref_v_m) > _pm_dv:
                                            peak_ng_flags[_ji] = True
                            # 傾向解析に出す
                            if bool(st.session_state.get(f"{_dkey}_trend_on", False)):
                                if "wi_det_trend" not in st.session_state:
                                    st.session_state["wi_det_trend"] = {}
                                st.session_state["wi_det_trend"][f"{_vkey}_{_did}"] = {
                                    "label":  f"{var} #{_di+1} {_det_disp_name or '傾き変化点'}",
                                    "color":  _color,
                                    "t_vals": [p[0] if p else None for p in _cyc_pts_inf],
                                    "v_vals": [p[1] if p else None for p in _cyc_pts_inf],
                                }

                    elif _dtype == "上下判定比較":
                        # ── 上下判定比較 ─────────────────────────────
                        st.caption("上下限の曲線を定義し、各サイクルの波形が範囲内に収まるか判定します。")
                        _bnd_type = st.radio(
                            "境界タイプ",
                            ["絶対値キーポイント", "基準±Nσ",
                             "現データ エンベロープ±マージン", "参照サイクル±オフセット"],
                            horizontal=True, key=f"{_dkey}_btype",
                        )
                        _bnd_on = st.checkbox("有効", key=f"{_dkey}_on")

                        # ── キーポイントテーブル または パラメータ ──
                        if _bnd_type == "絶対値キーポイント":
                            st.caption("t [ms] と上限・下限の値を直接入力してください。行を追加して折れ線を定義できます。")
                            _kp_default = pd.DataFrame({
                                "t [ms]": [0.0, 100.0],
                                "上限": [float(np.nanmax(mean_v)) * 1.1,
                                         float(np.nanmax(mean_v)) * 1.1],
                                "下限": [float(np.nanmin(mean_v)) * 0.9,
                                         float(np.nanmin(mean_v)) * 0.9],
                            })
                            _kp_key = f"{_dkey}_kp_df"
                            if _kp_key not in st.session_state:
                                st.session_state[_kp_key] = _kp_default
                            _kp_edited = st.data_editor(
                                st.session_state[_kp_key],
                                num_rows="dynamic", use_container_width=True,
                                key=f"{_dkey}_kp_editor",
                            )
                            _bnd_hi_fn = lambda t_arr: np.interp(
                                t_arr,
                                _kp_edited["t [ms]"].values.astype(float),
                                _kp_edited["上限"].values.astype(float),
                            )
                            _bnd_lo_fn = lambda t_arr: np.interp(
                                t_arr,
                                _kp_edited["t [ms]"].values.astype(float),
                                _kp_edited["下限"].values.astype(float),
                            )

                        elif _bnd_type == "基準±Nσ":
                            _nsig_val = st.number_input(
                                "N（±Nσ）", min_value=0.1, max_value=10.0,
                                value=3.0, step=0.5, key=f"{_dkey}_nsig")
                            _bl_key_t = f"{pname}_{name}_{var}"
                            _wv_bl_t  = st.session_state.get(pk(pname, "wv_baseline"), {})
                            _bl_t     = _wv_bl_t.get(_bl_key_t, {})
                            if _bl_t:
                                _bl_m = np.interp(t_common,
                                                  np.array(_bl_t["t"]),
                                                  np.array(_bl_t["mean"]))
                                _bl_s = np.interp(t_common,
                                                  np.array(_bl_t["t"]),
                                                  np.array(_bl_t["std"]))
                            else:
                                _bl_m, _bl_s = mean_v, std_v
                                st.caption("⚠️ 基準未登録のため現データ平均±Nσを仮使用")
                            _nsig_val_f = float(st.session_state.get(f"{_dkey}_nsig", 3.0))
                            _bnd_hi_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _bl_m + _nsig_val_f * _bl_s)
                            _bnd_lo_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _bl_m - _nsig_val_f * _bl_s)

                        elif _bnd_type == "現データ エンベロープ±マージン":
                            _env_margin = st.number_input(
                                "マージン（上下に加算）", value=0.0, step=0.1,
                                key=f"{_dkey}_margin")
                            # 全サイクル最大最小エンベロープ
                            _env_mat = np.full((len(step_waves), len(t_common)), np.nan)
                            for _ej, (t_sw, v_sw) in enumerate(step_waves):
                                _env_mat[_ej] = np.interp(t_common, t_sw, v_sw,
                                                           left=np.nan, right=np.nan)
                            _env_hi_c = np.nanmax(_env_mat, axis=0)
                            _env_lo_c = np.nanmin(_env_mat, axis=0)
                            _env_mg   = float(st.session_state.get(f"{_dkey}_margin", 0.0))
                            _bnd_hi_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _env_hi_c + _env_mg)
                            _bnd_lo_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _env_lo_c - _env_mg)

                        else:  # 参照サイクル±オフセット
                            _ref_mode = st.radio(
                                "参照元", ["基準登録データ", "サイクル番号指定"],
                                horizontal=True, key=f"{_dkey}_refmode")
                            _ref_offset = st.number_input(
                                "±オフセット（上下に加算）", value=0.0, step=0.1,
                                key=f"{_dkey}_offset")
                            _ref_offset_f = float(st.session_state.get(
                                f"{_dkey}_offset", 0.0))
                            if _ref_mode == "基準登録データ":
                                _bl_key_t = f"{pname}_{name}_{var}"
                                _wv_bl_t  = st.session_state.get(
                                    pk(pname, "wv_baseline"), {})
                                _bl_t     = _wv_bl_t.get(_bl_key_t, {})
                                if _bl_t:
                                    _ref_wave = np.interp(t_common,
                                                          np.array(_bl_t["t"]),
                                                          np.array(_bl_t["mean"]))
                                else:
                                    _ref_wave = mean_v
                                    st.caption("⚠️ 基準未登録のため現データ平均を仮使用")
                            else:
                                _ref_cyc = st.number_input(
                                    "サイクル番号（0起算）", min_value=0,
                                    max_value=max(0, len(step_waves) - 1),
                                    value=0, step=1, key=f"{_dkey}_refcyc")
                                _rc = int(st.session_state.get(f"{_dkey}_refcyc", 0))
                                _rc = min(_rc, len(step_waves) - 1)
                                _ref_wave = np.interp(t_common,
                                                      step_waves[_rc][0],
                                                      step_waves[_rc][1],
                                                      left=np.nan, right=np.nan)
                            _bnd_hi_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _ref_wave + _ref_offset_f)
                            _bnd_lo_fn = lambda t_arr: np.interp(
                                t_arr, t_common, _ref_wave - _ref_offset_f)

                        # 境界プレビュー描画
                        if True:
                            _prev_hi = _bnd_hi_fn(t_common)
                            _prev_lo = _bnd_lo_fn(t_common)
                            _fig_bnd = go.Figure()
                            _fig_bnd.add_trace(go.Scatter(
                                x=np.concatenate([t_common, t_common[::-1]]),
                                y=np.concatenate([_prev_hi, _prev_lo[::-1]]),
                                fill="toself", fillcolor="rgba(0,180,80,0.12)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name="合格範囲",
                            ))
                            _fig_bnd.add_trace(go.Scatter(
                                x=t_common, y=_prev_hi, mode="lines",
                                line=dict(color="green", width=1.5, dash="dash"),
                                name="上限",
                            ))
                            _fig_bnd.add_trace(go.Scatter(
                                x=t_common, y=_prev_lo, mode="lines",
                                line=dict(color="green", width=1.5, dash="dash"),
                                name="下限", showlegend=False,
                            ))
                            _bnd_prev_t, _bnd_prev_v = step_waves[0]
                            _fig_bnd.add_trace(go.Scatter(
                                x=_bnd_prev_t, y=_bnd_prev_v, mode="lines",
                                line=dict(color="royalblue", width=1.5),
                                name="サンプル波形 #1",
                            ))
                            _fig_bnd.update_layout(
                                height=200, margin=dict(t=10, b=30, l=50, r=10),
                                xaxis_title="t [ms]", yaxis_title=var,
                            )
                            st.plotly_chart(_fig_bnd, width="stretch",
                                            key=f"{_dkey}_bnd_preview")

                        # NG 判定（各サイクル）
                        if _bnd_on:
                            _bnd_markers_hi = {"t": [], "v": [], "label": f"#{_di+1} 上限超過",
                                               "color": _color}
                            _bnd_markers_lo = {"t": [], "v": [], "label": f"#{_di+1} 下限超過",
                                               "color": _color}
                            for j, (t_sw, v_sw) in enumerate(step_waves):
                                t_insp, v_insp = t_sw, v_sw
                                if len(t_insp) == 0:
                                    continue
                                hi_j = _bnd_hi_fn(t_insp)
                                lo_j = _bnd_lo_fn(t_insp)
                                over_hi = v_insp > hi_j
                                over_lo = v_insp < lo_j
                                if np.any(over_hi) or np.any(over_lo):
                                    peak_ng_flags[j] = True
                                for ti, vi in zip(t_insp[over_hi], v_insp[over_hi]):
                                    _bnd_markers_hi["t"].append(float(ti))
                                    _bnd_markers_hi["v"].append(float(vi))
                                for ti, vi in zip(t_insp[over_lo], v_insp[over_lo]):
                                    _bnd_markers_lo["t"].append(float(ti))
                                    _bnd_markers_lo["v"].append(float(vi))
                            # グラフ用上下限ライン
                            all_inf_markers.append({
                                "t": [], "v": [],
                                "label": f"#{_di+1} 上下限ライン",
                                "color": _color,
                                "_bnd_hi": _bnd_hi_fn(t_common).tolist(),
                                "_bnd_lo": _bnd_lo_fn(t_common).tolist(),
                                "_t_common": t_common.tolist(),
                            })
                            if _bnd_markers_hi["t"]:
                                all_inf_markers.append(_bnd_markers_hi)
                            if _bnd_markers_lo["t"]:
                                all_inf_markers.append(_bnd_markers_lo)

                    elif _dtype in ("最大値判定", "最小値判定"):
                        _is_max = (_dtype == "最大値判定")
                        _vlabel = "最大値" if _is_max else "最小値"
                        st.caption(f"各サイクルの{_vlabel}を計算し、合格基準と照合します。")
                        _pk_ca, _pk_cb, _pk_cc = st.columns([1, 2, 2])
                        with _pk_ca:
                            st.markdown(f"**{_vlabel}**")
                            st.checkbox("有効", key=f"{_dkey}_on")
                        with _pk_cb:
                            st.number_input("合格 上限（超えたらNG）",
                                            value=0.0, step=0.1,
                                            key=f"{_dkey}_hi", help="0 = 判定なし")
                        with _pk_cc:
                            st.number_input("合格 下限（下回ったらNG）",
                                            value=0.0, step=0.1,
                                            key=f"{_dkey}_lo", help="0 = 判定なし")

                        _det_on = bool(st.session_state.get(f"{_dkey}_on", False))
                        _det_hi = float(st.session_state.get(f"{_dkey}_hi", 0.0))
                        _det_lo = float(st.session_state.get(f"{_dkey}_lo", 0.0))
                        if _det_on:
                            for j, (t_sw, v_sw) in enumerate(step_waves):
                                _vp = v_sw
                                _val = float(np.nanmax(_vp) if _is_max else np.nanmin(_vp))
                                if _det_hi != 0.0 and _val > _det_hi:
                                    peak_ng_flags[j] = True
                                if _det_lo != 0.0 and _val < _det_lo:
                                    peak_ng_flags[j] = True

                    elif _dtype == "閾値超え検出":
                        # ── 閾値超え検出 ─────────────────────────────
                        st.caption(
                            "波形値が閾値を超えた（または下回った）瞬間を検出します。"
                            "N番目の交差点のみを使うことも可能です。"
                        )
                        _thc1, _thc2, _thc3 = st.columns([3, 3, 1])
                        with _thc1:
                            st.markdown("**閾値**")
                            st.number_input("閾値", value=1.0, step=0.1,
                                            key=f"{_dkey}_tv")
                        with _thc2:
                            st.markdown("**方向**")
                            st.radio("方向", ["上昇 ↑", "下降 ↓", "両方"],
                                     horizontal=True, key=f"{_dkey}_tdir")
                        with _thc3:
                            st.markdown("**有効**")
                            st.checkbox("ON", key=f"{_dkey}_on")

                        _nth_c1, _nth_c2 = st.columns([2, 4])
                        with _nth_c1:
                            st.markdown("**🎯 使用する点**")
                        with _nth_c2:
                            st.number_input(
                                "N番目（0=全て, 1=最初, -1=最後）",
                                value=1, step=1, key=f"{_dkey}_nth",
                                help="サイクルごとにN番目の交差点のみ使用。0=全点。")

                        _rng_tc1, _rng_tc2, _rng_tc3 = st.columns([1, 2, 2])
                        with _rng_tc1:
                            st.markdown("**🔍 t 検索範囲 [ms]**")
                            st.checkbox("指定する", key=f"{_dkey}_use_range")
                        _use_rng_t = bool(st.session_state.get(
                            f"{_dkey}_use_range", False))
                        with _rng_tc2:
                            st.number_input("開始", value=0.0, step=5.0,
                                            key=f"{_dkey}_range_s",
                                            disabled=not _use_rng_t)
                        with _rng_tc3:
                            st.number_input("終了", value=200.0, step=5.0,
                                            key=f"{_dkey}_range_e",
                                            disabled=not _use_rng_t)
                        _rng_tvc1, _rng_tvc2, _rng_tvc3 = st.columns([1, 2, 2])
                        with _rng_tvc1:
                            st.markdown("**🔍 v 検索範囲**")
                            st.checkbox("指定する", key=f"{_dkey}_use_vrange")
                        _use_vrng_t = bool(st.session_state.get(f"{_dkey}_use_vrange", False))
                        with _rng_tvc2:
                            st.number_input("v 下限", value=0.0, step=0.1,
                                            key=f"{_dkey}_vrange_lo",
                                            disabled=not _use_vrng_t)
                        with _rng_tvc3:
                            st.number_input("v 上限", value=1.0, step=0.1,
                                            key=f"{_dkey}_vrange_hi",
                                            disabled=not _use_vrng_t)

                        # ── OK/NG 判定（基準±Δ）& 傾向解析 ──────────
                        st.markdown("**✅ OK/NG 判定**")
                        _thr_pm1, _thr_pm2, _thr_pm3 = st.columns([1.5, 2, 2])
                        with _thr_pm1:
                            st.checkbox("基準±Δで判定", key=f"{_dkey}_pm_on",
                                        help="現データの平均検出値を基準に±許容差を超えたサイクルをNG")
                        _pm_on_thr = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                        with _thr_pm2:
                            st.number_input("t 許容差 [ms]（0=無効）", min_value=0.0,
                                            value=0.0, step=1.0, key=f"{_dkey}_pm_dt",
                                            disabled=not _pm_on_thr)
                        with _thr_pm3:
                            st.number_input("v 許容差（0=無効）", min_value=0.0,
                                            value=0.0, step=0.01, key=f"{_dkey}_pm_dv",
                                            disabled=not _pm_on_thr)
                        st.checkbox("📈 傾向解析に出す", key=f"{_dkey}_trend_on",
                                    help="検出点の t・v 値をサイクルごとに 📈 傾向解析タブへ送ります")

                        _tv_val   = float(st.session_state.get(f"{_dkey}_tv", 1.0))
                        _tdir_raw = st.session_state.get(f"{_dkey}_tdir", "上昇 ↑")
                        _tdir_str = ("rise" if "上昇" in _tdir_raw
                                     else "fall" if "下降" in _tdir_raw else "both")
                        _t_nth    = int(st.session_state.get(f"{_dkey}_nth", 1))
                        _t_det_on = bool(st.session_state.get(f"{_dkey}_on", False))
                        _t_rs     = float(st.session_state.get(
                            f"{_dkey}_range_s", 0.0)) if _use_rng_t else None
                        _t_re     = float(st.session_state.get(
                            f"{_dkey}_range_e", 200.0)) if _use_rng_t else None
                        _t_rvlo   = float(st.session_state.get(
                            f"{_dkey}_vrange_lo", 0.0)) if _use_vrng_t else None
                        _t_rvhi   = float(st.session_state.get(
                            f"{_dkey}_vrange_hi", 1.0)) if _use_vrng_t else None

                        if _t_det_on:
                            _tmkt, _tmkv = [], []
                            _cyc_pts_thr = []
                            for (t_sw, v_sw) in step_waves:
                                _crs = _detect_threshold_crossings(
                                    t_sw, v_sw, _tv_val,
                                    direction=_tdir_str,
                                    range_s=_t_rs, range_e=_t_re)
                                # v 検索範囲フィルター
                                if _t_rvlo is not None and _t_rvhi is not None:
                                    _crs = [(tc, vc) for tc, vc in _crs
                                            if _t_rvlo <= vc <= _t_rvhi]
                                _sel = _select_nth_pts(_crs, _t_nth)
                                for tc, vc in _sel:
                                    _tmkt.append(tc); _tmkv.append(vc)
                                _cyc_pts_thr.append(_sel[0] if _sel else None)
                            _det_pts_per_cycle[_di] = _cyc_pts_thr
                            if _tmkt:
                                all_inf_markers.append({
                                    "t": _tmkt, "v": _tmkv,
                                    "label": f"#{_di+1} {_det_disp_name or '閾値超え'} ({len(_tmkt)}点)",
                                    "color": _color,
                                })
                            # OK/NG 判定（基準±Δ）
                            _pm_on = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                            _pm_dt = float(st.session_state.get(f"{_dkey}_pm_dt", 0.0))
                            _pm_dv = float(st.session_state.get(f"{_dkey}_pm_dv", 0.0))
                            if _pm_on and (_pm_dt > 0 or _pm_dv > 0):
                                _valid_thr = [p for p in _cyc_pts_thr if p is not None]
                                if _valid_thr:
                                    _ref_t_m = float(np.mean([p[0] for p in _valid_thr]))
                                    _ref_v_m = float(np.mean([p[1] for p in _valid_thr]))
                                    for _jt, _pts_jt in enumerate(_cyc_pts_thr):
                                        if _pts_jt is None:
                                            peak_ng_flags[_jt] = True; continue
                                        _tc_c, _vc_c = _pts_jt
                                        if _pm_dt > 0 and abs(_tc_c - _ref_t_m) > _pm_dt:
                                            peak_ng_flags[_jt] = True
                                        if _pm_dv > 0 and abs(_vc_c - _ref_v_m) > _pm_dv:
                                            peak_ng_flags[_jt] = True
                            # 傾向解析に出す
                            if bool(st.session_state.get(f"{_dkey}_trend_on", False)):
                                if "wi_det_trend" not in st.session_state:
                                    st.session_state["wi_det_trend"] = {}
                                st.session_state["wi_det_trend"][f"{_vkey}_{_did}"] = {
                                    "label":  f"{var} #{_di+1} {_det_disp_name or '閾値超え検出'}",
                                    "color":  _color,
                                    "t_vals": [p[0] if p else None for p in _cyc_pts_thr],
                                    "v_vals": [p[1] if p else None for p in _cyc_pts_thr],
                                }

                    elif _dtype in ("最大値点", "最小値点"):
                        # ── 最大値点 / 最小値点 取得 ──────────────────
                        _is_max_pt = (_dtype == "最大値点")
                        _vlabel_pt = "最大値" if _is_max_pt else "最小値"
                        st.caption(
                            f"各サイクルの**検索範囲内**の{_vlabel_pt}点 (t, v) を記録し、"
                            "マーカーとして表示します。OK/NG 範囲判定も設定できます。"
                        )
                        _ptp_hd, _ptp_on_col = st.columns([8, 1])
                        with _ptp_on_col:
                            st.markdown("**有効**")
                            st.checkbox("ON", key=f"{_dkey}_on")

                        # ── 検索範囲 (t / v) ─────────────────────────
                        st.markdown("**🔍 検索範囲**")
                        _srng_t1, _srng_t2, _srng_t3 = st.columns([1, 2, 2])
                        with _srng_t1:
                            st.markdown("**t [ms]**")
                            st.checkbox("指定する", key=f"{_dkey}_use_range")
                        _use_srng_t = bool(st.session_state.get(f"{_dkey}_use_range", False))
                        with _srng_t2:
                            st.number_input("t 開始", value=0.0, step=5.0,
                                            key=f"{_dkey}_range_s",
                                            disabled=not _use_srng_t)
                        with _srng_t3:
                            st.number_input("t 終了", value=200.0, step=5.0,
                                            key=f"{_dkey}_range_e",
                                            disabled=not _use_srng_t)
                        _srng_v1, _srng_v2, _srng_v3 = st.columns([1, 2, 2])
                        with _srng_v1:
                            st.markdown("**v 値**")
                            st.checkbox("指定する", key=f"{_dkey}_use_vrange")
                        _use_srng_v = bool(st.session_state.get(f"{_dkey}_use_vrange", False))
                        with _srng_v2:
                            st.number_input("v 下限", value=0.0, step=0.1,
                                            key=f"{_dkey}_vrange_lo",
                                            disabled=not _use_srng_v)
                        with _srng_v3:
                            st.number_input("v 上限", value=1.0, step=0.1,
                                            key=f"{_dkey}_vrange_hi",
                                            disabled=not _use_srng_v)
                        # 検索範囲パラメータ読み込み
                        _pt_srng_s = float(st.session_state.get(f"{_dkey}_range_s", 0.0)) \
                                     if _use_srng_t else None
                        _pt_srng_e = float(st.session_state.get(f"{_dkey}_range_e", 200.0)) \
                                     if _use_srng_t else None
                        _pt_srng_vlo = float(st.session_state.get(f"{_dkey}_vrange_lo", 0.0)) \
                                       if _use_srng_v else None
                        _pt_srng_vhi = float(st.session_state.get(f"{_dkey}_vrange_hi", 1.0)) \
                                       if _use_srng_v else None

                        # OK/NG 判定（基準±Δ）& 傾向解析
                        st.markdown("**✅ OK/NG 判定**")
                        _pt_pm1, _pt_pm2, _pt_pm3 = st.columns([1.5, 2, 2])
                        with _pt_pm1:
                            st.checkbox("基準±Δで判定", key=f"{_dkey}_pm_on",
                                        help="現データの平均検出値を基準に±許容差を超えたサイクルをNG")
                        _pm_on_pt = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                        with _pt_pm2:
                            st.number_input("t 許容差 [ms]（0=無効）", min_value=0.0,
                                            value=0.0, step=1.0, key=f"{_dkey}_pm_dt",
                                            disabled=not _pm_on_pt)
                        with _pt_pm3:
                            st.number_input("v 許容差（0=無効）", min_value=0.0,
                                            value=0.0, step=0.01, key=f"{_dkey}_pm_dv",
                                            disabled=not _pm_on_pt)
                        st.checkbox("📈 傾向解析に出す", key=f"{_dkey}_trend_on",
                                    help="検出点の t・v 値をサイクルごとに 📈 傾向解析タブへ送ります")

                        _det_on_pt = bool(st.session_state.get(f"{_dkey}_on", False))
                        if _det_on_pt:
                            _pt_mkt, _pt_mkv = [], []
                            _cyc_pts_peak = []
                            for _jp, (t_sw, v_sw) in enumerate(step_waves):
                                # t 検索範囲でマスク
                                if _pt_srng_s is not None and _pt_srng_e is not None:
                                    _mp_t = (t_sw >= _pt_srng_s) & (t_sw <= _pt_srng_e)
                                    _tp_w = t_sw[_mp_t] if _mp_t.sum() > 0 else t_sw
                                    _vp_w = v_sw[_mp_t] if _mp_t.sum() > 0 else v_sw
                                else:
                                    _tp_w = t_sw; _vp_w = v_sw
                                # v 検索範囲でマスク
                                if _pt_srng_vlo is not None and _pt_srng_vhi is not None:
                                    _mp_v = (_vp_w >= _pt_srng_vlo) & (_vp_w <= _pt_srng_vhi)
                                    if _mp_v.sum() > 0:
                                        _tp_w = _tp_w[_mp_v]; _vp_w = _vp_w[_mp_v]
                                if len(_vp_w) == 0:
                                    _cyc_pts_peak.append(None); continue
                                _idx_pt = int(np.nanargmax(_vp_w)
                                              if _is_max_pt else np.nanargmin(_vp_w))
                                _t_pt_v = float(_tp_w[_idx_pt])
                                _v_pt_v = float(_vp_w[_idx_pt])
                                _pt_mkt.append(_t_pt_v); _pt_mkv.append(_v_pt_v)
                                _cyc_pts_peak.append((_t_pt_v, _v_pt_v))
                            if _pt_mkt:
                                all_inf_markers.append({
                                    "t": _pt_mkt, "v": _pt_mkv,
                                    "label": f"#{_di+1} {_det_disp_name or _vlabel_pt + '点'} ({len(_pt_mkt)}点)",
                                    "color": _color,
                                })
                            # OK/NG 判定（基準±Δ）
                            _pm_on = bool(st.session_state.get(f"{_dkey}_pm_on", False))
                            _pm_dt = float(st.session_state.get(f"{_dkey}_pm_dt", 0.0))
                            _pm_dv = float(st.session_state.get(f"{_dkey}_pm_dv", 0.0))
                            if _pm_on and (_pm_dt > 0 or _pm_dv > 0):
                                _valid_pk = [p for p in _cyc_pts_peak if p is not None]
                                if _valid_pk:
                                    _ref_t_m = float(np.mean([p[0] for p in _valid_pk]))
                                    _ref_v_m = float(np.mean([p[1] for p in _valid_pk]))
                                    for _jp2, _pts_jp in enumerate(_cyc_pts_peak):
                                        if _pts_jp is None:
                                            peak_ng_flags[_jp2] = True; continue
                                        _tp_c, _vp_c = _pts_jp
                                        if _pm_dt > 0 and abs(_tp_c - _ref_t_m) > _pm_dt:
                                            peak_ng_flags[_jp2] = True
                                        if _pm_dv > 0 and abs(_vp_c - _ref_v_m) > _pm_dv:
                                            peak_ng_flags[_jp2] = True
                            # 傾向解析に出す
                            if bool(st.session_state.get(f"{_dkey}_trend_on", False)):
                                if "wi_det_trend" not in st.session_state:
                                    st.session_state["wi_det_trend"] = {}
                                st.session_state["wi_det_trend"][f"{_vkey}_{_did}"] = {
                                    "label":  f"{var} #{_di+1} {_det_disp_name or _vlabel_pt + '点'}",
                                    "color":  _color,
                                    "t_vals": [p[0] if p else None for p in _cyc_pts_peak],
                                    "v_vals": [p[1] if p else None for p in _cyc_pts_peak],
                                }

                    elif _dtype == "検出点比較":  # 検出点比較
                        # ── 検出点比較 ─────────────────────────────
                        st.caption(
                            "変曲点を検出し、その検出点が指定した合格ゾーン（t × V の矩形）に"
                            "入るか/入らないかでNG判定します。"
                        )
                        # 傾き変化点パラメータ（コンパクト版）
                        _zc1, _zc2, _zc3, _zc4, _zc5 = st.columns([2, 2, 2, 3, 1])
                        with _zc1:
                            st.markdown("**平滑化幅**")
                            st.number_input("サンプル数", min_value=1, max_value=50,
                                            value=5, step=1, key=f"{_dkey}_smooth")
                        with _zc2:
                            st.markdown("**n_L（左）**")
                            st.number_input("左サンプル", min_value=1, max_value=100,
                                            value=3, step=1, key=f"{_dkey}_nleft")
                        with _zc3:
                            st.markdown("**n_R（右）**")
                            st.number_input("右サンプル", min_value=1, max_value=100,
                                            value=3, step=1, key=f"{_dkey}_nright")
                        with _zc4:
                            _zsm = int(st.session_state.get(f"{_dkey}_smooth", 5))
                            _znl = int(st.session_state.get(f"{_dkey}_nleft",  3))
                            _znr = int(st.session_state.get(f"{_dkey}_nright", 3))
                            _zref = _slope_diff_max_ref_t(
                                step_waves[:min(5, len(step_waves))],
                                smooth_w=_zsm, n_left=_znl, n_right=_znr)
                            _zstp = max(0.0001, round(_zref / 20, 4))
                            st.markdown("**閾値 |R − L|**")
                            st.number_input(f"（参考最大 ≈ {_zref:.4f}）",
                                            min_value=0.0, value=0.0,
                                            step=_zstp, format="%.4f",
                                            key=f"{_dkey}_thresh")
                        with _zc5:
                            st.markdown("**有効**")
                            st.checkbox("ON", key=f"{_dkey}_on")
                        _zd1, _zd2 = st.columns(2)
                        with _zd1:
                            st.checkbox("📈 増加方向を検出 (R > L)",
                                        value=True, key=f"{_dkey}_dir_inc")
                        with _zd2:
                            st.checkbox("📉 減少方向を検出 (L > R)",
                                        value=True, key=f"{_dkey}_dir_dec")

                        st.markdown("**🎯 合格ゾーン（検出点がこの矩形に入ること）**")
                        _zr1, _zr2, _zr3, _zr4 = st.columns(4)
                        with _zr1:
                            st.number_input("t 開始 [ms]", value=0.0, step=5.0,
                                            key=f"{_dkey}_zt_s")
                        with _zr2:
                            st.number_input("t 終了 [ms]", value=100.0, step=5.0,
                                            key=f"{_dkey}_zt_e")
                        with _zr3:
                            st.number_input("V 下限", value=0.0, step=0.1,
                                            key=f"{_dkey}_zv_lo")
                        with _zr4:
                            st.number_input("V 上限", value=0.0, step=0.1,
                                            key=f"{_dkey}_zv_hi")

                        st.markdown("**⚠️ NG 条件（独立設定）**")
                        _zng1, _zng2 = st.columns(2)
                        with _zng1:
                            st.checkbox("合格ゾーン内に検出点がない → NG",
                                        key=f"{_dkey}_ng_empty")
                        with _zng2:
                            st.checkbox("合格ゾーン外に検出点がある → NG",
                                        key=f"{_dkey}_ng_outside")

                        _zdet_on     = bool(st.session_state.get(f"{_dkey}_on", False))
                        _zdet_thr    = float(st.session_state.get(f"{_dkey}_thresh", 0.0))
                        _zsm_d       = int(st.session_state.get(f"{_dkey}_smooth", 5))
                        _znl_d       = int(st.session_state.get(f"{_dkey}_nleft",  3))
                        _znr_d       = int(st.session_state.get(f"{_dkey}_nright", 3))
                        _zdinc       = bool(st.session_state.get(f"{_dkey}_dir_inc", True))
                        _zddec       = bool(st.session_state.get(f"{_dkey}_dir_dec", True))
                        _zts         = float(st.session_state.get(f"{_dkey}_zt_s", 0.0))
                        _zte         = float(st.session_state.get(f"{_dkey}_zt_e", 100.0))
                        _zvlo        = float(st.session_state.get(f"{_dkey}_zv_lo", 0.0))
                        _zvhi        = float(st.session_state.get(f"{_dkey}_zv_hi", 0.0))
                        _zuse_vrange = _zvlo < _zvhi
                        _zng_empty   = bool(st.session_state.get(f"{_dkey}_ng_empty", False))
                        _zng_outside = bool(st.session_state.get(f"{_dkey}_ng_outside", False))

                        if _zdet_on and _zdet_thr > 0 and (_zng_empty or _zng_outside):
                            _zmkt_in, _zmkv_in   = [], []
                            _zmkt_out, _zmkv_out = [], []
                            for j, (t_sw, v_sw) in enumerate(step_waves):
                                _zts_arr = _detect_inflections(
                                    t_sw, v_sw,
                                    smooth_w=_zsm_d, n_left=_znl_d, n_right=_znr_d,
                                    threshold=_zdet_thr,
                                    detect_increase=_zdinc, detect_decrease=_zddec)
                                _z_in, _z_out = [], []
                                for ti in _zts_arr:
                                    vi = float(np.interp(ti, t_sw, v_sw))
                                    in_t = _zts <= ti <= _zte
                                    in_v = (not _zuse_vrange) or (_zvlo <= vi <= _zvhi)
                                    if in_t and in_v:
                                        _z_in.append((ti, vi))
                                        _zmkt_in.append(ti); _zmkv_in.append(vi)
                                    else:
                                        _z_out.append((ti, vi))
                                        _zmkt_out.append(ti); _zmkv_out.append(vi)
                                # NG判定
                                if _zng_empty and len(_z_in) == 0:
                                    peak_ng_flags[j] = True
                                if _zng_outside and len(_z_out) > 0:
                                    peak_ng_flags[j] = True
                            if _zmkt_in:
                                all_inf_markers.append({
                                    "t": _zmkt_in, "v": _zmkv_in,
                                    "label": f"#{_di+1} 検出点（ゾーン内） ({len(_zmkt_in)}点)",
                                    "color": _color,
                                })
                            if _zmkt_out:
                                all_inf_markers.append({
                                    "t": _zmkt_out, "v": _zmkv_out,
                                    "label": f"#{_di+1} 検出点（ゾーン外） ({len(_zmkt_out)}点)",
                                    "color": "red",
                                })

                    elif _dtype == "数式":
                        # ── 数式 ─────────────────────────────────────
                        st.caption(
                            "他の検出点の座標・値を組み合わせた数式でNG判定します。  \n"
                            "構文: `#N.t`（N番目の検出点の時間座標）、`#N.v`（N番目の検出点の値）  \n"
                            "演算子: `+` `-` `*` `/` `<` `>` `AND` `OR`  \n"
                            "例: `#1.t - #2.t`（1番目と2番目の検出時間差）"
                        )
                        _fm_c1, _fm_c2 = st.columns([5, 1])
                        with _fm_c1:
                            st.text_input("数式", value="",
                                          placeholder="#1.t - #2.t",
                                          key=f"{_dkey}_expr")
                        with _fm_c2:
                            st.markdown("**有効**")
                            st.checkbox("ON", key=f"{_dkey}_on")

                        _fm_hi_c, _fm_lo_c = st.columns(2)
                        with _fm_hi_c:
                            st.number_input("NG 上限（超えたらNG、0=判定なし）",
                                            value=0.0, step=0.1,
                                            key=f"{_dkey}_hi")
                        with _fm_lo_c:
                            st.number_input("NG 下限（下回ったらNG、0=判定なし）",
                                            value=0.0, step=0.1,
                                            key=f"{_dkey}_lo")
                        st.checkbox("📈 傾向解析に出す", key=f"{_dkey}_trend_on",
                                    help="数式の計算結果をサイクルごとに 📈 傾向解析タブへ送ります")

                        _fm_expr = str(st.session_state.get(f"{_dkey}_expr", ""))
                        _fm_on   = bool(st.session_state.get(f"{_dkey}_on", False))
                        _fm_hi   = float(st.session_state.get(f"{_dkey}_hi", 0.0))
                        _fm_lo   = float(st.session_state.get(f"{_dkey}_lo", 0.0))

                        if _fm_on and _fm_expr.strip():
                            _fm_expr_py = _translate_formula(_fm_expr)
                            _fm_results = []
                            for _fj in range(len(step_waves)):
                                _vd = {}
                                for _ref_di, _ref_pts in _det_pts_per_cycle.items():
                                    if _fj < len(_ref_pts) and _ref_pts[_fj] is not None:
                                        _rc, _rv = _ref_pts[_fj]
                                        _vd[f"p{_ref_di + 1}t"] = _rc
                                        _vd[f"p{_ref_di + 1}v"] = _rv
                                _res = _safe_eval_expr(_fm_expr_py, _vd)
                                _fm_results.append(_res)
                                if _res is not None:
                                    if _fm_hi != 0.0 and _res > _fm_hi:
                                        peak_ng_flags[_fj] = True
                                    if _fm_lo != 0.0 and _res < _fm_lo:
                                        peak_ng_flags[_fj] = True

                            # トレンドグラフ（サイクルごとの数式結果）
                            _fm_valid = [(j, r) for j, r in enumerate(_fm_results)
                                         if r is not None]
                            if _fm_valid:
                                _fm_cycs = [v[0] for v in _fm_valid]
                                _fm_vals = [v[1] for v in _fm_valid]
                                _fm_bar_colors = [
                                    "red" if (
                                        (_fm_hi != 0.0 and v > _fm_hi) or
                                        (_fm_lo != 0.0 and v < _fm_lo)
                                    ) else _color
                                    for v in _fm_vals
                                ]
                                _fig_fm = go.Figure()
                                _fig_fm.add_trace(go.Bar(
                                    x=_fm_cycs, y=_fm_vals,
                                    marker_color=_fm_bar_colors,
                                    name="数式結果",
                                ))
                                if _fm_hi != 0.0:
                                    _fig_fm.add_hline(
                                        y=_fm_hi, line_color="red",
                                        line_dash="dash",
                                        annotation_text=f"上限 {_fm_hi}",
                                        annotation_position="top right",
                                    )
                                if _fm_lo != 0.0:
                                    _fig_fm.add_hline(
                                        y=_fm_lo, line_color="blue",
                                        line_dash="dash",
                                        annotation_text=f"下限 {_fm_lo}",
                                        annotation_position="bottom right",
                                    )
                                _fig_fm.update_layout(
                                    height=220,
                                    margin=dict(t=10, b=30, l=50, r=10),
                                    xaxis_title="サイクル番号",
                                    yaxis_title="数式結果",
                                )
                                st.plotly_chart(_fig_fm, width="stretch",
                                                key=f"{_dkey}_fm_trend")
                            else:
                                st.info(
                                    "参照している検出点が未検出のため数式を評価できません。  \n"
                                    "参照先の検出点タイプ（傾き変化点・閾値超え検出）を先に追加し有効にしてください。"
                                )

            # ── 検出点追加（リスト末尾） ────────────────────────────
            _badd_c1, _badd_c2 = st.columns([5, 2])
            with _badd_c1:
                st.selectbox("", _DET_TYPES_T, key=f"{_vkey}_t_det_type_sel",
                             label_visibility="collapsed",
                             help="点取得系: 傾き変化点・閾値超え検出・最大値点・最小値点  |  判定系: それ以外")
            with _badd_c2:
                if st.button("＋ 追加", key=f"{_vkey}_t_det_add", use_container_width=True):
                    _cnt = st.session_state[_tdet_cnt_key]
                    _new_type = st.session_state.get(f"{_vkey}_t_det_type_sel", "傾き変化点")
                    _tdet_list.append({"id": f"td{_cnt}", "type": _new_type})
                    st.session_state[_tdet_list_key] = _tdet_list
                    st.session_state[_tdet_cnt_key] = _cnt + 1
                    st.rerun()

            if _t_del_idx is not None:
                _tdet_list.pop(_t_del_idx)
                st.session_state[_tdet_list_key] = _tdet_list
                st.rerun()

            # ── グラフ描画 ─────────────────────────────────────────
            fig = go.Figure()
            _in_cmp_wv = _compare_entries is not None and len(_compare_entries) > 0

            _MAX_WAVE = 80
            if not _in_cmp_wv:
                # 通常モード: 個別サイクル波形（グレー / 赤）
                for j, (t_sw, v_sw) in enumerate(step_waves):
                    if j >= _MAX_WAVE:
                        break
                    _env_ng  = ng_flags[j]      if j < len(ng_flags)      else False
                    _peak_ng = peak_ng_flags[j] if j < len(peak_ng_flags) else False
                    if _env_ng or _peak_ng:
                        color_cyc = "rgba(220,50,50,0.25)"
                    else:
                        color_cyc = "rgba(100,120,200,0.12)"
                    fig.add_trace(go.Scatter(
                        x=t_sw, y=v_sw, mode="lines",
                        line=dict(color=color_cyc, width=1),
                        showlegend=False,
                    ))
                fig.add_trace(go.Scatter(
                    x=t_common, y=mean_v, mode="lines",
                    line=dict(color="royalblue", width=2.5),
                    name="平均波形（比較）" if _ref_df is not None else "平均波形",
                ))
            else:
                # 比較モード: 各CSV の平均波形を色分けして重ねる
                # まず現在CSVの平均波形を先頭に（デフォルト: ロイヤルブルー）
                fig.add_trace(go.Scatter(
                    x=t_common, y=mean_v, mode="lines",
                    line=dict(color="royalblue", width=2.5),
                    name="現在CSV（平均）",
                ))
                for _ce_wv in _compare_entries:
                    try:
                        _ce_wavs_cmp = cached_waveforms(
                            _ce_wv["df"], trigger_col, edge, tuple(waveform_vars)
                        )
                        if _standalone:
                            _ce_soff_arr = np.zeros(len(_ce_wavs_cmp))
                        else:
                            _ce_rd_wv = _ce_wv.get("result_df")
                            if _ce_rd_wv is None:
                                continue
                            _ce_scol = start_col if "start_col" in dir() else (
                                f"{name}_遅れ[ms]" if mode == "single"
                                else f"{name}_start[ms]"
                            )
                            if _ce_scol not in _ce_rd_wv.columns:
                                continue
                            _ce_soff_arr = _ce_rd_wv[_ce_scol].values
                        _ce_sw_cmp = []
                        _n_ce = min(len(_ce_wavs_cmp), len(_ce_soff_arr))
                        for _ri in range(_n_ce):
                            _rcyc = _ce_wavs_cmp[_ri]
                            if var not in _rcyc.columns:
                                continue
                            _rt = _rcyc["time_offset_ms"].values
                            _rv = _rcyc[var].values.astype(np.float64)
                            _ce_soff = float(_ce_soff_arr[_ri])
                            if np.isnan(_ce_soff):
                                continue
                            _rt_rel = _rt - _ce_soff
                            _rm = (_rt_rel >= -win_pre) & (_rt_rel <= win_post)
                            if _rm.sum() >= 2:
                                _ce_sw_cmp.append((_rt_rel[_rm], _rv[_rm]))
                        if not _ce_sw_cmp:
                            continue
                        _ce_mat = np.full((len(_ce_sw_cmp), len(t_common)), np.nan)
                        for _rj, (_rts, _rvs) in enumerate(_ce_sw_cmp):
                            _ridx = np.searchsorted(_rts, t_common).clip(0, len(_rts) - 1)
                            _rin  = (t_common >= _rts[0]) & (t_common <= _rts[-1])
                            _ce_mat[_rj, _rin] = _rvs[_ridx[_rin]]
                        _ce_mean_wv = np.nanmean(_ce_mat, axis=0)
                        _ce_std_wv  = np.nanstd(_ce_mat,  axis=0)
                        _ce_col_wv  = _ce_wv["color"]
                        _ce_hex_wv  = _ce_col_wv.lstrip("#")
                        _ce_r_wv = int(_ce_hex_wv[0:2], 16)
                        _ce_g_wv = int(_ce_hex_wv[2:4], 16)
                        _ce_b_wv = int(_ce_hex_wv[4:6], 16)
                        # ±σ 帯
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([t_common, t_common[::-1]]),
                            y=np.concatenate([
                                _ce_mean_wv + _ce_std_wv,
                                (_ce_mean_wv - _ce_std_wv)[::-1]
                            ]),
                            fill="toself",
                            fillcolor=f"rgba({_ce_r_wv},{_ce_g_wv},{_ce_b_wv},0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name=f"{_ce_wv['label']} ±σ",
                            showlegend=True,
                        ))
                        # 平均波形
                        fig.add_trace(go.Scatter(
                            x=t_common, y=_ce_mean_wv, mode="lines",
                            line=dict(color=_ce_col_wv, width=2.5),
                            name=_ce_wv["label"],
                        ))
                    except Exception:
                        continue

            # ── 基準CSV オーバーレイ（参照データ）──────────────────────
            if _ref_mean_v is not None:
                _rhi = _ref_mean_v + _ref_std_v
                _rlo = _ref_mean_v - _ref_std_v
                fig.add_trace(go.Scatter(
                    x=np.concatenate([t_common, t_common[::-1]]),
                    y=np.concatenate([_rhi, _rlo[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,140,0,0.12)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="基準 ±1σ",
                    showlegend=True,
                ))
                fig.add_trace(go.Scatter(
                    x=t_common, y=_ref_mean_v, mode="lines",
                    line=dict(color="darkorange", width=2.5, dash="dash"),
                    name="基準 平均",
                ))

            fig.add_vline(x=0, line_color="gray", line_dash="dot",
                          annotation_text="ステップ開始", annotation_position="top left")

            ng_count = sum(ng_flags)
            pk_ng_count = sum(peak_ng_flags)
            if ng_count > 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color="rgba(220,50,50,0.7)", width=2),
                    name=f"波形NG ({ng_count}/{len(step_waves)})",
                ))
            if pk_ng_count > 0:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines",
                    line=dict(color="rgba(200,50,200,0.7)", width=2),
                    name=f"ピークNG ({pk_ng_count}/{len(step_waves)})",
                ))

            # 検出マーカー・上下判定比較ライン（各検出項目の色で描画）
            for _mk in all_inf_markers:
                if "_bnd_hi" in _mk:
                    # 上下判定比較の境界ライン
                    _tc = np.array(_mk["_t_common"])
                    _hi = np.array(_mk["_bnd_hi"])
                    _lo = np.array(_mk["_bnd_lo"])
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([_tc, _tc[::-1]]),
                        y=np.concatenate([_hi, _lo[::-1]]),
                        fill="toself",
                        fillcolor=f"rgba(0,180,80,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name=_mk["label"], showlegend=True,
                    ))
                    fig.add_trace(go.Scatter(
                        x=_tc, y=_hi, mode="lines",
                        line=dict(color=_mk["color"], width=1.5, dash="dash"),
                        name=f"{_mk['label']} 上限", showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=_tc, y=_lo, mode="lines",
                        line=dict(color=_mk["color"], width=1.5, dash="dash"),
                        name=f"{_mk['label']} 下限", showlegend=False,
                    ))
                elif _mk["t"]:
                    fig.add_trace(go.Scatter(
                        x=_mk["t"], y=_mk["v"], mode="markers",
                        marker=dict(symbol="diamond", color=_mk["color"], size=8,
                                    line=dict(color="white", width=1)),
                        name=_mk["label"],
                    ))

            fig.update_layout(
                xaxis_title="ステップ開始からの時間 [ms]",
                yaxis_title=var,
                height=340,
                margin=dict(t=20, b=40, l=60, r=20),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch", key=f"{_vkey}_fig")

            # ── サマリー & 基準登録 ───────────────────────────────────
            total_w = len(step_waves)
            sm_c1, sm_c2, sm_c3, sm_c4, sm_c5, sm_c6 = st.columns(6)
            sm_c1.metric("サイクル数", total_w)
            sm_c2.metric("波形NG数", ng_count,
                         delta=None if ng_count == 0 else f"{ng_count/total_w*100:.1f}%")
            sm_c3.metric("ピークNG数", pk_ng_count,
                         delta=None if pk_ng_count == 0 else f"{pk_ng_count/total_w*100:.1f}%")
            _mx_vals = [np.nanmax(v) for _, v in step_waves]
            _mn_vals = [np.nanmin(v) for _, v in step_waves]
            sm_c4.metric("最大値（平均）", f"{float(np.mean(_mx_vals)):.3f}")
            sm_c5.metric("最小値（平均）", f"{float(np.mean(_mn_vals)):.3f}")
            sm_c6.metric("最大値σ",       f"{float(np.std(_mx_vals)):.3f}")

            bl_key = f"{pname}_{name}_{var}"
            if st.button(f"📌 現データを基準として登録（{var}）",
                         key=f"{_vkey}_bl_reg", type="secondary"):
                _wv_bl[bl_key] = {
                    "t":    t_common.tolist(),
                    "mean": mean_v.tolist(),
                    "std":  std_v.tolist(),
                }
                st.session_state[pk(pname, "wv_baseline")] = _wv_bl
                st.success(f"✅ {var} の波形基準を登録しました（{len(step_waves)} サイクル）")
                st.rerun()

            if bl_key in _wv_bl:
                if st.button(f"🗑 基準をリセット（{var}）",
                             key=f"{_vkey}_bl_del", type="secondary"):
                    _wv_bl.pop(bl_key, None)
                    st.session_state[pk(pname, "wv_baseline")] = _wv_bl
                    st.rerun()
                st.caption(f"✅ 基準登録済み")

        # ══════════════════════════════════════════════════════════
        with _tab_xy:
            # ── XY グラフ タブ ──────────────────────────────────────
            _num_cols_xy = [c for c in df.columns
                            if c != "time_offset_ms"
                            and pd.api.types.is_numeric_dtype(df[c])
                            and c != var]
            if not _num_cols_xy:
                st.info("XYグラフに使える数値変数が他にありません。")
            else:
                xy_xvar = st.selectbox(
                    "X軸変数", _num_cols_xy, index=0,
                    key=f"{_vkey}_xy_xvar",
                )

                # X軸変数を含めて波形取得（キャッシュ利用）
                _vars_needed = tuple(sorted(set(waveform_vars) | {xy_xvar}))
                try:
                    _wf_xy = cached_waveforms(df, trigger_col, edge, _vars_needed)
                except Exception as _e:
                    st.error(f"波形取得エラー: {_e}")
                    _wf_xy = []

                # ── XY コントロールパネル ───────────────────────────
                xy_c1, xy_c2, xy_c3, xy_c4 = st.columns([3, 3, 3, 3])
                with xy_c1:
                    st.markdown("**時間ウィンドウ（ステップ開始基準）**")
                    st.caption("※ 時間軸タブの表示ウィンドウとは独立して設定可")
                    xy_use_twin = st.checkbox("時間で絞り込む",
                                              key=f"{_vkey}_xy_use_twin")
                    if xy_use_twin:
                        xy_t_s = st.number_input("開始 [ms]", min_value=-500,
                                                  max_value=2000, value=0, step=5,
                                                  key=f"{_vkey}_xy_ts")
                        xy_t_e = st.number_input("終了 [ms]", min_value=-500,
                                                  max_value=2000, value=300, step=5,
                                                  key=f"{_vkey}_xy_te")
                        xy_t_e = max(xy_t_e, xy_t_s + 1)
                    else:
                        # 時間軸タブの表示ウィンドウをデフォルトとして継承
                        xy_t_s = -int(st.session_state.get(f"{_vkey}_wpre",  50))
                        xy_t_e =  int(st.session_state.get(f"{_vkey}_wpost", 300))
                with xy_c2:
                    st.markdown("**X軸表示範囲**")
                    xy_xlo = st.number_input("X下限", value=0.0, step=0.1,
                                              key=f"{_vkey}_xy_xlo")
                    xy_xhi = st.number_input("X上限", value=0.0, step=0.1,
                                              key=f"{_vkey}_xy_xhi")
                    _use_xlim = xy_xlo < xy_xhi
                with xy_c3:
                    st.markdown("**検査ウィンドウ（X範囲）**")
                    xy_insp_xs = st.number_input("X 開始", value=0.0, step=0.1,
                                                  key=f"{_vkey}_xy_ixs")
                    xy_insp_xe = st.number_input("X 終了", value=0.0, step=0.1,
                                                  key=f"{_vkey}_xy_ixe")
                    _use_xy_insp = xy_insp_xs < xy_insp_xe
                with xy_c4:
                    st.markdown("**良品範囲（Y軸）**")
                    xy_good_mode = st.radio("種別", ["手動入力", "自動（基準±Nσ）"],
                                             horizontal=True, key=f"{_vkey}_xy_gmode")
                    if xy_good_mode == "手動入力":
                        xy_good_lo = st.number_input("Y下限", value=0.0, step=0.1,
                                                      key=f"{_vkey}_xy_glo")
                        xy_good_hi = st.number_input("Y上限", value=0.0, step=0.1,
                                                      key=f"{_vkey}_xy_ghi")
                        _use_xy_manual = xy_good_lo < xy_good_hi
                        xy_good_nsig = None
                    else:
                        xy_good_nsig = st.number_input("N（±Nσ）", min_value=1.0,
                                                        max_value=6.0, value=3.0,
                                                        step=0.5, key=f"{_vkey}_xy_nsig")
                        _use_xy_manual = False
                        xy_good_lo = xy_good_hi = None

                # ── XY 波形データ抽出 ─────────────────────────────────
                xy_waves = []
                _n_cyc_xy = min(len(_wf_xy), len(start_offsets))
                for i in range(_n_cyc_xy):
                    cyc      = _wf_xy[i]
                    step_off = float(start_offsets[i]) \
                               if not np.isnan(start_offsets[i]) else None
                    if step_off is None:
                        continue
                    if xy_xvar not in cyc.columns or var not in cyc.columns:
                        continue
                    t_abs  = cyc["time_offset_ms"].values
                    t_step = t_abs - step_off
                    # 時間ウィンドウで絞り込み
                    t_mask = (t_step >= xy_t_s) & (t_step <= xy_t_e)
                    if t_mask.sum() < 2:
                        continue
                    x_arr = cyc[xy_xvar].values[t_mask].astype(np.float64)
                    y_arr = cyc[var].values[t_mask].astype(np.float64)
                    # さらに X 軸表示範囲でも絞り込み
                    if _use_xlim:
                        xmask = (x_arr >= xy_xlo) & (x_arr <= xy_xhi)
                        if xmask.sum() < 2:
                            continue
                        x_arr, y_arr = x_arr[xmask], y_arr[xmask]
                    xy_waves.append((x_arr, y_arr))

                if not xy_waves:
                    st.warning(f"{var} vs {xy_xvar}: 有効な波形データがありません")
                else:
                    # ── XY エンベロープ計算 ───────────────────────
                    _all_x   = np.concatenate([xw for xw, _ in xy_waves])
                    _x_min   = float(np.nanmin(_all_x))
                    _x_max   = float(np.nanmax(_all_x))
                    if _use_xlim:
                        _x_min = max(_x_min, xy_xlo)
                        _x_max = min(_x_max, xy_xhi)
                    x_common_xy = np.linspace(_x_min, _x_max, 400)
                    xy_mat = np.full((len(xy_waves), len(x_common_xy)), np.nan)
                    for j, (xw, yw) in enumerate(xy_waves):
                        si = np.argsort(xw)
                        xws, yws = xw[si], yw[si]
                        if len(np.unique(xws)) < 2:
                            continue
                        in_rng = (x_common_xy >= xws[0]) & (x_common_xy <= xws[-1])
                        xy_mat[j, in_rng] = np.interp(x_common_xy[in_rng], xws, yws)

                    xy_mean = np.nanmean(xy_mat, axis=0)
                    xy_std  = np.nanstd(xy_mat, axis=0)

                    # XY 良品範囲
                    _xy_bl_all = st.session_state.get(pk(pname, "wv_xy_baseline"), {})
                    _xy_bl_key = f"{pname}_{name}_{var}_{xy_xvar}"
                    if xy_good_mode == "自動（基準±Nσ）":
                        _xy_bl = _xy_bl_all.get(_xy_bl_key, {})
                        if _xy_bl:
                            _xbl_m = np.interp(x_common_xy, np.array(_xy_bl["x"]),
                                               np.array(_xy_bl["mean"]),
                                               left=np.nan, right=np.nan)
                            _xbl_s = np.interp(x_common_xy, np.array(_xy_bl["x"]),
                                               np.array(_xy_bl["std"]),
                                               left=np.nan, right=np.nan)
                            xy_env_hi = _xbl_m + xy_good_nsig * _xbl_s
                            xy_env_lo = _xbl_m - xy_good_nsig * _xbl_s
                            xy_has_env = True
                        else:
                            xy_env_hi = xy_mean + xy_good_nsig * xy_std
                            xy_env_lo = xy_mean - xy_good_nsig * xy_std
                            xy_has_env = True
                            st.caption("⚠️ XY基準未登録のため現データ平均±Nσを仮表示")
                    elif _use_xy_manual:
                        xy_env_hi  = np.full_like(x_common_xy, xy_good_hi)
                        xy_env_lo  = np.full_like(x_common_xy, xy_good_lo)
                        xy_has_env = True
                    else:
                        xy_has_env = False
                        xy_env_hi = xy_env_lo = None

                    # ── XY NG判定 ─────────────────────────────────
                    xy_ng_flags = []
                    if xy_has_env and _use_xy_insp:
                        for (xw, yw) in xy_waves:
                            si = np.argsort(xw)
                            xws, yws = xw[si], yw[si]
                            mxi = (xws >= xy_insp_xs) & (xws <= xy_insp_xe)
                            if mxi.sum() == 0:
                                xy_ng_flags.append(False)
                                continue
                            hi_p = np.interp(xws[mxi], x_common_xy, xy_env_hi)
                            lo_p = np.interp(xws[mxi], x_common_xy, xy_env_lo)
                            xy_ng_flags.append(
                                bool(np.any(yws[mxi] > hi_p) or np.any(yws[mxi] < lo_p)))
                    else:
                        xy_ng_flags = [False] * len(xy_waves)

                    # ── XY 検出点リスト ──────────────────────────────────
                    _xydet_list_key = f"{_vkey}_xy_det_list"
                    _xydet_cnt_key  = f"{_vkey}_xy_det_cnt"
                    if _xydet_list_key not in st.session_state:
                        st.session_state[_xydet_list_key] = []
                    if _xydet_cnt_key not in st.session_state:
                        st.session_state[_xydet_cnt_key] = 0
                    _xydet_list = st.session_state[_xydet_list_key]

                    _DET_TYPES_XY_POINT = ["傾き変化点", "閾値超え検出", "Y最大値点", "Y最小値点"]
                    _DET_TYPES_XY_JUDGE = ["上下判定比較", "Y最大値判定", "Y最小値判定", "検出点比較", "数式"]
                    _DET_TYPES_XY  = _DET_TYPES_XY_POINT + _DET_TYPES_XY_JUDGE
                    _DET_COLORS_XY = ["darkorange", "deeppink", "limegreen", "dodgerblue",
                                      "gold", "orchid", "coral", "steelblue"]

                    # 結果収集用
                    all_xy_inf_markers = []
                    xy_peak_ng_flags   = [False] * len(xy_waves)
                    xy_cyc_maxY = [float(np.nanmax(yw)) for _, yw in xy_waves]
                    xy_cyc_minY = [float(np.nanmin(yw)) for _, yw in xy_waves]
                    # 数式タイプが参照するサイクルごとの検出点 {xdi: [(coord, val)|None, ...]}
                    _xdet_pts_per_cycle = {}

                    if _xydet_list:
                        _xh1, _xh2, _xh3, _xh4, _xh5 = st.columns([0.5, 3.5, 1.8, 0.8, 0.5])
                        _xh1.caption("種別")
                        _xh2.caption("名前 / タイプ")
                        _xh3.caption("")
                        _xh4.caption("有効")
                    _xy_del_idx = None
                    for _xdi, _xdet in enumerate(_xydet_list):
                        _xdid   = _xdet["id"]
                        _xdtype = _xdet["type"]
                        _xdkey  = f"{_vkey}_{_xdid}"
                        _xcolor = _DET_COLORS_XY[_xdi % len(_DET_COLORS_XY)]
                        _xicon  = {"傾き変化点": "📐", "閾値超え検出": "🎯",
                                   "Y最大値点": "⬆️", "Y最小値点": "⬇️",
                                   "上下判定比較": "📊", "Y最大値判定": "🔺", "Y最小値判定": "🔻",
                                   "検出点比較": "🔲", "数式": "🧮"}.get(_xdtype, "📏")

                        _xdet_disp_name = st.session_state.get(f"{_xdkey}_name", "").strip()
                        _xis_on = bool(st.session_state.get(f"{_xdkey}_on", False))
                        _xri1, _xri2, _xri3, _xri4, _xri5 = st.columns([0.5, 3.5, 1.8, 0.8, 0.5])
                        with _xri1: st.markdown(f"{_xicon} **{_xdi+1}**")
                        with _xri2: st.markdown(_xdet_disp_name or f"*{_xdtype}*")
                        with _xri3: st.caption(_xdtype if _xdet_disp_name else "")
                        with _xri4: st.markdown("✅" if _xis_on else "⬜")
                        with _xri5:
                            if st.button("🗑", key=f"{_xdkey}_del", help="削除"):
                                _xy_del_idx = _xdi
                        with st.popover(f"⚙️ #{_xdi+1} {_xdet_disp_name or _xdtype}"):
                            st.text_input("名前（任意）", key=f"{_xdkey}_name",
                                          placeholder=f"例: {_xdtype}①",
                                          help="傾向解析ラベル・サマリーに表示されます")

                            if _xdtype == "傾き変化点":
                                st.caption(
                                    "① 移動平均でノイズ除去 → "
                                    "② 左右 n サンプル離れた点で傾き L・R を計算（dY/dX）→ "
                                    "③ |R − L| > 閾値 かつ 検出範囲内 の点を検出"
                                )
                                xyi_c1, xyi_c2, xyi_c3, xyi_c4, xyi_c5 = st.columns(
                                    [2, 2, 2, 3, 1])
                                with xyi_c1:
                                    st.markdown("**① 平滑化幅**")
                                    st.number_input("サンプル数",
                                                    min_value=1, max_value=50,
                                                    value=5, step=1,
                                                    key=f"{_xdkey}_smooth")
                                with xyi_c2:
                                    st.markdown("**n_L（左）**")
                                    st.number_input("左へ何サンプル",
                                                    min_value=1, max_value=100,
                                                    value=3, step=1,
                                                    key=f"{_xdkey}_nleft",
                                                    help="L = (Y[i]−Y[i−n_L]) / ΔX")
                                with xyi_c3:
                                    st.markdown("**n_R（右）**")
                                    st.number_input("右へ何サンプル",
                                                    min_value=1, max_value=100,
                                                    value=3, step=1,
                                                    key=f"{_xdkey}_nright",
                                                    help="R = (Y[i+n_R]−Y[i]) / ΔX")
                                with xyi_c4:
                                    _xsm = int(st.session_state.get(f"{_xdkey}_smooth", 5))
                                    _xnl = int(st.session_state.get(f"{_xdkey}_nleft",  3))
                                    _xnr = int(st.session_state.get(f"{_xdkey}_nright", 3))
                                    _xref = _slope_diff_max_ref_xy(
                                        xy_waves[:min(5, len(xy_waves))],
                                        smooth_w=_xsm, n_left=_xnl, n_right=_xnr)
                                    _xstp = max(0.0001, round(_xref / 20, 4))
                                    st.markdown("**③ 閾値 |R − L|**")
                                    st.number_input(f"（参考最大 ≈ {_xref:.4f}）",
                                                    min_value=0.0, value=0.0,
                                                    step=_xstp, format="%.4f",
                                                    key=f"{_xdkey}_thresh")
                                with xyi_c5:
                                    st.markdown("**有効**")
                                    st.checkbox("ON", key=f"{_xdkey}_on")

                                # 方向指定
                                _xdir_ca, _xdir_cb = st.columns(2)
                                with _xdir_ca:
                                    st.checkbox("📈 増加方向を検出 (R > L)",
                                                value=True, key=f"{_xdkey}_dir_inc")
                                with _xdir_cb:
                                    st.checkbox("📉 減少方向を検出 (L > R)",
                                                value=True, key=f"{_xdkey}_dir_dec")

                                xyi_rng_c1, xyi_rng_c2, xyi_rng_c3 = st.columns([1, 2, 2])
                                with xyi_rng_c1:
                                    st.markdown("**🔍 検出範囲（X値）**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_range")
                                _xuse_rng = bool(st.session_state.get(
                                    f"{_xdkey}_use_range", False))
                                with xyi_rng_c2:
                                    st.number_input("X 開始", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_range_s",
                                                    disabled=not _xuse_rng)
                                with xyi_rng_c3:
                                    st.number_input("X 終了", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_range_e",
                                                    disabled=not _xuse_rng)

                                _xsm_d  = int(st.session_state.get(f"{_xdkey}_smooth", 5))
                                _xnl_d  = int(st.session_state.get(f"{_xdkey}_nleft",  3))
                                _xnr_d  = int(st.session_state.get(f"{_xdkey}_nright", 3))
                                _xth_d  = float(st.session_state.get(f"{_xdkey}_thresh", 0.0))
                                _xrs_d  = float(st.session_state.get(
                                    f"{_xdkey}_range_s", 0.0)) if _xuse_rng else None
                                _xre_d  = float(st.session_state.get(
                                    f"{_xdkey}_range_e", 0.0)) if _xuse_rng else None
                                _xdinc  = bool(st.session_state.get(
                                    f"{_xdkey}_dir_inc", True))
                                _xddec  = bool(st.session_state.get(
                                    f"{_xdkey}_dir_dec", True))

                                # Y 検索範囲
                                xyi_vrng_c1, xyi_vrng_c2, xyi_vrng_c3 = st.columns([1, 2, 2])
                                with xyi_vrng_c1:
                                    st.markdown("**🔍 Y 検索範囲**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_vrange")
                                _xuse_vrng = bool(st.session_state.get(f"{_xdkey}_use_vrange", False))
                                with xyi_vrng_c2:
                                    st.number_input("Y 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_lo",
                                                    disabled=not _xuse_vrng)
                                with xyi_vrng_c3:
                                    st.number_input("Y 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_hi",
                                                    disabled=not _xuse_vrng)
                                _xrvlo_d = float(st.session_state.get(f"{_xdkey}_vrange_lo", 0.0)) if _xuse_vrng else None
                                _xrvhi_d = float(st.session_state.get(f"{_xdkey}_vrange_hi", 1.0)) if _xuse_vrng else None

                                st.markdown("---")
                                st.markdown("**📊 3段階プレビュー**（サンプル波形 #1）")
                                _xprev_xw0, _xprev_yw0 = xy_waves[0]
                                _xprev_si0 = np.argsort(_xprev_xw0)
                                _render_inflection_debug_xy(
                                    _xprev_xw0[_xprev_si0], _xprev_yw0[_xprev_si0],
                                    smooth_w=_xsm_d, n_left=_xnl_d, n_right=_xnr_d,
                                    threshold=_xth_d,
                                    range_s=_xrs_d, range_e=_xre_d,
                                    x_label=xy_xvar, y_label=var,
                                    chart_key=f"{_xdkey}_preview",
                                )

                                # N番目の点
                                _xnth_c1, _xnth_c2 = st.columns([2, 4])
                                with _xnth_c1:
                                    st.markdown("**🎯 使用する点**")
                                with _xnth_c2:
                                    st.number_input(
                                        "N番目（0=全て, 1=最初, -1=最後）",
                                        value=0, step=1, key=f"{_xdkey}_nth",
                                        help="サイクルごとにN番目の検出点のみ使用。0=全点。")

                                # OK/NG 判定範囲
                                st.markdown("**✅ OK/NG 判定範囲**")
                                _xok_x1, _xok_x2, _xok_x3 = st.columns([1, 2, 2])
                                with _xok_x1:
                                    st.checkbox("X 範囲で判定", key=f"{_xdkey}_ok_t_on")
                                _xok_x_on = bool(st.session_state.get(f"{_xdkey}_ok_t_on", False))
                                with _xok_x2:
                                    st.number_input("X OK 下限", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_lo", disabled=not _xok_x_on)
                                with _xok_x3:
                                    st.number_input("X OK 上限", value=1.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_hi", disabled=not _xok_x_on)
                                _xok_y1, _xok_y2, _xok_y3 = st.columns([1, 2, 2])
                                with _xok_y1:
                                    st.checkbox("Y 値で判定", key=f"{_xdkey}_ok_v_on")
                                _xok_y_on = bool(st.session_state.get(f"{_xdkey}_ok_v_on", False))
                                with _xok_y2:
                                    st.number_input("Y OK 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_lo", disabled=not _xok_y_on)
                                with _xok_y3:
                                    st.number_input("Y OK 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_hi", disabled=not _xok_y_on)
                                st.checkbox("📈 傾向解析に出す", key=f"{_xdkey}_trend_on",
                                            help="検出点の X・Y 値を CSVファイルごとに 📈 傾向解析タブへ送ります")

                                _xdet_on  = bool(st.session_state.get(f"{_xdkey}_on", False))
                                _xdet_thr = float(st.session_state.get(
                                    f"{_xdkey}_thresh", 0.0))
                                _xdet_nth = int(st.session_state.get(f"{_xdkey}_nth", 0))
                                if _xdet_on and _xdet_thr > 0:
                                    _xmkx, _xmky = [], []
                                    _xcyc_pts_inf = []
                                    for (xw, yw) in xy_waves:
                                        ix_pts, _ = _detect_xy_inflections(
                                            xw, yw, smooth_w=_xsm_d,
                                            n_left=_xnl_d, n_right=_xnr_d,
                                            threshold=_xdet_thr,
                                            range_s=_xrs_d, range_e=_xre_d,
                                            detect_increase=_xdinc,
                                            detect_decrease=_xddec)
                                        si = np.argsort(xw)
                                        _xpts = [(float(xi),
                                                  float(np.interp(xi, xw[si], yw[si])))
                                                 for xi in ix_pts]
                                        # Y 検索範囲フィルター
                                        if _xrvlo_d is not None and _xrvhi_d is not None:
                                            _xpts = [(xi, yi) for xi, yi in _xpts if _xrvlo_d <= yi <= _xrvhi_d]
                                        _xsel = _select_nth_pts(_xpts, _xdet_nth)
                                        for xi, yi in _xsel:
                                            _xmkx.append(xi); _xmky.append(yi)
                                        _xcyc_pts_inf.append(
                                            _xsel[0] if _xsel else None)
                                    _xdet_pts_per_cycle[_xdi] = _xcyc_pts_inf
                                    if _xmkx:
                                        all_xy_inf_markers.append({
                                            "x": _xmkx, "y": _xmky,
                                            "label": f"#{_xdi+1} 傾き変化点 ({len(_xmkx)}点)",
                                            "color": _xcolor,
                                        })
                                    # OK/NG 範囲チェック
                                    _xinf_ok_x_lo = float(st.session_state.get(f"{_xdkey}_ok_t_lo", 0.0))
                                    _xinf_ok_x_hi = float(st.session_state.get(f"{_xdkey}_ok_t_hi", 1.0))
                                    _xinf_ok_y_lo = float(st.session_state.get(f"{_xdkey}_ok_v_lo", 0.0))
                                    _xinf_ok_y_hi = float(st.session_state.get(f"{_xdkey}_ok_v_hi", 1.0))
                                    if _xok_x_on or _xok_y_on:
                                        for _jxi, _pts_jxi in enumerate(_xcyc_pts_inf):
                                            if _pts_jxi is None:
                                                xy_peak_ng_flags[_jxi] = True; continue
                                            _xi_c, _yi_c = _pts_jxi
                                            if _xok_x_on and not (_xinf_ok_x_lo <= _xi_c <= _xinf_ok_x_hi):
                                                xy_peak_ng_flags[_jxi] = True
                                            if _xok_y_on and not (_xinf_ok_y_lo <= _yi_c <= _xinf_ok_y_hi):
                                                xy_peak_ng_flags[_jxi] = True
                                    # 傾向解析に出す
                                    if bool(st.session_state.get(f"{_xdkey}_trend_on", False)):
                                        if "wi_det_trend" not in st.session_state:
                                            st.session_state["wi_det_trend"] = {}
                                        st.session_state["wi_det_trend"][f"{_vkey}_{_xdid}"] = {
                                            "label":  f"{var} vs {xy_xvar} #{_xdi+1} 傾き変化点",
                                            "color":  _xcolor,
                                            "t_vals": [p[0] if p else None for p in _xcyc_pts_inf],
                                            "v_vals": [p[1] if p else None for p in _xcyc_pts_inf],
                                        }

                            elif _xdtype == "上下判定比較":
                                # ── XY 上下判定比較 ──────────────────────────
                                st.caption("X軸ごとに上下限を定義し、各サイクルの曲線が範囲内か判定します。")
                                _xbnd_type = st.radio(
                                    "境界タイプ",
                                    ["絶対値キーポイント", "基準±Nσ",
                                     "現データ エンベロープ±マージン",
                                     "参照サイクル±オフセット"],
                                    horizontal=True, key=f"{_xdkey}_btype",
                                )
                                _xbnd_on = st.checkbox("有効", key=f"{_xdkey}_on")

                                if _xbnd_type == "絶対値キーポイント":
                                    st.caption("X値と上限・下限を直接入力してください。")
                                    _xkp_default = pd.DataFrame({
                                        f"X ({xy_xvar})": [
                                            float(_x_min), float(_x_max)],
                                        "上限": [float(np.nanmax(xy_mean)) * 1.1,
                                                float(np.nanmax(xy_mean)) * 1.1],
                                        "下限": [float(np.nanmin(xy_mean)) * 0.9,
                                                float(np.nanmin(xy_mean)) * 0.9],
                                    })
                                    _xkp_key = f"{_xdkey}_kp_df"
                                    if _xkp_key not in st.session_state:
                                        st.session_state[_xkp_key] = _xkp_default
                                    _xkp_edited = st.data_editor(
                                        st.session_state[_xkp_key],
                                        num_rows="dynamic",
                                        use_container_width=True,
                                        key=f"{_xdkey}_kp_editor",
                                    )
                                    _xkp_x  = _xkp_edited.iloc[:, 0].values.astype(float)
                                    _xkp_hi = _xkp_edited["上限"].values.astype(float)
                                    _xkp_lo = _xkp_edited["下限"].values.astype(float)
                                    _xbnd_hi_fn = lambda xa: np.interp(xa, _xkp_x, _xkp_hi)
                                    _xbnd_lo_fn = lambda xa: np.interp(xa, _xkp_x, _xkp_lo)

                                elif _xbnd_type == "基準±Nσ":
                                    _xnsig = st.number_input(
                                        "N（±Nσ）", min_value=0.1, max_value=10.0,
                                        value=3.0, step=0.5, key=f"{_xdkey}_nsig")
                                    _xnsig_f = float(st.session_state.get(
                                        f"{_xdkey}_nsig", 3.0))
                                    _xy_bl_all2 = st.session_state.get(
                                        pk(pname, "wv_xy_baseline"), {})
                                    _xy_bl2 = _xy_bl_all2.get(
                                        f"{pname}_{name}_{var}_{xy_xvar}", {})
                                    if _xy_bl2:
                                        _xbl_m2 = np.interp(x_common_xy,
                                                             np.array(_xy_bl2["x"]),
                                                             np.array(_xy_bl2["mean"]))
                                        _xbl_s2 = np.interp(x_common_xy,
                                                             np.array(_xy_bl2["x"]),
                                                             np.array(_xy_bl2["std"]))
                                    else:
                                        _xbl_m2, _xbl_s2 = xy_mean, xy_std
                                        st.caption("⚠️ XY基準未登録のため現データ平均±Nσを仮使用")
                                    _xbnd_hi_fn = lambda xa: np.interp(
                                        xa, x_common_xy, _xbl_m2 + _xnsig_f * _xbl_s2)
                                    _xbnd_lo_fn = lambda xa: np.interp(
                                        xa, x_common_xy, _xbl_m2 - _xnsig_f * _xbl_s2)

                                elif _xbnd_type == "現データ エンベロープ±マージン":
                                    _xenv_mg = st.number_input(
                                        "マージン（上下に加算）", value=0.0, step=0.1,
                                        key=f"{_xdkey}_margin")
                                    _xenv_mg_f = float(st.session_state.get(
                                        f"{_xdkey}_margin", 0.0))
                                    _xbnd_hi_fn = lambda xa: np.interp(
                                        xa, x_common_xy,
                                        np.nanmax(xy_mat, axis=0) + _xenv_mg_f)
                                    _xbnd_lo_fn = lambda xa: np.interp(
                                        xa, x_common_xy,
                                        np.nanmin(xy_mat, axis=0) - _xenv_mg_f)

                                else:  # 参照サイクル±オフセット
                                    _xref_mode = st.radio(
                                        "参照元", ["基準登録データ", "サイクル番号指定"],
                                        horizontal=True, key=f"{_xdkey}_refmode")
                                    _xref_ofs = st.number_input(
                                        "±オフセット", value=0.0, step=0.1,
                                        key=f"{_xdkey}_offset")
                                    _xref_ofs_f = float(st.session_state.get(
                                        f"{_xdkey}_offset", 0.0))
                                    if _xref_mode == "基準登録データ":
                                        _xy_bl_all3 = st.session_state.get(
                                            pk(pname, "wv_xy_baseline"), {})
                                        _xy_bl3 = _xy_bl_all3.get(
                                            f"{pname}_{name}_{var}_{xy_xvar}", {})
                                        if _xy_bl3:
                                            _xref_wave = np.interp(
                                                x_common_xy,
                                                np.array(_xy_bl3["x"]),
                                                np.array(_xy_bl3["mean"]))
                                        else:
                                            _xref_wave = xy_mean
                                            st.caption("⚠️ XY基準未登録のため現データ平均を仮使用")
                                    else:
                                        _xref_cyc = st.number_input(
                                            "サイクル番号（0起算）", min_value=0,
                                            max_value=max(0, len(xy_waves) - 1),
                                            value=0, step=1, key=f"{_xdkey}_refcyc")
                                        _xrc = min(int(st.session_state.get(
                                            f"{_xdkey}_refcyc", 0)),
                                            len(xy_waves) - 1)
                                        _rcxw, _rcyw = xy_waves[_xrc]
                                        _xsi = np.argsort(_rcxw)
                                        _xref_wave = np.interp(
                                            x_common_xy, _rcxw[_xsi], _rcyw[_xsi])
                                    _xbnd_hi_fn = lambda xa: np.interp(
                                        xa, x_common_xy, _xref_wave + _xref_ofs_f)
                                    _xbnd_lo_fn = lambda xa: np.interp(
                                        xa, x_common_xy, _xref_wave - _xref_ofs_f)

                                # プレビュー
                                _xprev_hi = _xbnd_hi_fn(x_common_xy)
                                _xprev_lo = _xbnd_lo_fn(x_common_xy)
                                _xfig_bnd = go.Figure()
                                _xfig_bnd.add_trace(go.Scatter(
                                    x=np.concatenate([x_common_xy, x_common_xy[::-1]]),
                                    y=np.concatenate([_xprev_hi, _xprev_lo[::-1]]),
                                    fill="toself", fillcolor="rgba(0,180,80,0.12)",
                                    line=dict(color="rgba(0,0,0,0)"), name="合格範囲",
                                ))
                                _xfig_bnd.add_trace(go.Scatter(
                                    x=x_common_xy, y=_xprev_hi, mode="lines",
                                    line=dict(color="green", width=1.5, dash="dash"),
                                    name="上限",
                                ))
                                _xfig_bnd.add_trace(go.Scatter(
                                    x=x_common_xy, y=_xprev_lo, mode="lines",
                                    line=dict(color="green", width=1.5, dash="dash"),
                                    name="下限", showlegend=False,
                                ))
                                _xbnd_prev_xw, _xbnd_prev_yw = xy_waves[0]
                                _xbnd_prev_si = np.argsort(_xbnd_prev_xw)
                                _xfig_bnd.add_trace(go.Scatter(
                                    x=_xbnd_prev_xw[_xbnd_prev_si],
                                    y=_xbnd_prev_yw[_xbnd_prev_si],
                                    mode="lines",
                                    line=dict(color="royalblue", width=1.5),
                                    name="サンプル波形 #1",
                                ))
                                _xfig_bnd.update_layout(
                                    height=200, margin=dict(t=10, b=30, l=50, r=10),
                                    xaxis_title=xy_xvar, yaxis_title=var,
                                )
                                st.plotly_chart(_xfig_bnd, width="stretch",
                                                key=f"{_xdkey}_bnd_preview")

                                _xeff_xs_bnd, _xeff_xe_bnd, _xeff_use_bnd = \
                                    _render_item_insp_win_xy(
                                        _xdkey, _x_min, _x_max,
                                        xy_insp_xs, xy_insp_xe, _use_xy_insp)

                                # NG判定
                                if _xbnd_on:
                                    _xbnd_mk_hi = {"x": [], "y": [],
                                                   "label": f"#{_xdi+1} 上限超過",
                                                   "color": _xcolor}
                                    _xbnd_mk_lo = {"x": [], "y": [],
                                                   "label": f"#{_xdi+1} 下限超過",
                                                   "color": _xcolor}
                                    for j, (xw, yw) in enumerate(xy_waves):
                                        if _xeff_use_bnd:
                                            si_b = np.argsort(xw)
                                            xws_b, yws_b = xw[si_b], yw[si_b]
                                            mxi_b = ((xws_b >= _xeff_xs_bnd) &
                                                     (xws_b <= _xeff_xe_bnd))
                                            xi_j, yi_j = xws_b[mxi_b], yws_b[mxi_b]
                                        else:
                                            si_b = np.argsort(xw)
                                            xi_j, yi_j = xw[si_b], yw[si_b]
                                        if len(xi_j) == 0:
                                            continue
                                        hi_j = _xbnd_hi_fn(xi_j)
                                        lo_j = _xbnd_lo_fn(xi_j)
                                        o_hi = yi_j > hi_j
                                        o_lo = yi_j < lo_j
                                        if np.any(o_hi) or np.any(o_lo):
                                            xy_peak_ng_flags[j] = True
                                        for xi, yi in zip(xi_j[o_hi], yi_j[o_hi]):
                                            _xbnd_mk_hi["x"].append(float(xi))
                                            _xbnd_mk_hi["y"].append(float(yi))
                                        for xi, yi in zip(xi_j[o_lo], yi_j[o_lo]):
                                            _xbnd_mk_lo["x"].append(float(xi))
                                            _xbnd_mk_lo["y"].append(float(yi))
                                    all_xy_inf_markers.append({
                                        "x": [], "y": [],
                                        "label": f"#{_xdi+1} 上下限ライン",
                                        "color": _xcolor,
                                        "_bnd_hi": _xbnd_hi_fn(x_common_xy).tolist(),
                                        "_bnd_lo": _xbnd_lo_fn(x_common_xy).tolist(),
                                        "_x_common": x_common_xy.tolist(),
                                    })
                                    if _xbnd_mk_hi["x"]:
                                        all_xy_inf_markers.append(_xbnd_mk_hi)
                                    if _xbnd_mk_lo["x"]:
                                        all_xy_inf_markers.append(_xbnd_mk_lo)

                            elif _xdtype in ("Y最大値点", "Y最小値点"):
                                # ── Y最大値点 / Y最小値点 取得 ─────────────────────
                                _xis_max_pt = (_xdtype == "Y最大値点")
                                _xvlabel_pt = "Y最大値" if _xis_max_pt else "Y最小値"
                                st.caption(
                                    f"各サイクルの検索範囲内の**{_xvlabel_pt}**点 (X, Y) を記録し、"
                                    "マーカーとして表示します。OK/NG 範囲判定も設定できます。"
                                )
                                _xptp_hd, _xptp_on_col = st.columns([8, 1])
                                with _xptp_on_col:
                                    st.markdown("**有効**")
                                    st.checkbox("ON", key=f"{_xdkey}_on")

                                # 検索範囲 (X / Y)
                                st.markdown("**🔍 検索範囲**")
                                _xsrng_x1, _xsrng_x2, _xsrng_x3 = st.columns([1, 2, 2])
                                with _xsrng_x1:
                                    st.markdown("**X 値**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_range")
                                _xuse_srng_x = bool(st.session_state.get(f"{_xdkey}_use_range", False))
                                with _xsrng_x2:
                                    st.number_input("X 開始", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_range_s",
                                                    disabled=not _xuse_srng_x)
                                with _xsrng_x3:
                                    st.number_input("X 終了", value=1.0, step=0.5,
                                                    key=f"{_xdkey}_range_e",
                                                    disabled=not _xuse_srng_x)
                                _xsrng_y1, _xsrng_y2, _xsrng_y3 = st.columns([1, 2, 2])
                                with _xsrng_y1:
                                    st.markdown("**Y 値**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_vrange")
                                _xuse_srng_y = bool(st.session_state.get(f"{_xdkey}_use_vrange", False))
                                with _xsrng_y2:
                                    st.number_input("Y 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_lo",
                                                    disabled=not _xuse_srng_y)
                                with _xsrng_y3:
                                    st.number_input("Y 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_hi",
                                                    disabled=not _xuse_srng_y)
                                _xpt_srng_xs  = float(st.session_state.get(f"{_xdkey}_range_s",   0.0)) if _xuse_srng_x else None
                                _xpt_srng_xe  = float(st.session_state.get(f"{_xdkey}_range_e",   1.0)) if _xuse_srng_x else None
                                _xpt_srng_ylo = float(st.session_state.get(f"{_xdkey}_vrange_lo", 0.0)) if _xuse_srng_y else None
                                _xpt_srng_yhi = float(st.session_state.get(f"{_xdkey}_vrange_hi", 1.0)) if _xuse_srng_y else None

                                # OK/NG 判定範囲
                                st.markdown("**✅ OK/NG 判定範囲**")
                                _xpt_ox1, _xpt_ox2, _xpt_ox3 = st.columns([1, 2, 2])
                                with _xpt_ox1:
                                    st.checkbox("X 範囲で判定", key=f"{_xdkey}_ok_t_on")
                                _xok_x_on_pt = bool(st.session_state.get(f"{_xdkey}_ok_t_on", False))
                                with _xpt_ox2:
                                    st.number_input("X OK 下限", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_lo", disabled=not _xok_x_on_pt)
                                with _xpt_ox3:
                                    st.number_input("X OK 上限", value=1.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_hi", disabled=not _xok_x_on_pt)
                                _xpt_oy1, _xpt_oy2, _xpt_oy3 = st.columns([1, 2, 2])
                                with _xpt_oy1:
                                    st.checkbox("Y 値で判定", key=f"{_xdkey}_ok_v_on")
                                _xok_y_on_pt = bool(st.session_state.get(f"{_xdkey}_ok_v_on", False))
                                with _xpt_oy2:
                                    st.number_input("Y OK 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_lo", disabled=not _xok_y_on_pt)
                                with _xpt_oy3:
                                    st.number_input("Y OK 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_hi", disabled=not _xok_y_on_pt)
                                st.checkbox("📈 傾向解析に出す", key=f"{_xdkey}_trend_on",
                                            help="検出点の X・Y 値を CSVファイルごとに 📈 傾向解析タブへ送ります")

                                _xdet_on_pt = bool(st.session_state.get(f"{_xdkey}_on", False))
                                if _xdet_on_pt:
                                    _xpt_mkx, _xpt_mky = [], []
                                    _xcyc_pts_peak = []
                                    for _jxp, (xw, yw) in enumerate(xy_waves):
                                        # X 検索範囲マスク
                                        _xtp_w, _xyp_w = xw, yw
                                        if _xpt_srng_xs is not None and _xpt_srng_xe is not None:
                                            _xmp_x = (xw >= _xpt_srng_xs) & (xw <= _xpt_srng_xe)
                                            if _xmp_x.sum() > 0:
                                                _xtp_w = xw[_xmp_x]; _xyp_w = yw[_xmp_x]
                                        # Y 検索範囲マスク
                                        if _xpt_srng_ylo is not None and _xpt_srng_yhi is not None:
                                            _xmp_y = (_xyp_w >= _xpt_srng_ylo) & (_xyp_w <= _xpt_srng_yhi)
                                            if _xmp_y.sum() > 0:
                                                _xtp_w = _xtp_w[_xmp_y]; _xyp_w = _xyp_w[_xmp_y]
                                        if len(_xyp_w) == 0:
                                            _xcyc_pts_peak.append(None); continue
                                        _xidx_pt = int(np.nanargmax(_xyp_w) if _xis_max_pt else np.nanargmin(_xyp_w))
                                        _x_pt_v  = float(_xtp_w[_xidx_pt])
                                        _y_pt_v  = float(_xyp_w[_xidx_pt])
                                        _xpt_mkx.append(_x_pt_v); _xpt_mky.append(_y_pt_v)
                                        _xcyc_pts_peak.append((_x_pt_v, _y_pt_v))
                                    if _xpt_mkx:
                                        all_xy_inf_markers.append({
                                            "x": _xpt_mkx, "y": _xpt_mky,
                                            "label": f"#{_xdi+1} {_xvlabel_pt}点 ({len(_xpt_mkx)}点)",
                                            "color": _xcolor,
                                        })
                                    # OK/NG 範囲チェック
                                    _xpt_ok_x_lo = float(st.session_state.get(f"{_xdkey}_ok_t_lo", 0.0))
                                    _xpt_ok_x_hi = float(st.session_state.get(f"{_xdkey}_ok_t_hi", 1.0))
                                    _xpt_ok_y_lo = float(st.session_state.get(f"{_xdkey}_ok_v_lo", 0.0))
                                    _xpt_ok_y_hi = float(st.session_state.get(f"{_xdkey}_ok_v_hi", 1.0))
                                    if _xok_x_on_pt or _xok_y_on_pt:
                                        for _jxp2, _pts_jxp in enumerate(_xcyc_pts_peak):
                                            if _pts_jxp is None:
                                                xy_peak_ng_flags[_jxp2] = True; continue
                                            _xp_c, _yp_c = _pts_jxp
                                            if _xok_x_on_pt and not (_xpt_ok_x_lo <= _xp_c <= _xpt_ok_x_hi):
                                                xy_peak_ng_flags[_jxp2] = True
                                            if _xok_y_on_pt and not (_xpt_ok_y_lo <= _yp_c <= _xpt_ok_y_hi):
                                                xy_peak_ng_flags[_jxp2] = True
                                    # 傾向解析に出す
                                    if bool(st.session_state.get(f"{_xdkey}_trend_on", False)):
                                        if "wi_det_trend" not in st.session_state:
                                            st.session_state["wi_det_trend"] = {}
                                        st.session_state["wi_det_trend"][f"{_vkey}_{_xdid}"] = {
                                            "label":  f"{var} vs {xy_xvar} #{_xdi+1} {_xvlabel_pt}点",
                                            "color":  _xcolor,
                                            "t_vals": [p[0] if p else None for p in _xcyc_pts_peak],
                                            "v_vals": [p[1] if p else None for p in _xcyc_pts_peak],
                                        }

                            elif _xdtype in ("Y最大値判定", "Y最小値判定"):
                                _xis_max = (_xdtype == "Y最大値判定")
                                _xvlabel = "Y最大値" if _xis_max else "Y最小値"
                                st.caption(
                                    f"検査ウィンドウ（X範囲）内の{_xvlabel}を各サイクルで"
                                    "計算し、合格基準と照合します。")
                                _xpk_ca, _xpk_cb, _xpk_cc = st.columns([1, 2, 2])
                                with _xpk_ca:
                                    st.markdown(f"**{_xvlabel}**")
                                    st.checkbox("有効", key=f"{_xdkey}_on")
                                with _xpk_cb:
                                    st.number_input("合格 上限（超えたらNG）",
                                                    value=0.0, step=0.1,
                                                    key=f"{_xdkey}_hi", help="0 = 判定なし")
                                with _xpk_cc:
                                    st.number_input("合格 下限（下回ったらNG）",
                                                    value=0.0, step=0.1,
                                                    key=f"{_xdkey}_lo", help="0 = 判定なし")

                                _xeff_xs, _xeff_xe, _xeff_use = _render_item_insp_win_xy(
                                    _xdkey, _x_min, _x_max,
                                    xy_insp_xs, xy_insp_xe, _use_xy_insp)

                                _xdet_on = bool(st.session_state.get(f"{_xdkey}_on", False))
                                _xdet_hi = float(st.session_state.get(f"{_xdkey}_hi", 0.0))
                                _xdet_lo = float(st.session_state.get(f"{_xdkey}_lo", 0.0))
                                if _xdet_on:
                                    for j, (xw, yw) in enumerate(xy_waves):
                                        if _xeff_use:
                                            si_p = np.argsort(xw)
                                            xws_p, yws_p = xw[si_p], yw[si_p]
                                            mxi_p = ((xws_p >= _xeff_xs) &
                                                     (xws_p <= _xeff_xe))
                                            y_insp_p = (yws_p[mxi_p]
                                                        if mxi_p.sum() > 0 else yw)
                                        else:
                                            y_insp_p = yw
                                        _xval = float(
                                            np.nanmax(y_insp_p) if _xis_max
                                            else np.nanmin(y_insp_p))
                                        if _xdet_hi != 0.0 and _xval > _xdet_hi:
                                            xy_peak_ng_flags[j] = True
                                        if _xdet_lo != 0.0 and _xval < _xdet_lo:
                                            xy_peak_ng_flags[j] = True

                            elif _xdtype == "閾値超え検出":
                                # ── XY 閾値超え検出 ──────────────────────────
                                st.caption(
                                    "Y値が閾値を超えた（または下回った）X座標を検出します。"
                                    "N番目の交差点のみを使うことも可能です。"
                                )
                                _xthc1, _xthc2, _xthc3 = st.columns([3, 3, 1])
                                with _xthc1:
                                    st.markdown("**閾値（Y値）**")
                                    st.number_input("閾値", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_tv")
                                with _xthc2:
                                    st.markdown("**方向**")
                                    st.radio("方向", ["上昇 ↑", "下降 ↓", "両方"],
                                             horizontal=True, key=f"{_xdkey}_tdir")
                                with _xthc3:
                                    st.markdown("**有効**")
                                    st.checkbox("ON", key=f"{_xdkey}_on")

                                _xnth_c1, _xnth_c2 = st.columns([2, 4])
                                with _xnth_c1:
                                    st.markdown("**🎯 使用する点**")
                                with _xnth_c2:
                                    st.number_input(
                                        "N番目（0=全て, 1=最初, -1=最後）",
                                        value=1, step=1, key=f"{_xdkey}_nth",
                                        help="サイクルごとにN番目の交差点のみ使用。0=全点。")

                                _xrng_tc1, _xrng_tc2, _xrng_tc3 = st.columns([1, 2, 2])
                                with _xrng_tc1:
                                    st.markdown("**🔍 検出範囲（X値）**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_range")
                                _xuse_rng_t = bool(st.session_state.get(
                                    f"{_xdkey}_use_range", False))
                                with _xrng_tc2:
                                    st.number_input("X 開始", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_range_s",
                                                    disabled=not _xuse_rng_t)
                                with _xrng_tc3:
                                    st.number_input("X 終了", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_range_e",
                                                    disabled=not _xuse_rng_t)

                                # Y 検索範囲
                                _xrng_tv1, _xrng_tv2, _xrng_tv3 = st.columns([1, 2, 2])
                                with _xrng_tv1:
                                    st.markdown("**🔍 Y 検索範囲**")
                                    st.checkbox("指定する", key=f"{_xdkey}_use_vrange")
                                _xuse_vrng_t = bool(st.session_state.get(f"{_xdkey}_use_vrange", False))
                                with _xrng_tv2:
                                    st.number_input("Y 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_lo",
                                                    disabled=not _xuse_vrng_t)
                                with _xrng_tv3:
                                    st.number_input("Y 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_vrange_hi",
                                                    disabled=not _xuse_vrng_t)

                                _xtv_val   = float(st.session_state.get(
                                    f"{_xdkey}_tv", 0.0))
                                _xtdir_raw = st.session_state.get(
                                    f"{_xdkey}_tdir", "上昇 ↑")
                                _xtdir_str = ("rise" if "上昇" in _xtdir_raw
                                              else "fall" if "下降" in _xtdir_raw
                                              else "both")
                                _xt_nth    = int(st.session_state.get(f"{_xdkey}_nth", 1))
                                _xt_on     = bool(st.session_state.get(f"{_xdkey}_on", False))
                                _xtrs      = float(st.session_state.get(
                                    f"{_xdkey}_range_s", 0.0)) if _xuse_rng_t else None
                                _xtre      = float(st.session_state.get(
                                    f"{_xdkey}_range_e", 0.0)) if _xuse_rng_t else None
                                _xt_rvlo   = float(st.session_state.get(f"{_xdkey}_vrange_lo", 0.0)) if _xuse_vrng_t else None
                                _xt_rvhi   = float(st.session_state.get(f"{_xdkey}_vrange_hi", 1.0)) if _xuse_vrng_t else None

                                # OK/NG 判定範囲
                                st.markdown("**✅ OK/NG 判定範囲**")
                                _xtok_x1, _xtok_x2, _xtok_x3 = st.columns([1, 2, 2])
                                with _xtok_x1:
                                    st.checkbox("X 範囲で判定", key=f"{_xdkey}_ok_t_on")
                                _xtok_x_on = bool(st.session_state.get(f"{_xdkey}_ok_t_on", False))
                                with _xtok_x2:
                                    st.number_input("X OK 下限", value=0.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_lo", disabled=not _xtok_x_on)
                                with _xtok_x3:
                                    st.number_input("X OK 上限", value=1.0, step=0.5,
                                                    key=f"{_xdkey}_ok_t_hi", disabled=not _xtok_x_on)
                                _xtok_y1, _xtok_y2, _xtok_y3 = st.columns([1, 2, 2])
                                with _xtok_y1:
                                    st.checkbox("Y 値で判定", key=f"{_xdkey}_ok_v_on")
                                _xtok_y_on = bool(st.session_state.get(f"{_xdkey}_ok_v_on", False))
                                with _xtok_y2:
                                    st.number_input("Y OK 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_lo", disabled=not _xtok_y_on)
                                with _xtok_y3:
                                    st.number_input("Y OK 上限", value=1.0, step=0.1,
                                                    key=f"{_xdkey}_ok_v_hi", disabled=not _xtok_y_on)
                                st.checkbox("📈 傾向解析に出す", key=f"{_xdkey}_trend_on",
                                            help="検出点の X・Y 値を CSVファイルごとに 📈 傾向解析タブへ送ります")

                                _xtok_x_lo = float(st.session_state.get(f"{_xdkey}_ok_t_lo", 0.0))
                                _xtok_x_hi = float(st.session_state.get(f"{_xdkey}_ok_t_hi", 1.0))
                                _xtok_y_lo = float(st.session_state.get(f"{_xdkey}_ok_v_lo", 0.0))
                                _xtok_y_hi = float(st.session_state.get(f"{_xdkey}_ok_v_hi", 1.0))

                                if _xt_on:
                                    _xtmkx, _xtmky = [], []
                                    _xcyc_pts_thr = []
                                    for (xw, yw) in xy_waves:
                                        si = np.argsort(xw)
                                        _xcs = _detect_threshold_crossings(
                                            xw[si], yw[si], _xtv_val,
                                            direction=_xtdir_str,
                                            range_s=_xtrs, range_e=_xtre)
                                        # Y 検索範囲フィルター
                                        if _xt_rvlo is not None and _xt_rvhi is not None:
                                            _xcs = [(xc, yc) for xc, yc in _xcs if _xt_rvlo <= yc <= _xt_rvhi]
                                        _xsel = _select_nth_pts(_xcs, _xt_nth)
                                        for xc, yc in _xsel:
                                            _xtmkx.append(xc); _xtmky.append(yc)
                                        _xcyc_pts_thr.append(
                                            _xsel[0] if _xsel else None)
                                    _xdet_pts_per_cycle[_xdi] = _xcyc_pts_thr
                                    if _xtmkx:
                                        all_xy_inf_markers.append({
                                            "x": _xtmkx, "y": _xtmky,
                                            "label": f"#{_xdi+1} 閾値超え ({len(_xtmkx)}点)",
                                            "color": _xcolor,
                                        })
                                    # OK/NG
                                    if _xtok_x_on or _xtok_y_on:
                                        for _jxt, _pts_jxt in enumerate(_xcyc_pts_thr):
                                            if _pts_jxt is None:
                                                xy_peak_ng_flags[_jxt] = True; continue
                                            _xc_c, _yc_c = _pts_jxt
                                            if _xtok_x_on and not (_xtok_x_lo <= _xc_c <= _xtok_x_hi):
                                                xy_peak_ng_flags[_jxt] = True
                                            if _xtok_y_on and not (_xtok_y_lo <= _yc_c <= _xtok_y_hi):
                                                xy_peak_ng_flags[_jxt] = True
                                    # 傾向解析
                                    if bool(st.session_state.get(f"{_xdkey}_trend_on", False)):
                                        if "wi_det_trend" not in st.session_state:
                                            st.session_state["wi_det_trend"] = {}
                                        st.session_state["wi_det_trend"][f"{_vkey}_{_xdid}"] = {
                                            "label":  f"{var} vs {xy_xvar} #{_xdi+1} 閾値超え検出",
                                            "color":  _xcolor,
                                            "t_vals": [p[0] if p else None for p in _xcyc_pts_thr],
                                            "v_vals": [p[1] if p else None for p in _xcyc_pts_thr],
                                        }

                            elif _xdtype == "検出点比較":  # 検出点比較
                                # ── XY 検出点比較 ────────────────────────────
                                st.caption(
                                    "変曲点を検出し、その検出点が指定した合格ゾーン"
                                    "（X × Y の矩形）に入るか/入らないかでNG判定します。"
                                )
                                _xzc1, _xzc2, _xzc3, _xzc4, _xzc5 = st.columns(
                                    [2, 2, 2, 3, 1])
                                with _xzc1:
                                    st.markdown("**平滑化幅**")
                                    st.number_input("サンプル数", min_value=1,
                                                    max_value=50, value=5, step=1,
                                                    key=f"{_xdkey}_smooth")
                                with _xzc2:
                                    st.markdown("**n_L（左）**")
                                    st.number_input("左サンプル", min_value=1,
                                                    max_value=100, value=3, step=1,
                                                    key=f"{_xdkey}_nleft")
                                with _xzc3:
                                    st.markdown("**n_R（右）**")
                                    st.number_input("右サンプル", min_value=1,
                                                    max_value=100, value=3, step=1,
                                                    key=f"{_xdkey}_nright")
                                with _xzc4:
                                    _xzsm = int(st.session_state.get(
                                        f"{_xdkey}_smooth", 5))
                                    _xznl = int(st.session_state.get(
                                        f"{_xdkey}_nleft",  3))
                                    _xznr = int(st.session_state.get(
                                        f"{_xdkey}_nright", 3))
                                    _xzref = _slope_diff_max_ref_xy(
                                        xy_waves[:min(5, len(xy_waves))],
                                        smooth_w=_xzsm, n_left=_xznl, n_right=_xznr)
                                    _xzstp = max(0.0001, round(_xzref / 20, 4))
                                    st.markdown("**閾値 |R − L|**")
                                    st.number_input(
                                        f"（参考最大 ≈ {_xzref:.4f}）",
                                        min_value=0.0, value=0.0,
                                        step=_xzstp, format="%.4f",
                                        key=f"{_xdkey}_thresh")
                                with _xzc5:
                                    st.markdown("**有効**")
                                    st.checkbox("ON", key=f"{_xdkey}_on")
                                _xzdir1, _xzdir2 = st.columns(2)
                                with _xzdir1:
                                    st.checkbox("📈 増加方向を検出 (R > L)",
                                                value=True, key=f"{_xdkey}_dir_inc")
                                with _xzdir2:
                                    st.checkbox("📉 減少方向を検出 (L > R)",
                                                value=True, key=f"{_xdkey}_dir_dec")

                                st.markdown("**🎯 合格ゾーン（検出点がこの矩形に入ること）**")
                                _xzr1, _xzr2, _xzr3, _xzr4 = st.columns(4)
                                with _xzr1:
                                    st.number_input("X 開始", value=float(_x_min),
                                                    step=0.5, key=f"{_xdkey}_zx_s")
                                with _xzr2:
                                    st.number_input("X 終了", value=float(_x_max),
                                                    step=0.5, key=f"{_xdkey}_zx_e")
                                with _xzr3:
                                    st.number_input("Y 下限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_zy_lo")
                                with _xzr4:
                                    st.number_input("Y 上限", value=0.0, step=0.1,
                                                    key=f"{_xdkey}_zy_hi")

                                st.markdown("**⚠️ NG 条件（独立設定）**")
                                _xzng1, _xzng2 = st.columns(2)
                                with _xzng1:
                                    st.checkbox("合格ゾーン内に検出点がない → NG",
                                                key=f"{_xdkey}_ng_empty")
                                with _xzng2:
                                    st.checkbox("合格ゾーン外に検出点がある → NG",
                                                key=f"{_xdkey}_ng_outside")

                                _xzdet_on  = bool(st.session_state.get(
                                    f"{_xdkey}_on", False))
                                _xzdet_thr = float(st.session_state.get(
                                    f"{_xdkey}_thresh", 0.0))
                                _xzsm_d    = int(st.session_state.get(
                                    f"{_xdkey}_smooth", 5))
                                _xznl_d    = int(st.session_state.get(
                                    f"{_xdkey}_nleft",  3))
                                _xznr_d    = int(st.session_state.get(
                                    f"{_xdkey}_nright", 3))
                                _xzdinc    = bool(st.session_state.get(
                                    f"{_xdkey}_dir_inc", True))
                                _xzddec    = bool(st.session_state.get(
                                    f"{_xdkey}_dir_dec", True))
                                _xzxs      = float(st.session_state.get(
                                    f"{_xdkey}_zx_s", _x_min))
                                _xzxe      = float(st.session_state.get(
                                    f"{_xdkey}_zx_e", _x_max))
                                _xzylo     = float(st.session_state.get(
                                    f"{_xdkey}_zy_lo", 0.0))
                                _xzyhi     = float(st.session_state.get(
                                    f"{_xdkey}_zy_hi", 0.0))
                                _xzuse_y   = _xzylo < _xzyhi
                                _xzng_empty   = bool(st.session_state.get(
                                    f"{_xdkey}_ng_empty", False))
                                _xzng_outside = bool(st.session_state.get(
                                    f"{_xdkey}_ng_outside", False))

                                if (_xzdet_on and _xzdet_thr > 0 and
                                        (_xzng_empty or _xzng_outside)):
                                    _xzmk_in_x, _xzmk_in_y   = [], []
                                    _xzmk_out_x, _xzmk_out_y = [], []
                                    for j, (xw, yw) in enumerate(xy_waves):
                                        ix_pts, _ = _detect_xy_inflections(
                                            xw, yw, smooth_w=_xzsm_d,
                                            n_left=_xznl_d, n_right=_xznr_d,
                                            threshold=_xzdet_thr,
                                            detect_increase=_xzdinc,
                                            detect_decrease=_xzddec)
                                        si = np.argsort(xw)
                                        _xz_in, _xz_out = [], []
                                        for xi in ix_pts:
                                            yi = float(np.interp(xi, xw[si], yw[si]))
                                            in_x = _xzxs <= xi <= _xzxe
                                            in_y = (not _xzuse_y) or (_xzylo <= yi <= _xzyhi)
                                            if in_x and in_y:
                                                _xz_in.append((xi, yi))
                                                _xzmk_in_x.append(xi)
                                                _xzmk_in_y.append(yi)
                                            else:
                                                _xz_out.append((xi, yi))
                                                _xzmk_out_x.append(xi)
                                                _xzmk_out_y.append(yi)
                                        if _xzng_empty and len(_xz_in) == 0:
                                            xy_peak_ng_flags[j] = True
                                        if _xzng_outside and len(_xz_out) > 0:
                                            xy_peak_ng_flags[j] = True
                                    if _xzmk_in_x:
                                        all_xy_inf_markers.append({
                                            "x": _xzmk_in_x, "y": _xzmk_in_y,
                                            "label": f"#{_xdi+1} 検出点（ゾーン内）"
                                                     f" ({len(_xzmk_in_x)}点)",
                                            "color": _xcolor,
                                        })
                                    if _xzmk_out_x:
                                        all_xy_inf_markers.append({
                                            "x": _xzmk_out_x, "y": _xzmk_out_y,
                                            "label": f"#{_xdi+1} 検出点（ゾーン外）"
                                                     f" ({len(_xzmk_out_x)}点)",
                                            "color": "red",
                                        })

                            elif _xdtype == "数式":
                                # ── XY 数式 ──────────────────────────────────
                                st.caption(
                                    "他の検出点の座標・値を組み合わせた数式でNG判定します。  \n"
                                    "構文: `#N.t`（N番目の検出点のX座標）、`#N.v`（Y値）  \n"
                                    "演算子: `+` `-` `*` `/` `<` `>` `AND` `OR`  \n"
                                    "例: `#1.v - #2.v`（1番目と2番目の検出点Y値の差）"
                                )
                                _xfm_c1, _xfm_c2 = st.columns([5, 1])
                                with _xfm_c1:
                                    st.text_input("数式", value="",
                                                  placeholder="#1.v - #2.v",
                                                  key=f"{_xdkey}_expr")
                                with _xfm_c2:
                                    st.markdown("**有効**")
                                    st.checkbox("ON", key=f"{_xdkey}_on")

                                _xfm_hi_c, _xfm_lo_c = st.columns(2)
                                with _xfm_hi_c:
                                    st.number_input("NG 上限（超えたらNG、0=判定なし）",
                                                    value=0.0, step=0.1,
                                                    key=f"{_xdkey}_hi")
                                with _xfm_lo_c:
                                    st.number_input("NG 下限（下回ったらNG、0=判定なし）",
                                                    value=0.0, step=0.1,
                                                    key=f"{_xdkey}_lo")

                                _xfm_expr = str(st.session_state.get(
                                    f"{_xdkey}_expr", ""))
                                _xfm_on   = bool(st.session_state.get(
                                    f"{_xdkey}_on", False))
                                _xfm_hi   = float(st.session_state.get(
                                    f"{_xdkey}_hi", 0.0))
                                _xfm_lo   = float(st.session_state.get(
                                    f"{_xdkey}_lo", 0.0))

                                if _xfm_on and _xfm_expr.strip():
                                    _xfm_expr_py = _translate_formula(_xfm_expr)
                                    _xfm_results = []
                                    for _xfj in range(len(xy_waves)):
                                        _xvd = {}
                                        for _xref_di, _xref_pts in \
                                                _xdet_pts_per_cycle.items():
                                            if (_xfj < len(_xref_pts) and
                                                    _xref_pts[_xfj] is not None):
                                                _xrc, _xrv = _xref_pts[_xfj]
                                                _xvd[f"p{_xref_di + 1}t"] = _xrc
                                                _xvd[f"p{_xref_di + 1}v"] = _xrv
                                        _xres = _safe_eval_expr(_xfm_expr_py, _xvd)
                                        _xfm_results.append(_xres)
                                        if _xres is not None:
                                            if _xfm_hi != 0.0 and _xres > _xfm_hi:
                                                xy_peak_ng_flags[_xfj] = True
                                            if _xfm_lo != 0.0 and _xres < _xfm_lo:
                                                xy_peak_ng_flags[_xfj] = True

                                    _xfm_valid = [(j, r) for j, r in
                                                   enumerate(_xfm_results)
                                                   if r is not None]
                                    if _xfm_valid:
                                        _xfm_cycs = [v[0] for v in _xfm_valid]
                                        _xfm_vals = [v[1] for v in _xfm_valid]
                                        _xfm_bar_colors = [
                                            "red" if (
                                                (_xfm_hi != 0.0 and v > _xfm_hi) or
                                                (_xfm_lo != 0.0 and v < _xfm_lo)
                                            ) else _xcolor
                                            for v in _xfm_vals
                                        ]
                                        _xfig_fm = go.Figure()
                                        _xfig_fm.add_trace(go.Bar(
                                            x=_xfm_cycs, y=_xfm_vals,
                                            marker_color=_xfm_bar_colors,
                                            name="数式結果",
                                        ))
                                        if _xfm_hi != 0.0:
                                            _xfig_fm.add_hline(
                                                y=_xfm_hi, line_color="red",
                                                line_dash="dash",
                                                annotation_text=f"上限 {_xfm_hi}",
                                                annotation_position="top right",
                                            )
                                        if _xfm_lo != 0.0:
                                            _xfig_fm.add_hline(
                                                y=_xfm_lo, line_color="blue",
                                                line_dash="dash",
                                                annotation_text=f"下限 {_xfm_lo}",
                                                annotation_position="bottom right",
                                            )
                                        _xfig_fm.update_layout(
                                            height=220,
                                            margin=dict(t=10, b=30, l=50, r=10),
                                            xaxis_title="サイクル番号",
                                            yaxis_title="数式結果",
                                        )
                                        st.plotly_chart(
                                            _xfig_fm, width="stretch",
                                            key=f"{_xdkey}_fm_trend")
                                    else:
                                        st.info(
                                            "参照している検出点が未検出のため数式を評価できません。  \n"
                                            "参照先の検出点タイプ（傾き変化点・閾値超え検出）を"
                                            "先に追加し有効にしてください。"
                                        )

                    # ── XY 検出点追加（リスト末尾） ──────────────────
                    _xbadd_c1, _xbadd_c2 = st.columns([5, 2])
                    with _xbadd_c1:
                        st.selectbox("", _DET_TYPES_XY, key=f"{_vkey}_xy_det_type_sel",
                                     label_visibility="collapsed")
                    with _xbadd_c2:
                        if st.button("＋ 追加", key=f"{_vkey}_xy_det_add",
                                     use_container_width=True):
                            _xcnt = st.session_state[_xydet_cnt_key]
                            _xnew_type = st.session_state.get(
                                f"{_vkey}_xy_det_type_sel", "傾き変化点")
                            _xydet_list.append({"id": f"xyd{_xcnt}", "type": _xnew_type})
                            st.session_state[_xydet_list_key] = _xydet_list
                            st.session_state[_xydet_cnt_key] = _xcnt + 1
                            st.rerun()

                    if _xy_del_idx is not None:
                        _xydet_list.pop(_xy_del_idx)
                        st.session_state[_xydet_list_key] = _xydet_list
                        st.rerun()

                    # ── XY グラフ描画 ─────────────────────────────
                    fig_xy = go.Figure()

                    for j, (xw, yw) in enumerate(xy_waves):
                        if j >= _MAX_WAVE:
                            break
                        _env_ng_xy  = xy_ng_flags[j]      if j < len(xy_ng_flags)      else False
                        _peak_ng_xy = xy_peak_ng_flags[j] if j < len(xy_peak_ng_flags) else False
                        c_xy = "rgba(220,50,50,0.25)" if (_env_ng_xy or _peak_ng_xy) \
                               else "rgba(100,120,200,0.12)"
                        si = np.argsort(xw)
                        fig_xy.add_trace(go.Scatter(
                            x=xw[si], y=yw[si], mode="lines",
                            line=dict(color=c_xy, width=1),
                            showlegend=False,
                        ))

                    fig_xy.add_trace(go.Scatter(
                        x=x_common_xy, y=xy_mean, mode="lines",
                        line=dict(color="royalblue", width=2.5),
                        name="平均曲線",
                    ))

                    if xy_has_env and xy_env_hi is not None:
                        fig_xy.add_trace(go.Scatter(
                            x=np.concatenate([x_common_xy, x_common_xy[::-1]]),
                            y=np.concatenate([xy_env_hi, xy_env_lo[::-1]]),
                            fill="toself", fillcolor="rgba(0,200,100,0.10)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="良品範囲",
                        ))
                        fig_xy.add_trace(go.Scatter(
                            x=x_common_xy, y=xy_env_hi, mode="lines",
                            line=dict(color="green", width=1.5, dash="dash"),
                            name="上限", showlegend=False,
                        ))
                        fig_xy.add_trace(go.Scatter(
                            x=x_common_xy, y=xy_env_lo, mode="lines",
                            line=dict(color="green", width=1.5, dash="dash"),
                            name="下限", showlegend=False,
                        ))

                    if _use_xy_insp:
                        fig_xy.add_vrect(
                            x0=xy_insp_xs, x1=xy_insp_xe,
                            fillcolor="rgba(255,200,0,0.08)",
                            line_width=1, line_color="orange",
                            annotation_text="検査ウィンドウ",
                            annotation_position="top right",
                        )

                    xy_ng_count   = sum(xy_ng_flags)
                    xy_pk_ng_count = sum(xy_peak_ng_flags)
                    if xy_ng_count > 0:
                        fig_xy.add_trace(go.Scatter(
                            x=[None], y=[None], mode="lines",
                            line=dict(color="rgba(220,50,50,0.7)", width=2),
                            name=f"波形NG ({xy_ng_count}/{len(xy_waves)})",
                        ))
                    if xy_pk_ng_count > 0:
                        fig_xy.add_trace(go.Scatter(
                            x=[None], y=[None], mode="lines",
                            line=dict(color="rgba(200,50,200,0.7)", width=2),
                            name=f"ピークNG ({xy_pk_ng_count}/{len(xy_waves)})",
                        ))

                    for _xmk in all_xy_inf_markers:
                        if "_bnd_hi" in _xmk:
                            _xc = np.array(_xmk["_x_common"])
                            _xhi = np.array(_xmk["_bnd_hi"])
                            _xlo = np.array(_xmk["_bnd_lo"])
                            fig_xy.add_trace(go.Scatter(
                                x=np.concatenate([_xc, _xc[::-1]]),
                                y=np.concatenate([_xhi, _xlo[::-1]]),
                                fill="toself",
                                fillcolor="rgba(0,180,80,0.08)",
                                line=dict(color="rgba(0,0,0,0)"),
                                name=_xmk["label"], showlegend=True,
                            ))
                            fig_xy.add_trace(go.Scatter(
                                x=_xc, y=_xhi, mode="lines",
                                line=dict(color=_xmk["color"], width=1.5, dash="dash"),
                                name=f"{_xmk['label']} 上限", showlegend=False,
                            ))
                            fig_xy.add_trace(go.Scatter(
                                x=_xc, y=_xlo, mode="lines",
                                line=dict(color=_xmk["color"], width=1.5, dash="dash"),
                                name=f"{_xmk['label']} 下限", showlegend=False,
                            ))
                        elif _xmk["x"]:
                            fig_xy.add_trace(go.Scatter(
                                x=_xmk["x"], y=_xmk["y"], mode="markers",
                                marker=dict(symbol="diamond", color=_xmk["color"],
                                            size=8, line=dict(color="white", width=1)),
                                name=_xmk["label"],
                            ))

                    fig_xy.update_layout(
                        xaxis_title=xy_xvar,
                        yaxis_title=var,
                        height=360,
                        margin=dict(t=20, b=40, l=60, r=20),
                        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_xy, width="stretch",
                                    key=f"{_vkey}_xy_fig")

                    # ── XY サマリー & 基準登録 ─────────────────────
                    _xy_maxY = xy_cyc_maxY if xy_cyc_maxY else [np.nanmax(yw) for _, yw in xy_waves]
                    _xy_minY = xy_cyc_minY if xy_cyc_minY else [np.nanmin(yw) for _, yw in xy_waves]
                    xy_sm1, xy_sm2, xy_sm3, xy_sm4, xy_sm5, xy_sm6 = st.columns(6)
                    xy_sm1.metric("サイクル数", len(xy_waves))
                    xy_sm2.metric("波形NG数", xy_ng_count,
                                  delta=None if xy_ng_count == 0
                                  else f"{xy_ng_count/len(xy_waves)*100:.1f}%")
                    xy_sm3.metric("ピークNG数", xy_pk_ng_count,
                                  delta=None if xy_pk_ng_count == 0
                                  else f"{xy_pk_ng_count/len(xy_waves)*100:.1f}%")
                    xy_sm4.metric("Y最大値（平均）", f"{float(np.nanmean(_xy_maxY)):.3f}")
                    xy_sm5.metric("Y最小値（平均）", f"{float(np.nanmean(_xy_minY)):.3f}")
                    xy_sm6.metric("Y最大値σ",       f"{float(np.nanstd(_xy_maxY)):.3f}")

                    if st.button(f"📌 XY基準を登録（{var} vs {xy_xvar}）",
                                 key=f"{_vkey}_xy_bl_reg", type="secondary"):
                        _xy_bl_all[_xy_bl_key] = {
                            "x":    x_common_xy.tolist(),
                            "mean": xy_mean.tolist(),
                            "std":  xy_std.tolist(),
                        }
                        st.session_state[pk(pname, "wv_xy_baseline")] = _xy_bl_all
                        st.success(
                            f"✅ XY基準登録: {var} vs {xy_xvar}（{len(xy_waves)}サイクル）")
                        st.rerun()

                    if _xy_bl_key in _xy_bl_all:
                        if st.button(f"🗑 XY基準リセット（{var} vs {xy_xvar}）",
                                     key=f"{_vkey}_xy_bl_del", type="secondary"):
                            _xy_bl_all.pop(_xy_bl_key, None)
                            st.session_state[pk(pname, "wv_xy_baseline")] = _xy_bl_all
                            st.rerun()
                        st.caption("✅ XY基準登録済み")

@st.dialog("波形検査 - 検出点編集", width="large")
def _wi_edit_dialog():
    ss   = st.session_state
    item = ss.get("wi_edit_target", {})
    if not item:
        st.error("編集対象が見つかりません"); return

    _r_svkey = item["_svkey"]
    _r_did   = item["_did"]
    _r_sv    = item["_sv"]
    _r_graph = item["_graph"]
    _r_dtype = item["_dtype"]
    _r_no    = item["No"]
    _dkey    = f"{_r_svkey}_{_r_did}"
    _vkey    = _r_svkey
    _dpfx    = f"_dlg_{_dkey}_"      # draft: 検出点設定
    _vpfx    = f"_dlg_{_r_svkey}_"   # draft: 変数レベル設定

    # ── 初回オープン: draft キーを実キーからコピー ──────────────
    _init_flag = f"_dlg_init_{_dkey}"
    if _init_flag not in ss:
        for _rk, _rv in list(ss.items()):
            if isinstance(_rk, str) and _rk.startswith(f"{_dkey}_") \
                    and not _wi_skip_key(_rk):
                ss[f"_dlg_{_rk}"] = _rv
        ss[f"{_vpfx}trigger"] = ss.get(
            f"{_r_svkey}_trigger",
            bool_cols[0] if bool_cols else "")
        ss[f"{_vpfx}edge"] = ss.get(f"{_r_svkey}_edge", "RISE")
        if _r_graph == "XY":
            ss[f"{_vpfx}xy_xvar"] = ss.get(f"{_r_svkey}_xy_xvar", "")
        ss[_init_flag] = True

    st.caption(
        f"変数: **{_r_sv}** / グラフ: {_r_graph} / #{_r_no}  {_r_dtype}")

    # ── 波形プレビュー ────────────────────────────────────────────
    _cached_sw = ss.get(f"{_vkey}_step_waves_cache", [])
    if _cached_sw:
        _sw_dlg = [(np.array(t), np.array(v)) for t, v in _cached_sw]
        _fig_dlg = go.Figure()
        for _t_sw, _v_sw in _sw_dlg[:60]:
            _fig_dlg.add_trace(go.Scatter(
                x=_t_sw, y=_v_sw, mode="lines",
                line=dict(color="rgba(100,120,200,0.10)", width=1),
                showlegend=False))
        _tc_d = np.linspace(
            float(min(t[0] for t, _ in _sw_dlg)),
            float(max(t[-1] for t, _ in _sw_dlg)), 500)
        _mat_d = np.full((len(_sw_dlg), len(_tc_d)), np.nan)
        for _ji, (_t_sw, _v_sw) in enumerate(_sw_dlg):
            _ix = np.searchsorted(_t_sw, _tc_d).clip(0, len(_t_sw) - 1)
            _in = (_tc_d >= _t_sw[0]) & (_tc_d <= _t_sw[-1])
            _mat_d[_ji, _in] = _v_sw[_ix[_in]]
        _fig_dlg.add_trace(go.Scatter(
            x=_tc_d, y=np.nanmean(_mat_d, axis=0),
            mode="lines", line=dict(color="royalblue", width=2.5), name="平均"))
        _fig_dlg.update_layout(
            height=240, margin=dict(t=10, b=30, l=50, r=10),
            xaxis_title="t [ms]", yaxis_title=_r_sv)
        st.plotly_chart(_fig_dlg, use_container_width=True,
                        key=f"_dlg_chart_{_dkey}")
    else:
        st.info("波形プレビューは波形検査タブで一度データを表示後に利用できます")

    st.divider()

    # ── トリガー / エッジ ─────────────────────────────────────────
    _tc1, _tc2 = st.columns([3, 1])
    with _tc1:
        _trig_opts = bool_cols if bool_cols else [""]
        _cur_trig  = ss.get(f"{_vpfx}trigger", _trig_opts[0] if _trig_opts else "")
        _tidx      = _trig_opts.index(_cur_trig) if _cur_trig in _trig_opts else 0
        st.selectbox("トリガー変数", _trig_opts, index=_tidx, key=f"{_vpfx}trigger")
    with _tc2:
        _cur_edge = ss.get(f"{_vpfx}edge", "RISE")
        _eidx     = ["RISE", "FALL"].index(_cur_edge) if _cur_edge in ["RISE", "FALL"] else 0
        st.radio("エッジ", ["RISE", "FALL"], index=_eidx,
                 horizontal=True, key=f"{_vpfx}edge")

    # ── 名前 / 有効 / 傾向解析 ───────────────────────────────────
    _na1, _na2, _na3 = st.columns([4, 1, 1])
    with _na1:
        st.text_input("名前", value=ss.get(f"{_dpfx}name", ""),
                      key=f"{_dpfx}name", placeholder=f"例: {_r_dtype}①")
    with _na2:
        st.checkbox("有効", value=bool(ss.get(f"{_dpfx}on", False)),
                    key=f"{_dpfx}on")
    with _na3:
        st.checkbox("📈 傾向解析", value=bool(ss.get(f"{_dpfx}trend_on", False)),
                    key=f"{_dpfx}trend_on")

    # ── タイプ固有の設定 ──────────────────────────────────────────
    if _r_graph == "時間軸":
        if _r_dtype == "傾き変化点":
            _pa, _pb, _pc, _pd = st.columns([2, 2, 2, 3])
            with _pa:
                st.markdown("**① 平滑化幅**")
                st.number_input("サンプル数", min_value=1, max_value=50,
                                value=int(ss.get(f"{_dpfx}smooth", 5)), step=1,
                                key=f"{_dpfx}smooth")
            with _pb:
                st.markdown("**n_L（左）**")
                st.number_input("左へ何サンプル", min_value=1, max_value=100,
                                value=int(ss.get(f"{_dpfx}nleft", 3)), step=1,
                                key=f"{_dpfx}nleft")
            with _pc:
                st.markdown("**n_R（右）**")
                st.number_input("右へ何サンプル", min_value=1, max_value=100,
                                value=int(ss.get(f"{_dpfx}nright", 3)), step=1,
                                key=f"{_dpfx}nright")
            with _pd:
                st.markdown("**③ 閾値 |R − L|**")
                st.number_input("閾値", min_value=0.0,
                                value=float(ss.get(f"{_dpfx}thresh", 0.0)),
                                step=0.0001, format="%.4f", key=f"{_dpfx}thresh")
            _dc1, _dc2 = st.columns(2)
            with _dc1:
                st.checkbox("📈 増加方向を検出 (R > L)",
                            value=bool(ss.get(f"{_dpfx}dir_inc", True)),
                            key=f"{_dpfx}dir_inc")
            with _dc2:
                st.checkbox("📉 減少方向を検出 (L > R)",
                            value=bool(ss.get(f"{_dpfx}dir_dec", True)),
                            key=f"{_dpfx}dir_dec")
            _rca, _rcb, _rcc = st.columns([1, 2, 2])
            with _rca:
                st.markdown("**🔍 t 検索範囲 [ms]**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_range", False)),
                            key=f"{_dpfx}use_range")
            _use_r_dlg = bool(ss.get(f"{_dpfx}use_range", False))
            with _rcb:
                st.number_input("開始", value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                step=5.0, key=f"{_dpfx}range_s", disabled=not _use_r_dlg)
            with _rcc:
                st.number_input("終了", value=float(ss.get(f"{_dpfx}range_e", 200.0)),
                                step=5.0, key=f"{_dpfx}range_e", disabled=not _use_r_dlg)
            _vca, _vcb, _vcc = st.columns([1, 2, 2])
            with _vca:
                st.markdown("**🔍 v 検索範囲**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_vrange", False)),
                            key=f"{_dpfx}use_vrange")
            _use_vr_dlg = bool(ss.get(f"{_dpfx}use_vrange", False))
            with _vcb:
                st.number_input("v 下限", value=float(ss.get(f"{_dpfx}vrange_lo", 0.0)),
                                step=0.1, key=f"{_dpfx}vrange_lo", disabled=not _use_vr_dlg)
            with _vcc:
                st.number_input("v 上限", value=float(ss.get(f"{_dpfx}vrange_hi", 1.0)),
                                step=0.1, key=f"{_dpfx}vrange_hi", disabled=not _use_vr_dlg)
            st.number_input("🎯 N番目（0=全て, 1=最初, -1=最後）",
                            value=int(ss.get(f"{_dpfx}nth", 0)), step=1,
                            key=f"{_dpfx}nth")

        elif _r_dtype == "閾値超え検出":
            _thc1, _thc2 = st.columns([2, 2])
            with _thc1:
                st.markdown("**閾値**")
                st.number_input("閾値", value=float(ss.get(f"{_dpfx}tv", 1.0)),
                                step=0.1, key=f"{_dpfx}tv")
            with _thc2:
                st.markdown("**方向**")
                _tdir_opts = ["上昇 ↑", "下降 ↓", "両方"]
                _cur_tdir  = ss.get(f"{_dpfx}tdir", "上昇 ↑")
                _tdir_idx  = (_tdir_opts.index(_cur_tdir)
                              if _cur_tdir in _tdir_opts else 0)
                st.radio("方向", _tdir_opts, index=_tdir_idx,
                         horizontal=True, key=f"{_dpfx}tdir")
            st.number_input("🎯 N番目（0=全て, 1=最初, -1=最後）",
                            value=int(ss.get(f"{_dpfx}nth", 1)), step=1,
                            key=f"{_dpfx}nth")
            _rc1, _rc2, _rc3 = st.columns([1, 2, 2])
            with _rc1:
                st.markdown("**🔍 t 検索範囲 [ms]**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_range", False)),
                            key=f"{_dpfx}use_range")
            _use_r_dlg = bool(ss.get(f"{_dpfx}use_range", False))
            with _rc2:
                st.number_input("開始", value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                step=5.0, key=f"{_dpfx}range_s", disabled=not _use_r_dlg)
            with _rc3:
                st.number_input("終了", value=float(ss.get(f"{_dpfx}range_e", 200.0)),
                                step=5.0, key=f"{_dpfx}range_e", disabled=not _use_r_dlg)
            _vc1, _vc2, _vc3 = st.columns([1, 2, 2])
            with _vc1:
                st.markdown("**🔍 v 検索範囲**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_vrange", False)),
                            key=f"{_dpfx}use_vrange")
            _use_vr_dlg = bool(ss.get(f"{_dpfx}use_vrange", False))
            with _vc2:
                st.number_input("v 下限", value=float(ss.get(f"{_dpfx}vrange_lo", 0.0)),
                                step=0.1, key=f"{_dpfx}vrange_lo", disabled=not _use_vr_dlg)
            with _vc3:
                st.number_input("v 上限", value=float(ss.get(f"{_dpfx}vrange_hi", 1.0)),
                                step=0.1, key=f"{_dpfx}vrange_hi", disabled=not _use_vr_dlg)

        elif _r_dtype in ["最大値点", "最小値点"]:
            _rc1, _rc2, _rc3 = st.columns([1, 2, 2])
            with _rc1:
                st.markdown("**🔍 t 検索範囲 [ms]**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_range", False)),
                            key=f"{_dpfx}use_range")
            _use_r_dlg = bool(ss.get(f"{_dpfx}use_range", False))
            with _rc2:
                st.number_input("開始", value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                step=5.0, key=f"{_dpfx}range_s", disabled=not _use_r_dlg)
            with _rc3:
                st.number_input("終了", value=float(ss.get(f"{_dpfx}range_e", 200.0)),
                                step=5.0, key=f"{_dpfx}range_e", disabled=not _use_r_dlg)
            _vc1, _vc2, _vc3 = st.columns([1, 2, 2])
            with _vc1:
                st.markdown("**🔍 v 検索範囲**")
                st.checkbox("指定する", value=bool(ss.get(f"{_dpfx}use_vrange", False)),
                            key=f"{_dpfx}use_vrange")
            _use_vr_dlg = bool(ss.get(f"{_dpfx}use_vrange", False))
            with _vc2:
                st.number_input("v 下限", value=float(ss.get(f"{_dpfx}vrange_lo", 0.0)),
                                step=0.1, key=f"{_dpfx}vrange_lo", disabled=not _use_vr_dlg)
            with _vc3:
                st.number_input("v 上限", value=float(ss.get(f"{_dpfx}vrange_hi", 1.0)),
                                step=0.1, key=f"{_dpfx}vrange_hi", disabled=not _use_vr_dlg)

        elif _r_dtype == "数式":
            st.caption("参照点を p1t/p1v, p2t/p2v ... の形で数式を記述してください")
            st.text_input("数式", value=ss.get(f"{_dpfx}expr", ""),
                          key=f"{_dpfx}expr", placeholder="例: p1t - p2t")
            _fmc1, _fmc2 = st.columns(2)
            with _fmc1:
                st.number_input("上限 NG（0=無効）",
                                value=float(ss.get(f"{_dpfx}hi_limit", 0.0)),
                                step=0.01, key=f"{_dpfx}hi_limit")
            with _fmc2:
                st.number_input("下限 NG（0=無効）",
                                value=float(ss.get(f"{_dpfx}lo_limit", 0.0)),
                                step=0.01, key=f"{_dpfx}lo_limit")
        else:
            st.info(f"「{_r_dtype}」の詳細設定はメインエリアの ⚙️ ポップオーバーから行ってください")

        # ± Δ 判定（点取得系の場合のみ）
        if _r_dtype in ["傾き変化点", "閾値超え検出", "最大値点", "最小値点"]:
            st.markdown("**✅ OK/NG 判定（基準±Δ）**")
            _pm1, _pm2, _pm3 = st.columns([1.5, 2, 2])
            with _pm1:
                st.checkbox("基準±Δで判定",
                            value=bool(ss.get(f"{_dpfx}pm_on", False)),
                            key=f"{_dpfx}pm_on")
            _pm_on_d = bool(ss.get(f"{_dpfx}pm_on", False))
            with _pm2:
                st.number_input("t 許容差 [ms]（0=無効）", min_value=0.0,
                                value=float(ss.get(f"{_dpfx}pm_dt", 0.0)), step=1.0,
                                key=f"{_dpfx}pm_dt", disabled=not _pm_on_d)
            with _pm3:
                st.number_input("v 許容差（0=無効）", min_value=0.0,
                                value=float(ss.get(f"{_dpfx}pm_dv", 0.0)), step=0.01,
                                key=f"{_dpfx}pm_dv", disabled=not _pm_on_d)
    else:
        # ── XY グラフ設定 ─────────────────────────────────────────
        _xy_xvar_opts = [c for c in num_cols if c != _r_sv]
        if not _xy_xvar_opts:
            st.warning("X軸に使える変数がありません")
        else:
            _xy_xvar_cur = ss.get(f"{_vpfx}xy_xvar",
                                  _xy_xvar_opts[0])
            _xvidx = (_xy_xvar_opts.index(_xy_xvar_cur)
                      if _xy_xvar_cur in _xy_xvar_opts else 0)
            st.selectbox("X軸変数", _xy_xvar_opts, index=_xvidx,
                         key=f"{_vpfx}xy_xvar")

            if _r_dtype == "傾き変化点":
                _xa, _xb, _xc, _xd = st.columns([2, 2, 2, 3])
                with _xa:
                    st.markdown("**① 平滑化幅**")
                    st.number_input("サンプル数", min_value=1, max_value=50,
                                    value=int(ss.get(f"{_dpfx}smooth", 5)),
                                    step=1, key=f"{_dpfx}smooth")
                with _xb:
                    st.markdown("**n_L（左）**")
                    st.number_input("左へ何サンプル", min_value=1, max_value=100,
                                    value=int(ss.get(f"{_dpfx}nleft", 3)),
                                    step=1, key=f"{_dpfx}nleft")
                with _xc:
                    st.markdown("**n_R（右）**")
                    st.number_input("右へ何サンプル", min_value=1, max_value=100,
                                    value=int(ss.get(f"{_dpfx}nright", 3)),
                                    step=1, key=f"{_dpfx}nright")
                with _xd:
                    st.markdown("**③ 閾値 |R − L|**")
                    st.number_input("閾値", min_value=0.0,
                                    value=float(ss.get(f"{_dpfx}thresh", 0.0)),
                                    step=0.0001, format="%.4f",
                                    key=f"{_dpfx}thresh")
                _xdc1, _xdc2 = st.columns(2)
                with _xdc1:
                    st.checkbox("📈 増加方向を検出 (R > L)",
                                value=bool(ss.get(f"{_dpfx}dir_inc", True)),
                                key=f"{_dpfx}dir_inc")
                with _xdc2:
                    st.checkbox("📉 減少方向を検出 (L > R)",
                                value=bool(ss.get(f"{_dpfx}dir_dec", True)),
                                key=f"{_dpfx}dir_dec")
                _xrca, _xrcb, _xrcc = st.columns([1, 2, 2])
                with _xrca:
                    st.markdown("**🔍 X 検索範囲**")
                    st.checkbox("指定する",
                                value=bool(ss.get(f"{_dpfx}use_range", False)),
                                key=f"{_dpfx}use_range")
                _xuse_r = bool(ss.get(f"{_dpfx}use_range", False))
                with _xrcb:
                    st.number_input("X 開始",
                                    value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_s",
                                    disabled=not _xuse_r)
                with _xrcc:
                    st.number_input("X 終了",
                                    value=float(ss.get(f"{_dpfx}range_e", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_e",
                                    disabled=not _xuse_r)

            elif _r_dtype == "閾値超え検出":
                _xt1, _xt2, _xt3 = st.columns([2, 2, 2])
                with _xt1:
                    st.number_input("Y 閾値",
                                    value=float(ss.get(f"{_dpfx}tv", 0.0)),
                                    step=0.01, key=f"{_dpfx}tv")
                with _xt2:
                    _tdir_c = ss.get(f"{_dpfx}tdir", "RISE")
                    _tdir_i = (["RISE", "FALL"].index(_tdir_c)
                               if _tdir_c in ["RISE", "FALL"] else 0)
                    st.radio("方向", ["RISE", "FALL"], index=_tdir_i,
                             horizontal=True, key=f"{_dpfx}tdir")
                with _xt3:
                    st.number_input("N番目を検出", min_value=1,
                                    value=int(ss.get(f"{_dpfx}nth", 1)),
                                    step=1, key=f"{_dpfx}nth")
                _xrca, _xrcb, _xrcc = st.columns([1, 2, 2])
                with _xrca:
                    st.markdown("**🔍 X 検索範囲**")
                    st.checkbox("指定する",
                                value=bool(ss.get(f"{_dpfx}use_range", False)),
                                key=f"{_dpfx}use_range")
                _xuse_r = bool(ss.get(f"{_dpfx}use_range", False))
                with _xrcb:
                    st.number_input("X 開始",
                                    value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_s",
                                    disabled=not _xuse_r)
                with _xrcc:
                    st.number_input("X 終了",
                                    value=float(ss.get(f"{_dpfx}range_e", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_e",
                                    disabled=not _xuse_r)

            elif _r_dtype in ["Y最大値点", "Y最小値点"]:
                _xrca, _xrcb, _xrcc = st.columns([1, 2, 2])
                with _xrca:
                    st.markdown("**🔍 X 検索範囲**")
                    st.checkbox("指定する",
                                value=bool(ss.get(f"{_dpfx}use_range", False)),
                                key=f"{_dpfx}use_range")
                _xuse_r = bool(ss.get(f"{_dpfx}use_range", False))
                with _xrcb:
                    st.number_input("X 開始",
                                    value=float(ss.get(f"{_dpfx}range_s", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_s",
                                    disabled=not _xuse_r)
                with _xrcc:
                    st.number_input("X 終了",
                                    value=float(ss.get(f"{_dpfx}range_e", 0.0)),
                                    step=0.5, key=f"{_dpfx}range_e",
                                    disabled=not _xuse_r)

            elif _r_dtype == "数式":
                st.text_input("数式 (例: p1v - p2v)",
                              value=ss.get(f"{_dpfx}expr", ""),
                              key=f"{_dpfx}expr",
                              placeholder="例: p1v - p2v")
                _fxc1, _fxc2 = st.columns(2)
                with _fxc1:
                    st.number_input("上限 NG（0=無効）",
                                    value=float(ss.get(f"{_dpfx}hi_limit", 0.0)),
                                    step=0.01, key=f"{_dpfx}hi_limit")
                with _fxc2:
                    st.number_input("下限 NG（0=無効）",
                                    value=float(ss.get(f"{_dpfx}lo_limit", 0.0)),
                                    step=0.01, key=f"{_dpfx}lo_limit")

            else:
                st.info(
                    f"「{_r_dtype}」はメインエリアの ⚙️ ポップオーバーから設定してください")

            # ±Δ判定（XY 点取得系）
            if _r_dtype in ["傾き変化点", "閾値超え検出", "Y最大値点", "Y最小値点"]:
                st.markdown("**✅ OK/NG 判定（基準±Δ）**")
                _xpm1, _xpm2, _xpm3 = st.columns([1.5, 2, 2])
                with _xpm1:
                    st.checkbox("基準±Δで判定",
                                value=bool(ss.get(f"{_dpfx}pm_on", False)),
                                key=f"{_dpfx}pm_on")
                _xpm_on = bool(ss.get(f"{_dpfx}pm_on", False))
                with _xpm2:
                    st.number_input("X 許容差（0=無効）", min_value=0.0,
                                    value=float(ss.get(f"{_dpfx}pm_dt", 0.0)),
                                    step=0.5, key=f"{_dpfx}pm_dt",
                                    disabled=not _xpm_on)
                with _xpm3:
                    st.number_input("Y 許容差（0=無効）", min_value=0.0,
                                    value=float(ss.get(f"{_dpfx}pm_dv", 0.0)),
                                    step=0.01, key=f"{_dpfx}pm_dv",
                                    disabled=not _xpm_on)

    st.divider()

    # ── 保存 / キャンセル ─────────────────────────────────────────
    _btn1, _btn2 = st.columns(2)
    with _btn1:
        if st.button("💾 保存", use_container_width=True, type="primary",
                     key=f"_dlg_save_{_dkey}"):
            ss[f"{_r_svkey}_trigger"] = ss.get(f"{_vpfx}trigger", "")
            ss[f"{_r_svkey}_edge"]    = ss.get(f"{_vpfx}edge", "RISE")
            if _r_graph == "XY":
                _xy_xvar_val = ss.get(f"{_vpfx}xy_xvar", "")
                if _xy_xvar_val:
                    ss[f"{_r_svkey}_xy_xvar"] = _xy_xvar_val
            _dpfx_len = len(_dpfx)
            for _dk in list(ss.keys()):
                if isinstance(_dk, str) and _dk.startswith(_dpfx):
                    _real_k = f"{_dkey}_{_dk[_dpfx_len:]}"
                    if not _wi_skip_key(_real_k):
                        ss[_real_k] = ss[_dk]
            ss.pop("wi_edit_target", None)
            ss.pop(_init_flag, None)
            st.rerun()
    with _btn2:
        if st.button("✖ キャンセル", use_container_width=True,
                     key=f"_dlg_cancel_{_dkey}"):
            ss.pop("wi_edit_target", None)
            ss.pop(_init_flag, None)
            st.rerun()


@st.dialog("検出点を追加", width="large")
def _wi_add_dialog():
    ss = st.session_state
    st.caption("追加する検出点の設定を選択してください")

    _DET_TYPES_ADD_T  = ["傾き変化点", "閾値超え検出", "最大値点", "最小値点",
                         "上下判定比較", "最大値判定", "最小値判定", "検出点比較", "数式"]
    _DET_TYPES_ADD_XY = ["傾き変化点", "閾値超え検出", "Y最大値点", "Y最小値点",
                         "上下判定比較", "Y最大値判定", "Y最小値判定", "検出点比較", "数式"]

    _ac1, _ac2, _ac3 = st.columns([3, 1.5, 2])
    with _ac1:
        _add_var = st.selectbox("波形変数", num_cols, key="_wi_add_var")
    with _ac2:
        _add_graph = st.radio("グラフ", ["時間軸", "XY"], key="_wi_add_graph")
    with _ac3:
        _det_types_a = (_DET_TYPES_ADD_T if _add_graph == "時間軸"
                        else _DET_TYPES_ADD_XY)
        _add_dtype = st.selectbox("検出点タイプ", _det_types_a, key="_wi_add_dtype")

    st.divider()
    _ab1, _ab2 = st.columns(2)
    with _ab1:
        if st.button("波形で設定 →", use_container_width=True,
                     type="primary", key="_wi_add_confirm"):
            _svkey = f"wvol___global_{_add_var}"
            if _add_graph == "時間軸":
                _list_k = f"{_svkey}_t_det_list"
                _cnt_k  = f"{_svkey}_t_det_cnt"
                _id_pfx = "td"
            else:
                _list_k = f"{_svkey}_xy_det_list"
                _cnt_k  = f"{_svkey}_xy_det_cnt"
                _id_pfx = "xyd"
            _cnt     = ss.get(_cnt_k, 0)
            _new_id  = f"{_id_pfx}{_cnt}"
            _det_lst = list(ss.get(_list_k, []))
            _det_lst.append({"id": _new_id, "type": _add_dtype})
            ss[_list_k] = _det_lst
            ss[_cnt_k]  = _cnt + 1
            ss["wi_edit_target"] = {
                "_svkey": _svkey, "_did": _new_id,
                "_sv": _add_var, "_graph": _add_graph, "_dtype": _add_dtype,
                "No": len(_det_lst), "変数": _add_var,
                "グラフ": _add_graph, "タイプ": _add_dtype,
                "名前": "", "有効": "−", "傾向解析": "−",
            }
            ss["wi_transition_to_edit"] = True
            st.rerun()
    with _ab2:
        if st.button("キャンセル", use_container_width=True, key="_wi_add_cancel"):
            st.rerun()
