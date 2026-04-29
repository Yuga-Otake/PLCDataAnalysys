"""
app.py - APB タイミング解析ツール v6
・テキスト検索＋予測候補からステップ変数を追加
・単一変数モード / 開始-終了範囲モード
・ガントクリックでステップ詳細連動
・工程ごと異常比較インライン
"""
import os, glob, json, hashlib, urllib.parse
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Xbar-R / Xbar-S 管理図定数テーブル ─────────────────────────────
# n: (A2, D3, D4, c4, B3, B4)
#   A2, D3, D4 → Xbar-R 用
#   c4, B3, B4 → Xbar-S 用（n>10 推奨）
_SPC_CONSTS = {
    2:  (1.880, 0,     3.267, 0.7979, 0,     3.267),
    3:  (1.023, 0,     2.574, 0.8862, 0,     2.568),
    4:  (0.729, 0,     2.282, 0.9213, 0,     2.266),
    5:  (0.577, 0,     2.114, 0.9400, 0,     2.089),
    6:  (0.483, 0,     2.004, 0.9515, 0.030, 1.970),
    7:  (0.419, 0.076, 1.924, 0.9594, 0.118, 1.882),
    8:  (0.373, 0.136, 1.864, 0.9650, 0.185, 1.815),
    9:  (0.337, 0.184, 1.816, 0.9693, 0.239, 1.761),
    10: (0.308, 0.223, 1.777, 0.9727, 0.284, 1.716),
    11: (0.285, 0.256, 1.744, 0.9754, 0.321, 1.679),
    12: (0.266, 0.284, 1.717, 0.9776, 0.354, 1.646),
    13: (0.249, 0.308, 1.692, 0.9794, 0.382, 1.618),
    14: (0.235, 0.329, 1.671, 0.9810, 0.406, 1.594),
    15: (0.223, 0.347, 1.653, 0.9823, 0.428, 1.572),
    16: (0.212, 0.363, 1.637, 0.9835, 0.448, 1.552),
    17: (0.203, 0.378, 1.622, 0.9845, 0.466, 1.534),
    18: (0.194, 0.391, 1.608, 0.9854, 0.482, 1.518),
    19: (0.187, 0.403, 1.597, 0.9862, 0.497, 1.503),
    20: (0.180, 0.415, 1.585, 0.9869, 0.510, 1.490),
    21: (0.173, 0.425, 1.575, 0.9876, 0.523, 1.477),
    22: (0.167, 0.434, 1.566, 0.9882, 0.534, 1.466),
    23: (0.162, 0.443, 1.557, 0.9887, 0.545, 1.455),
    24: (0.157, 0.452, 1.548, 0.9892, 0.555, 1.445),
    25: (0.153, 0.459, 1.541, 0.9896, 0.565, 1.435),
}

def _spc_consts(n: int):
    """サブグループサイズ n に最も近い SPC 定数 (A2,D3,D4,c4,B3,B4) を返す"""
    n = max(2, int(round(n)))
    if n in _SPC_CONSTS:
        return _SPC_CONSTS[n]
    keys = sorted(_SPC_CONSTS.keys())
    return _SPC_CONSTS[min(keys, key=lambda k: abs(k - n))]

from analyzer import (
    load_csv, detect_bool_columns, detect_cycles,
    analyze_cycles_v2, calc_variable_periods,
    get_cycle_waveforms, calc_sturges_bins, calc_statistics, normalize_bool_series,
)
from comparator import compare_normal_abnormal, calc_diff_ranking, detect_anomalous_variables

st.set_page_config(page_title="APB タイミング解析", page_icon="🏭", layout="wide")

# ── wi_saved_setups 永続化ファイルパス ──────────────────────────────
_WI_SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wi_saved_setups.json")
_WI_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "wi_current_config.json"
)

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ステップカラーパレット
STEP_COLORS = [
    "#4472C4", "#ED7D31", "#70AD47", "#FF0000",
    "#7030A0", "#00B0F0", "#FFD966", "#FF6B6B",
    "#C00000", "#375623", "#833C00", "#1F3864",
]

# 比較モード用カラーパレット（CSV ごとに異なる色）
_CMP_PALETTE = [
    "#3b82f6",  # blue
    "#f59e0b",  # amber
    "#10b981",  # emerald
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#84cc16",  # lime
]


# ═══════════════════════════════════════════════════════════════
# ヘルパー関数
# ═══════════════════════════════════════════════════════════════

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
    snap: dict = {
        "wi_trigger": ss.get("wi_trigger", ""),
        "wi_edge":    ss.get("wi_edge", "RISE"),
    }
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


def pk(pname: str, suffix: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in pname)
    return f"P_{safe}__{suffix}"


def _default_color(idx: int) -> str:
    return STEP_COLORS[idx % len(STEP_COLORS)]


def init_proc_widgets(pname: str, pinfo: dict, bool_cols: list):
    def _set(key, val):
        if key not in st.session_state:
            st.session_state[key] = val

    trigger = pinfo.get("trigger_col", bool_cols[0] if bool_cols else "")
    if trigger not in bool_cols and bool_cols:
        trigger = bool_cols[0]
    _set(pk(pname, "trigger"), trigger)
    _set(pk(pname, "edge"),    pinfo.get("edge", "RISE"))
    _set(pk(pname, "takt"),    int(pinfo.get("takt_target_ms", 0)))

    # steps_list 初期化（旧形式 steps → 新形式に変換）
    if pk(pname, "steps_list") not in st.session_state:
        steps_list = []
        for i, s in enumerate(pinfo.get("steps", [])):
            v = s["variable"]
            steps_list.append({
                "name":     s.get("label", v),
                "color":    s.get("color", _default_color(i)),
                "mode":     "single",
                "variable": v,
                "edge":     "RISE",
            })
        st.session_state[pk(pname, "steps_list")] = steps_list


def rename_process(old_name: str, new_name: str):
    """工程名を変更し、関連 session_state キーをすべて移行する"""
    procs = st.session_state["processes"]
    new_procs = {(new_name if k == old_name else k): v for k, v in procs.items()}
    st.session_state["processes"] = new_procs

    old_safe = "".join(c if c.isalnum() else "_" for c in old_name)
    new_safe = "".join(c if c.isalnum() else "_" for c in new_name)
    old_pfx, new_pfx = f"P_{old_safe}__", f"P_{new_safe}__"
    for key in [k for k in list(st.session_state.keys()) if k.startswith(old_pfx)]:
        st.session_state[new_pfx + key[len(old_pfx):]] = st.session_state.pop(key)

    if st.session_state.get("_expand_new") == old_name:
        st.session_state["_expand_new"] = new_name
    ng_old, ng_new = f"_ng_df_{old_name}", f"_ng_df_{new_name}"
    if ng_old in st.session_state:
        st.session_state[ng_new] = st.session_state.pop(ng_old)


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


def steps_all_vars(steps_list: list, bool_cols: list) -> list:
    """steps_list から使用している変数名リストを返す（bool_cols の順序で）"""
    used = set()
    for s in steps_list:
        if s.get("mode", "single") == "single":
            if s.get("variable"):
                used.add(s["variable"])
        else:
            if s.get("start_var"): used.add(s["start_var"])
            if s.get("end_var"):   used.add(s["end_var"])
    return [v for v in bool_cols if v in used]


# ═══════════════════════════════════════════════════════════════
# キャッシュ付き解析
# ═══════════════════════════════════════════════════════════════

@st.cache_data
def cached_load_sample(path: str):
    df = load_csv(path)
    return df, detect_bool_columns(df)

@st.cache_data
def cached_detect_cycles(df: pd.DataFrame, trigger_col: str, edge: str) -> list:
    return list(detect_cycles(df, trigger_col, edge))

@st.cache_data
def cached_analyze_v2(df: pd.DataFrame, trigger_col: str, edge: str,
                       steps_json: str) -> pd.DataFrame:
    steps = json.loads(steps_json)
    if not steps:
        return pd.DataFrame()
    return analyze_cycles_v2(df, trigger_col, edge, steps)

@st.cache_data
def cached_variable_periods(df: pd.DataFrame, bool_cols_t: tuple) -> dict:
    return calc_variable_periods(df, list(bool_cols_t))

@st.cache_data
def cached_waveforms(df: pd.DataFrame, trigger_col: str, edge: str,
                     target_vars: tuple) -> list:
    idx = pd.Index(cached_detect_cycles(df, trigger_col, edge))
    return get_cycle_waveforms(df, idx, list(target_vars))

@st.cache_data
def cached_build_gantt(result_df: pd.DataFrame, steps_json: str, takt_target: int):
    """build_gantt_v2 のキャッシュラッパー（result_df + steps_json が同一なら再計算しない）"""
    import json as _json
    steps_list = _json.loads(steps_json)
    return build_gantt_v2(result_df, steps_list, takt_target)


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


# ═══════════════════════════════════════════════════════════════
# ガントチャート（v2: single/range 混在対応）
# ═══════════════════════════════════════════════════════════════

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


def build_gantt_v2(result_df: pd.DataFrame, steps_list: list, takt_target: int):
    step_stats, prev_mean = [], 0.0

    for step in steps_list:
        name  = step.get("name", "")
        color = step.get("color", "#4472C4")
        mode  = step.get("mode", "single")

        if mode == "single":
            dcol = f"{name}_遅れ[ms]"
            if dcol not in result_df.columns:
                continue
            delays = result_df[dcol].dropna().values
            if len(delays) == 0:
                continue
            mean_d = float(np.mean(delays))
            std_d  = float(np.std(delays))
            min_d, max_d = float(np.min(delays)), float(np.max(delays))
            step_stats.append(dict(
                name=name, color=color, mode="single",
                start=0.0, mean=mean_d,
                min=min_d, max=max_d,
                abs_mean=mean_d, abs_std=std_d, abs_min=min_d, abs_max=max_d,
            ))

        elif mode in ("range", "numeric", "on_period"):
            sc = f"{name}_start[ms]"
            dc = f"{name}_dur[ms]"
            if sc not in result_df.columns or dc not in result_df.columns:
                continue
            starts = result_df[sc].dropna().values
            durs   = result_df[dc].dropna().values
            if len(durs) == 0:
                continue
            mean_s = float(np.mean(starts)) if len(starts) > 0 else prev_mean
            mean_d = float(np.mean(durs))
            std_d  = float(np.std(durs))
            min_d, max_d = float(np.min(durs)), float(np.max(durs))
            step_stats.append(dict(
                name=name, color=color, mode=mode,
                start=mean_s, mean=mean_d,
                min=min_d, max=max_d,
                abs_mean=mean_s + mean_d, abs_std=std_d,
                abs_min=mean_s + min_d, abs_max=mean_s + max_d,
                abs_start=mean_s,
            ))
            prev_mean = mean_s + mean_d

    if not step_stats:
        return None, []

    total = max((s["abs_mean"] for s in step_stats), default=1.0) or 1.0
    fig   = go.Figure()

    for s in step_stats:
        pct      = round(s["abs_mean"] / total * 100, 1)
        smode    = s["mode"]  # "single" / "range" / "numeric" / "on_period"
        is_range = smode in ("range", "numeric", "on_period")
        pat      = {"range": "/", "numeric": "x", "on_period": "\\", "single": ""}.get(smode, "")
        if smode == "range":
            ht = (
                "<b>%{y}</b> [範囲]<br>"
                f"開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "長さ（平均）: %{customdata[1]:.1f} ms<br>"
                "長さばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        elif smode == "on_period":
            ht = (
                "<b>%{y}</b> [ON期間]<br>"
                f"ON開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "ON継続（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        elif smode == "numeric":
            ht = (
                "<b>%{y}</b> [数値条件]<br>"
                f"条件成立開始（平均）: {s.get('abs_start', s['start']):.1f} ms<br>"
                "継続時間（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<br>"
                "タクト比率: %{customdata[4]:.1f}%<extra></extra>"
            )
        else:
            ht = (
                "<b>%{y}</b><br>"
                "サイクル開始から（平均）: %{customdata[1]:.1f} ms<br>"
                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms<extra></extra>"
            )

        fig.add_trace(go.Bar(
            name=s["name"], y=[s["name"]], x=[s["mean"]], base=[s["start"]],
            orientation="h", width=0.5, marker_color=s["color"],
            marker_pattern_shape=pat,
            customdata=[[s["start"], s["mean"], s["min"], s["max"], pct]],
            hovertemplate=ht,
            text=[f"{s['mean']:.0f}ms"] if not is_range else [f"{pct:.0f}%"],
            textposition="outside",
            textfont=dict(size=11, color=s["color"]),
            constraintext="none",
        ))
        bw = s["max"] - s["min"]
        if bw > 0:
            fig.add_trace(go.Bar(
                y=[s["name"]], x=[bw], base=[s["start"] + s["min"]],
                orientation="h", width=0.5, marker_color=s["color"],
                opacity=0.2, showlegend=False, hoverinfo="skip",
            ))

    if takt_target > 0:
        fig.add_vline(x=takt_target, line_dash="dash", line_color="red",
                      annotation_text=f"目標 {takt_target}ms")
    fig.update_layout(
        barmode="overlay",
        xaxis_title="サイクル開始からの経過時間 [ms]",
        yaxis=dict(autorange="reversed"),
        height=max(240, len(step_stats) * 60 + 100),
        margin=dict(l=0, r=50, t=12, b=36),
        showlegend=False, plot_bgcolor="white", dragmode="select",
    )
    return fig, step_stats


# ═══════════════════════════════════════════════════════════════
# B-1: トレンドチャート（サイクルごとの推移）
# ═══════════════════════════════════════════════════════════════

def build_trend_chart(result_df: pd.DataFrame, step_stats: list, takt_target: int):
    """サイクルごとの遅れ時間トレンドチャートを生成する"""
    if result_df is None or len(result_df) == 0 or not step_stats:
        return None
    cycle_col = "サイクル#"
    if cycle_col not in result_df.columns:
        return None

    x = result_df[cycle_col].values
    fig = go.Figure()

    for s in step_stats:
        name  = s["name"]
        color = s["color"]
        mode  = s.get("mode", "single")
        dcol  = f"{name}_遅れ[ms]" if mode == "single" else f"{name}_dur[ms]"
        if dcol not in result_df.columns:
            continue

        y = result_df[dcol].values.astype(float)
        mean_y = float(np.nanmean(y))
        std_y  = float(np.nanstd(y))

        # 散布点
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color=color, opacity=0.55),
            name=name, legendgroup=name,
        ))

        # 5サイクル移動平均
        ma = pd.Series(y).rolling(5, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=x, y=ma, mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} 移動平均", legendgroup=name, showlegend=False,
        ))

        # 外れ値マーカー（|val - mean| > 2σ）
        if std_y > 0:
            outlier_mask = np.abs(y - mean_y) > 2 * std_y
            if np.any(outlier_mask):
                fig.add_trace(go.Scatter(
                    x=x[outlier_mask], y=y[outlier_mask], mode="markers",
                    marker=dict(size=11, color="red", symbol="circle-open",
                                line=dict(width=2, color="red")),
                    name=f"{name} 外れ値", legendgroup=name, showlegend=False,
                ))

            # mean+3σ ライン
            fig.add_hline(
                y=mean_y + 3 * std_y,
                line_dash="dot", line_color=color, line_width=1,
                annotation_text=f"{name} 3σ上限",
                annotation_font_size=9,
            )

    if takt_target > 0:
        fig.add_hline(y=takt_target, line_dash="dash", line_color="red", line_width=1.5,
                      annotation_text=f"タクト目標 {takt_target}ms")

    fig.update_layout(
        xaxis_title="サイクル番号",
        yaxis_title="時間 [ms]",
        height=400,
        margin=dict(t=16, b=36),
        showlegend=True,
        plot_bgcolor="white",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# B-2: IQR外れ値検出
# ═══════════════════════════════════════════════════════════════

def detect_outliers_iqr(result_df: pd.DataFrame, step_stats: list) -> list:
    """IQR法で各ステップの外れ値サイクルを検出する"""
    if result_df is None or len(result_df) == 0:
        return []
    cycle_col = "サイクル#"
    results = []
    for s in step_stats:
        name = s["name"]
        mode = s.get("mode", "single")
        dcol = f"{name}_遅れ[ms]" if mode == "single" else f"{name}_dur[ms]"
        if dcol not in result_df.columns:
            continue
        col = result_df[dcol].dropna()
        if len(col) < 4:
            continue
        Q1  = col.quantile(0.25)
        Q3  = col.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lo = Q1 - 1.5 * IQR
        hi = Q3 + 1.5 * IQR
        mask = (col < lo) | (col > hi)
        outlier_rows = result_df.loc[col[mask].index]
        if cycle_col in outlier_rows.columns:
            cycles = [int(c) for c in outlier_rows[cycle_col].tolist()]
        else:
            cycles = [int(i) for i in outlier_rows.index.tolist()]
        if cycles:
            results.append({
                "name":   name,
                "color":  s["color"],
                "cycles": cycles,
                "hi":     round(hi, 1),
                "lo":     round(lo, 1),
            })
    return results


# ═══════════════════════════════════════════════════════════════
# 波形監視ヘルパー（UIなし・再利用可能）
# ═══════════════════════════════════════════════════════════════

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


def _compute_wi_det_stats_for_csv(df, trigger_col, edge, var_list, ss):
    """
    CSV 1 ファイル分の波形検査検出点を解析し、各検出点の統計 (平均・σ) を返す。
    df          : DataFrame (1 CSV)
    trigger_col : トリガー列名 (wi_trigger)
    edge        : "RISE" | "FALL"
    var_list    : アナログ変数名リスト
    ss          : st.session_state
    Returns     : dict[det_key -> {label, color, t_mean, t_std, v_mean, v_std, n}]
    """
    _POINT_TYPES = ["傾き変化点", "閾値超え検出", "最大値点", "最小値点"]
    _DET_COLORS  = ["darkorange", "deeppink", "limegreen", "dodgerblue",
                    "gold", "orchid", "coral", "steelblue"]
    stats = {}
    if not trigger_col or trigger_col not in df.columns:
        return stats
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
        try:
            _waveforms = cached_waveforms(df, trigger_col, edge, (var,))
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
        try:
            _wf_y = cached_waveforms(df, trigger_col, edge, (var,))
            _wf_x = cached_waveforms(df, trigger_col, edge, (_xvar,))
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
        try:
            _wf_fm = cached_waveforms(df, trigger_col, edge, (var,))
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


# ═══════════════════════════════════════════════════════════════
# 波形監視オーバーレイ
# ═══════════════════════════════════════════════════════════════

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

    # 全サイクル波形取得
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
            for i in range(n_cyc):
                cyc        = waveforms[i]
                step_off   = float(start_offsets[i]) if not np.isnan(start_offsets[i]) else None
                if step_off is None or var not in cyc.columns:
                    continue
                t_abs  = cyc["time_offset_ms"].values          # cycle-relative [ms]
                t_step = t_abs - step_off                       # step-relative [ms]
                v_arr  = cyc[var].values.astype(np.float64)

                # 表示ウィンドウでクリップ
                mask = (t_step >= -win_pre) & (t_step <= win_post)
                if mask.sum() < 2:
                    continue
                step_waves.append((t_step[mask], v_arr[mask]))

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


# ═══════════════════════════════════════════════════════════════
# ステップ詳細
# ═══════════════════════════════════════════════════════════════

def render_step_detail(df: pd.DataFrame, trigger_col: str, edge: str,
                       step_stat: dict, step: dict, pname: str,
                       result_df: pd.DataFrame):
    """タイミング解析のみ表示（波形監視は🔍波形検査タブへ移動済み）"""
    mode = step.get("mode", "single")
    if mode == "single":
        _render_single_detail(df, trigger_col, edge, step_stat, step, pname, result_df)
    elif mode == "numeric":
        _render_numeric_detail(df, trigger_col, edge, step_stat, step, pname, result_df)
    else:
        _render_range_detail(df, trigger_col, edge, step_stat, step, pname, result_df)


def _render_single_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    var       = step.get("variable", "")
    name      = step_stat["name"]
    delay_col = f"{name}_遅れ[ms]"
    delays    = (result_df[delay_col].dropna().values
                 if delay_col in result_df.columns else np.array([]))

    if len(delays) == 0:
        st.warning(f"{var}: データが取得できませんでした")
        return

    mean_d, std_d = float(np.mean(delays)), float(np.std(delays))
    sig3 = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}"

    # 基準値チェック
    _baseline    = st.session_state.get(pk(pname, "baseline"), {})
    _bl_entry    = _baseline.get(name, {})
    _bl_ref      = _bl_entry.get("ref_ms")
    _bl_std      = _bl_entry.get("std_ms", 0.0)
    _delta_mode  = _bl_ref is not None
    delays_plot  = (delays - _bl_ref) if _delta_mode else delays

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        if _delta_mode:
            st.caption(f"基準値: **{_bl_ref:.1f} ms**")
        else:
            st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        _bkey  = f"{pname}_{name}_h"
        n_bins = calc_nice_bins(delays_plot, _bkey)
        _vmin_h = float(delays_plot.min()); _vmax_h = float(delays_plot.max())
        _bsz_h  = (_vmax_h - _vmin_h) / n_bins if n_bins > 0 and _vmax_h > _vmin_h else 1.0
        _xbins_h = dict(start=_vmin_h, end=_vmax_h + _bsz_h, size=_bsz_h)

        # ── 比較モード: 各CSV の分布を重ね合わせ ──────────────────
        _cmp_entries_h = st.session_state.get(f"_cmp_entries_{pname}", [])
        _in_compare_h  = (
            st.session_state.get("compare_mode", False)
            and len(_cmp_entries_h) >= 1
        )
        if _in_compare_h:
            st.markdown("**ヒストグラム（比較モード: CSV別重ね合わせ）**")
            fig_h = go.Figure()
            _stat_rows = []
            for _ci_h, _ce_h in enumerate(_cmp_entries_h):
                try:
                    _ce_res_h = cached_analyze_v2(
                        _ce_h["df"], trigger_col, edge,
                        json.dumps(st.session_state.get(pk(pname, "steps_list"), []))
                    )
                    if delay_col not in _ce_res_h.columns:
                        continue
                    _ce_dl_h = _ce_res_h[delay_col].dropna().values
                    if len(_ce_dl_h) == 0:
                        continue
                    # 共通レンジを拡張してビン幅を揃える
                    _vmin_h = min(_vmin_h, float(_ce_dl_h.min()))
                    _vmax_h = max(_vmax_h, float(_ce_dl_h.max()))
                    _ce_plot_h = (_ce_dl_h - _bl_ref) if _delta_mode else _ce_dl_h
                    _stat_rows.append({
                        "CSV":   _ce_h["label"],
                        "N":     len(_ce_dl_h),
                        "平均[ms]": round(float(np.mean(_ce_dl_h)), 2),
                        "σ[ms]": round(float(np.std(_ce_dl_h)),  2),
                    })
                except Exception:
                    continue
                _bsz_h   = (_vmax_h - _vmin_h) / n_bins if n_bins > 0 and _vmax_h > _vmin_h else 1.0
                _xbins_h = dict(start=_vmin_h, end=_vmax_h + _bsz_h, size=_bsz_h)
                # 16進カラーから rgba に変換して透明度を付与
                _hex = _ce_h["color"].lstrip("#")
                _r, _g, _b = int(_hex[0:2], 16), int(_hex[2:4], 16), int(_hex[4:6], 16)
                fig_h.add_trace(go.Histogram(
                    x=_ce_plot_h, xbins=_xbins_h,
                    name=_ce_h["label"],
                    marker_color=f"rgba({_r},{_g},{_b},0.72)", opacity=0.85,
                ))
            # 共通ビン幅に更新（全トレースに適用）
            _bsz_h   = (_vmax_h - _vmin_h) / n_bins if n_bins > 0 and _vmax_h > _vmin_h else 1.0
            _xbins_h = dict(start=_vmin_h, end=_vmax_h + _bsz_h, size=_bsz_h)
            for _tr in fig_h.data:
                _tr.xbins = _xbins_h
            if _delta_mode:
                fig_h.add_vline(x=0, line_color="#b45309", line_width=2,
                                annotation_text=f"基準 {_bl_ref:.1f}ms",
                                annotation_font_color="#b45309")
                xaxis_title = "基準値からのずれ [ms]"
            else:
                fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                                annotation_text=f"3σ {sig3:.1f}ms")
                if _bl_ref is not None:
                    fig_h.add_vline(x=_bl_ref, line_color="#b45309", line_width=2,
                                    annotation_text=f"基準 {_bl_ref:.1f}ms",
                                    annotation_font_color="#b45309")
                xaxis_title = "遅れ時間 [ms]"
            fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                                 barmode="overlay", height=280, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            if _stat_rows:
                st.dataframe(pd.DataFrame(_stat_rows), hide_index=True, width="stretch")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                      help="ヒストグラムのビン数を手動調整")
        else:
            # ── 通常モード ────────────────────────────────────────
            if _delta_mode:
                st.markdown("**差分ヒストグラム（基準値=0）**")
            else:
                st.markdown("**ヒストグラム**")

            # 基準CSV オーバーレイ用データを取得
            _ref_plot_h = None
            _ref_key_h  = st.session_state.get("ref_csv_key", "")
            _act_key_h  = st.session_state.get("active_csv", "")
            _store_h    = st.session_state.get("csv_store", {})
            if _ref_key_h and _ref_key_h != _act_key_h and _ref_key_h in _store_h:
                try:
                    _ref_steps_h = st.session_state.get(pk(pname, "steps_list"), [])
                    _ref_res_h   = cached_analyze_v2(
                        _store_h[_ref_key_h]["df"], trigger_col, edge,
                        json.dumps(_ref_steps_h)
                    )
                    if delay_col in _ref_res_h.columns:
                        _rv_h = _ref_res_h[delay_col].dropna().values
                        if len(_rv_h) > 0:
                            _ref_plot_h = (_rv_h - _bl_ref) if _delta_mode else _rv_h
                except Exception:
                    pass

            # ref データが取得できた場合は範囲を統合してビン幅を揃える
            if _ref_plot_h is not None:
                _vmin_h = min(_vmin_h, float(_ref_plot_h.min()))
                _vmax_h = max(_vmax_h, float(_ref_plot_h.max()))
                _bsz_h  = (_vmax_h - _vmin_h) / n_bins if n_bins > 0 and _vmax_h > _vmin_h else 1.0
                _xbins_h = dict(start=_vmin_h, end=_vmax_h + _bsz_h, size=_bsz_h)

            fig_h  = go.Figure()
            # 基準CSV 分布を最初のトレースとして追加（背面に描画）
            if _ref_plot_h is not None:
                fig_h.add_trace(go.Histogram(
                    x=_ref_plot_h,
                    xbins=_xbins_h,
                    name="基準データ分布",
                    marker_color="rgba(245,158,11,0.52)", opacity=0.88,
                ))

            if threshold > 0 and not _delta_mode:
                below = delays_plot[delays_plot <= threshold]
                above = delays_plot[delays_plot >  threshold]
                if len(below):
                    fig_h.add_trace(go.Histogram(x=below, xbins=_xbins_h, name="閾値以内",
                                                 marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                if len(above):
                    fig_h.add_trace(go.Histogram(x=above, xbins=_xbins_h, name="閾値超過",
                                                 marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                fig_h.add_vline(x=threshold, line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"閾値 {threshold}ms")
            elif _delta_mode and _bl_std > 0:
                _t3 = 3 * _bl_std
                in_r  = delays_plot[np.abs(delays_plot) <= _t3]
                out_r = delays_plot[np.abs(delays_plot) > _t3]
                if len(in_r) > 0:
                    fig_h.add_trace(go.Histogram(x=in_r, xbins=_xbins_h, name="±3σ以内",
                                                 marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                if len(out_r) > 0:
                    fig_h.add_trace(go.Histogram(x=out_r, xbins=_xbins_h, name="±3σ超過",
                                                 marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                fig_h.add_vline(x=_t3,  line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"+3σ({_t3:.1f}ms)")
                fig_h.add_vline(x=-_t3, line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"-3σ({_t3:.1f}ms)")
            else:
                fig_h.add_trace(go.Histogram(x=delays_plot, xbins=_xbins_h,
                                             name="現在データ",
                                             marker_color="rgba(59,130,246,0.72)", opacity=0.88))
            if _delta_mode:
                fig_h.add_vline(x=0, line_color="#b45309", line_width=2,
                                annotation_text=f"基準 {_bl_ref:.1f}ms",
                                annotation_font_color="#b45309")
                xaxis_title = "基準値からのずれ [ms]"
            else:
                fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                                annotation_text=f"3σ {sig3:.1f}ms")
                xaxis_title = "遅れ時間 [ms]"
            fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                                 barmode="overlay", height=260, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            sc = st.columns(4)
            mean_plot = float(np.mean(delays_plot))
            std_plot  = float(np.std(delays_plot))
            sc[0].metric("N",      len(delays_plot))
            sc[1].metric("平均" if not _delta_mode else "差分平均", f"{mean_plot:.1f}ms")
            sc[2].metric("σ",     f"{std_plot:.1f}ms")
            sc[3].metric("3σ" if not _delta_mode else "基準σ",
                         f"{(_bl_std if _delta_mode else sig3):.1f}ms")
            if threshold > 0 and not _delta_mode:
                rate = np.mean(delays <= threshold) * 100
                st.caption(f"閾値達成率: **{rate:.1f}%**")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                      help="ヒストグラムのビン数を手動調整")

    with w_col:
        st.markdown("**波形重ね（全サイクル＋平均）**")
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, (var,))
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        event_t    = step_stat.get("abs_mean", 30.0)
        half_win   = max(event_t * 0.3, 20.0)
        view_start = max(0.0, event_t - half_win)
        view_end   = event_t + half_win

        _MAX_WAVE = 60   # 表示サイクル上限（多いと描画が重くなるため）
        fig_w   = go.Figure()
        all_t, all_v = [], []
        _shown = 0
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = normalize_bool_series(sl[var]).values
            all_t.append(t); all_v.append(v)
            if _shown < _MAX_WAVE:
                fig_w.add_trace(go.Scatter(x=t, y=v, mode="lines",
                                           line=dict(color="rgba(100,100,200,0.10)", width=1),
                                           showlegend=False))
                _shown += 1
        if all_t:
            tmin = min(t[0]  for t in all_t if len(t) > 0)
            tmax = max(t[-1] for t in all_t if len(t) > 0)
            ct   = np.linspace(tmin, tmax, 300)
            mv   = [np.mean([v[np.searchsorted(t, tp)]
                             for t, v in zip(all_t, all_v)
                             if np.searchsorted(t, tp) < len(v)])
                    for tp in ct]
            fig_w.add_trace(go.Scatter(x=ct, y=mv, mode="lines",
                                       line=dict(color="royalblue", width=3), name="平均波形"))
        abs_mean = step_stat.get("abs_mean", mean_d)
        fig_w.add_vline(x=abs_mean, line_dash="dash", line_color="green",
                        annotation_text=f"平均 {abs_mean:.1f}ms")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title="変数値", height=260, margin=dict(t=8, b=32))
        st.plotly_chart(fig_w, width="stretch", key=f"wave_{pname}_{name}")
        st.caption(f"{len(all_t)} サイクル重ね　{view_start:.1f}〜{view_end:.1f} ms")


def _render_range_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    name = step_stat["name"]
    mode = step.get("mode", "range")
    if mode == "on_period":
        start_var = step.get("variable", "")
        end_var   = step.get("variable", "")
    else:
        start_var = step.get("start_var", "")
        end_var   = step.get("end_var", "")
    start_col = f"{name}_start[ms]"
    dur_col   = f"{name}_dur[ms]"

    starts = (result_df[start_col].dropna().values
              if start_col in result_df.columns else np.array([]))
    durs   = (result_df[dur_col].dropna().values
              if dur_col in result_df.columns else np.array([]))

    if len(durs) == 0:
        st.warning(f"データが取得できませんでした（変数: {start_var}）")
        return

    mean_s, mean_d = (float(np.mean(starts)) if len(starts) > 0 else 0), float(np.mean(durs))
    std_d = float(np.std(durs))
    sig3  = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}_dur"

    # 基準値チェック
    _baseline   = st.session_state.get(pk(pname, "baseline"), {})
    _bl_entry   = _baseline.get(name, {})
    _bl_ref_dur = _bl_entry.get("ref_dur_ms")
    _bl_std_dur = _bl_entry.get("std_dur_ms", 0.0)
    _delta_mode = _bl_ref_dur is not None
    durs_plot   = (durs - _bl_ref_dur) if _delta_mode else durs

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "所要時間閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="所要時間閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        if _delta_mode:
            st.caption(f"基準値: **{_bl_ref_dur:.1f} ms**")
        else:
            st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        _bkey  = f"{pname}_{name}_r"
        n_bins = calc_nice_bins(durs_plot, _bkey)
        _vmin_r = float(durs_plot.min()); _vmax_r = float(durs_plot.max())
        _bsz_r  = (_vmax_r - _vmin_r) / n_bins if n_bins > 0 and _vmax_r > _vmin_r else 1.0
        _xbins_r = dict(start=_vmin_r, end=_vmax_r + _bsz_r, size=_bsz_r)

        # ── 比較モード: 各CSV の分布を重ね合わせ ──────────────────
        _cmp_entries_r = st.session_state.get(f"_cmp_entries_{pname}", [])
        _in_compare_r  = (
            st.session_state.get("compare_mode", False)
            and len(_cmp_entries_r) >= 1
        )
        if _in_compare_r:
            st.markdown("**所要時間ヒストグラム（比較モード: CSV別重ね合わせ）**")
            fig_h = go.Figure()
            _stat_rows_r = []
            for _ci_r, _ce_r in enumerate(_cmp_entries_r):
                try:
                    _ce_res_r = cached_analyze_v2(
                        _ce_r["df"], trigger_col, edge,
                        json.dumps(st.session_state.get(pk(pname, "steps_list"), []))
                    )
                    if dur_col not in _ce_res_r.columns:
                        continue
                    _ce_durs_r = _ce_res_r[dur_col].dropna().values
                    if len(_ce_durs_r) == 0:
                        continue
                    _vmin_r = min(_vmin_r, float(_ce_durs_r.min()))
                    _vmax_r = max(_vmax_r, float(_ce_durs_r.max()))
                    _ce_plot_r = (_ce_durs_r - _bl_ref_dur) if _delta_mode else _ce_durs_r
                    _stat_rows_r.append({
                        "CSV":       _ce_r["label"],
                        "N":         len(_ce_durs_r),
                        "平均[ms]":  round(float(np.mean(_ce_durs_r)), 2),
                        "σ[ms]":    round(float(np.std(_ce_durs_r)),  2),
                    })
                except Exception:
                    continue
                _bsz_r   = (_vmax_r - _vmin_r) / n_bins if n_bins > 0 and _vmax_r > _vmin_r else 1.0
                _xbins_r = dict(start=_vmin_r, end=_vmax_r + _bsz_r, size=_bsz_r)
                _hex_r = _ce_r["color"].lstrip("#")
                _rr, _gr, _br = (int(_hex_r[0:2], 16),
                                  int(_hex_r[2:4], 16),
                                  int(_hex_r[4:6], 16))
                fig_h.add_trace(go.Histogram(
                    x=_ce_plot_r, xbins=_xbins_r,
                    name=_ce_r["label"],
                    marker_color=f"rgba({_rr},{_gr},{_br},0.72)", opacity=0.85,
                ))
            _bsz_r   = (_vmax_r - _vmin_r) / n_bins if n_bins > 0 and _vmax_r > _vmin_r else 1.0
            _xbins_r = dict(start=_vmin_r, end=_vmax_r + _bsz_r, size=_bsz_r)
            for _tr_r in fig_h.data:
                _tr_r.xbins = _xbins_r
            if _delta_mode:
                fig_h.add_vline(x=0, line_color="#b45309", line_width=2,
                                annotation_text=f"基準 {_bl_ref_dur:.1f}ms",
                                annotation_font_color="#b45309")
                xaxis_title = "基準値からのずれ [ms]"
            else:
                fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                                annotation_text=f"3σ {sig3:.1f}ms")
                if _bl_ref_dur is not None:
                    fig_h.add_vline(x=_bl_ref_dur, line_color="#b45309", line_width=2,
                                    annotation_text=f"基準 {_bl_ref_dur:.1f}ms",
                                    annotation_font_color="#b45309")
                xaxis_title = "所要時間 [ms]"
            fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                                 barmode="overlay", height=280, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            if _stat_rows_r:
                st.dataframe(pd.DataFrame(_stat_rows_r), hide_index=True, width="stretch")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                      help="ヒストグラムのビン数を手動調整")
        else:
            # ── 通常モード ────────────────────────────────────────
            if _delta_mode:
                st.markdown("**差分ヒストグラム（基準値=0）**")
            else:
                st.markdown("**所要時間ヒストグラム**")

            # 基準CSV オーバーレイ用データを取得
            _ref_plot_r = None
            _ref_key_r  = st.session_state.get("ref_csv_key", "")
            _act_key_r  = st.session_state.get("active_csv", "")
            _store_r    = st.session_state.get("csv_store", {})
            if _ref_key_r and _ref_key_r != _act_key_r and _ref_key_r in _store_r:
                try:
                    _ref_steps_r = st.session_state.get(pk(pname, "steps_list"), [])
                    _ref_res_r   = cached_analyze_v2(
                        _store_r[_ref_key_r]["df"], trigger_col, edge,
                        json.dumps(_ref_steps_r)
                    )
                    if dur_col in _ref_res_r.columns:
                        _rv_r = _ref_res_r[dur_col].dropna().values
                        if len(_rv_r) > 0:
                            _ref_plot_r = (_rv_r - _bl_ref_dur) if _delta_mode else _rv_r
                except Exception:
                    pass

            # ref データが取得できた場合は範囲を統合してビン幅を揃える
            if _ref_plot_r is not None:
                _vmin_r = min(_vmin_r, float(_ref_plot_r.min()))
                _vmax_r = max(_vmax_r, float(_ref_plot_r.max()))
                _bsz_r  = (_vmax_r - _vmin_r) / n_bins if n_bins > 0 and _vmax_r > _vmin_r else 1.0
                _xbins_r = dict(start=_vmin_r, end=_vmax_r + _bsz_r, size=_bsz_r)

            fig_h  = go.Figure()
            # 基準CSV 分布を最初のトレースとして追加（背面に描画）
            if _ref_plot_r is not None:
                fig_h.add_trace(go.Histogram(
                    x=_ref_plot_r,
                    xbins=_xbins_r,
                    name="基準データ分布",
                    marker_color="rgba(245,158,11,0.52)", opacity=0.88,
                ))

            if threshold > 0 and not _delta_mode:
                below = durs_plot[durs_plot <= threshold]
                above = durs_plot[durs_plot >  threshold]
                if len(below):
                    fig_h.add_trace(go.Histogram(x=below, xbins=_xbins_r, name="閾値以内",
                                                 marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                if len(above):
                    fig_h.add_trace(go.Histogram(x=above, xbins=_xbins_r, name="閾値超過",
                                                 marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                fig_h.add_vline(x=threshold, line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"閾値 {threshold}ms")
            elif _delta_mode and _bl_std_dur > 0:
                _t3 = 3 * _bl_std_dur
                in_r  = durs_plot[np.abs(durs_plot) <= _t3]
                out_r = durs_plot[np.abs(durs_plot) > _t3]
                if len(in_r) > 0:
                    fig_h.add_trace(go.Histogram(x=in_r, xbins=_xbins_r, name="±3σ以内",
                                                 marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                if len(out_r) > 0:
                    fig_h.add_trace(go.Histogram(x=out_r, xbins=_xbins_r, name="±3σ超過",
                                                 marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                fig_h.add_vline(x=_t3,  line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"+3σ({_t3:.1f}ms)")
                fig_h.add_vline(x=-_t3, line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"-3σ({_t3:.1f}ms)")
            else:
                fig_h.add_trace(go.Histogram(x=durs_plot, xbins=_xbins_r,
                                             name="現在データ",
                                             marker_color="rgba(59,130,246,0.72)", opacity=0.88))
            if _delta_mode:
                fig_h.add_vline(x=0, line_color="#b45309", line_width=2,
                                annotation_text=f"基準 {_bl_ref_dur:.1f}ms",
                                annotation_font_color="#b45309")
                xaxis_title = "基準値からのずれ [ms]"
            else:
                fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                                annotation_text=f"3σ {sig3:.1f}ms")
                xaxis_title = "所要時間 [ms]"
            fig_h.update_layout(xaxis_title=xaxis_title, yaxis_title="頻度",
                                 barmode="overlay", height=260, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            sc = st.columns(4)
            mean_plot = float(np.mean(durs_plot))
            sc[0].metric("N",        len(durs_plot))
            sc[1].metric("平均所要" if not _delta_mode else "差分平均", f"{mean_plot:.1f}ms")
            sc[2].metric("σ",       f"{std_d:.1f}ms")
            sc[3].metric("3σ上限" if not _delta_mode else "基準σ",
                         f"{(_bl_std_dur if _delta_mode else sig3):.1f}ms")
            if not _delta_mode:
                st.caption(f"開始タイミング平均: **{mean_s:.1f} ms**")
            if threshold > 0 and not _delta_mode:
                rate = np.mean(durs <= threshold) * 100
                st.caption(f"閾値達成率: **{rate:.1f}%**")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey}",
                      help="ヒストグラムのビン数を手動調整")

    with w_col:
        st.markdown("**開始/終了波形重ね**")
        vars_needed = tuple(set(v for v in [start_var, end_var] if v))
        if not vars_needed:
            st.warning("変数が設定されていません")
            return
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, vars_needed)
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        view_start = max(0.0, mean_s - mean_d * 0.5)
        view_end   = mean_s + mean_d * 2.5

        _MAX_WAVE = 60   # 表示サイクル上限
        fig_w  = go.Figure()
        _shown = 0
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            if _shown >= _MAX_WAVE:
                continue
            t = sl["time_offset_ms"].values
            if start_var and start_var in sl.columns:
                fig_w.add_trace(go.Scatter(x=t, y=normalize_bool_series(sl[start_var]).values,
                                           mode="lines", showlegend=False,
                                           line=dict(color="rgba(65,105,225,0.10)", width=1)))
            if end_var and end_var != start_var and end_var in sl.columns:
                fig_w.add_trace(go.Scatter(x=t, y=normalize_bool_series(sl[end_var]).values,
                                           mode="lines", showlegend=False,
                                           line=dict(color="rgba(220,100,60,0.10)", width=1)))
            _shown += 1

        # 平均波形
        if start_var:
            ta, mv = mean_waveform(waveforms, start_var)
            if len(mv) > 0:
                fig_w.add_trace(go.Scatter(x=ta, y=mv, mode="lines",
                                           line=dict(color="royalblue", width=2.5),
                                           name=f"開始変数 {start_var}"))
        if end_var and end_var != start_var:
            ta2, mv2 = mean_waveform(waveforms, end_var)
            if len(mv2) > 0:
                fig_w.add_trace(go.Scatter(x=ta2, y=mv2, mode="lines",
                                           line=dict(color="tomato", width=2.5),
                                           name=f"終了変数 {end_var}"))

        fig_w.add_vline(x=mean_s, line_dash="dash", line_color="green",
                        annotation_text=f"開始 {mean_s:.1f}ms")
        fig_w.add_vline(x=mean_s + mean_d, line_dash="dash", line_color="red",
                        annotation_text=f"終了 {mean_s+mean_d:.1f}ms")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title="変数値", height=260,
                            margin=dict(t=8, b=32), showlegend=True,
                            legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_w, width="stretch", key=f"wave_{pname}_{name}")
        st.caption(f"開始変数: {start_var} {step.get('start_edge','RISE')}  ／  "
                   f"終了変数: {end_var} {step.get('end_edge','FALL')}")


# ═══════════════════════════════════════════════════════════════
# ダイアログ: 工程追加
# ═══════════════════════════════════════════════════════════════

@st.dialog("工程を追加")
def add_process_dialog(bool_cols: list):
    pname   = st.text_input("工程名", placeholder="例: 組み付け工程", key="_dlg_pname")
    trigger = st.selectbox("基準変数（サイクル開始信号）", bool_cols, key="_dlg_trigger")
    edge    = st.radio(
        "トリガーエッジ", ["RISE", "FALL"], horizontal=True,
        format_func=lambda x: "↑ 立ち上がり（OFF→ON）" if x == "RISE" else "↓ 立ち下がり（ON→OFF）",
        key="_dlg_edge",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("登録する", type="primary", disabled=not pname.strip()):
            name = pname.strip()
            if name in st.session_state["processes"]:
                st.error(f"「{name}」は既に登録されています")
            else:
                st.session_state["processes"][name] = {
                    "trigger_col": trigger, "edge": edge,
                    "takt_target_ms": 0, "steps": [],
                }
                st.session_state["_expand_new"] = name
                st.rerun()
    with c2:
        if st.button("キャンセル"):
            st.rerun()


def _render_numeric_detail(df, trigger_col, edge, step_stat, step, pname, result_df):
    """数値条件モードの詳細表示（継続時間ヒストグラム＋変数波形重ね）"""
    name      = step_stat["name"]
    var       = step.get("variable", "")
    op        = step.get("op", "==")
    value     = step.get("value", 0)
    start_col = f"{name}_start[ms]"
    dur_col   = f"{name}_dur[ms]"

    starts = (result_df[start_col].dropna().values
              if start_col in result_df.columns else np.array([]))
    durs   = (result_df[dur_col].dropna().values
              if dur_col in result_df.columns else np.array([]))

    if len(durs) == 0:
        st.warning(f"条件「{var} {op} {value}」が成立するサイクルが見つかりませんでした")
        return

    mean_s, mean_d = (float(np.mean(starts)) if len(starts) > 0 else 0), float(np.mean(durs))
    std_d = float(np.std(durs))
    sig3  = mean_d + 3 * std_d
    threshold_key = f"thresh_{pname}_{name}_dur"

    st.caption(f"条件: **{var} {op} {value}**　　"
               f"成立タイミング平均: **{mean_s:.1f} ms**")

    tc1, tc2 = st.columns([5, 2])
    with tc1:
        threshold = st.number_input(
            "継続時間閾値[ms]", min_value=0.0,
            value=float(st.session_state.get(threshold_key, 0.0)),
            step=0.5, key=f"thresh_input_{pname}_{name}",
            label_visibility="collapsed", placeholder="継続時間閾値[ms]（0=なし）",
        )
        st.session_state[threshold_key] = threshold
    with tc2:
        st.caption(f"推奨（平均+3σ）: **{sig3:.1f} ms**")

    h_col, w_col = st.columns(2)

    with h_col:
        _bkey_n = f"{pname}_{name}_n"
        n_bins  = calc_nice_bins(durs, _bkey_n)
        _vmin_n = float(durs.min()); _vmax_n = float(durs.max())
        _bsz_n  = (_vmax_n - _vmin_n) / n_bins if n_bins > 0 and _vmax_n > _vmin_n else 1.0
        _xbins_n = dict(start=_vmin_n, end=_vmax_n + _bsz_n, size=_bsz_n)

        # ── 比較モード: 各CSV の分布を重ね合わせ ──────────────────
        _cmp_entries_n = st.session_state.get(f"_cmp_entries_{pname}", [])
        _in_compare_n  = (
            st.session_state.get("compare_mode", False)
            and len(_cmp_entries_n) >= 1
        )
        if _in_compare_n:
            st.markdown("**継続時間ヒストグラム（比較モード: CSV別重ね合わせ）**")
            fig_h = go.Figure()
            _stat_rows_n = []
            for _ci_n, _ce_n in enumerate(_cmp_entries_n):
                try:
                    _ce_res_n = cached_analyze_v2(
                        _ce_n["df"], trigger_col, edge,
                        json.dumps(st.session_state.get(pk(pname, "steps_list"), []))
                    )
                    if dur_col not in _ce_res_n.columns:
                        continue
                    _ce_durs_n = _ce_res_n[dur_col].dropna().values
                    if len(_ce_durs_n) == 0:
                        continue
                    _vmin_n = min(_vmin_n, float(_ce_durs_n.min()))
                    _vmax_n = max(_vmax_n, float(_ce_durs_n.max()))
                    _stat_rows_n.append({
                        "CSV":       _ce_n["label"],
                        "N":         len(_ce_durs_n),
                        "平均[ms]":  round(float(np.mean(_ce_durs_n)), 2),
                        "σ[ms]":    round(float(np.std(_ce_durs_n)),  2),
                    })
                except Exception:
                    continue
                _bsz_n   = (_vmax_n - _vmin_n) / n_bins if n_bins > 0 and _vmax_n > _vmin_n else 1.0
                _xbins_n = dict(start=_vmin_n, end=_vmax_n + _bsz_n, size=_bsz_n)
                _hex_n = _ce_n["color"].lstrip("#")
                _rn, _gn, _bn = (int(_hex_n[0:2], 16),
                                  int(_hex_n[2:4], 16),
                                  int(_hex_n[4:6], 16))
                fig_h.add_trace(go.Histogram(
                    x=_ce_durs_n, xbins=_xbins_n,
                    name=_ce_n["label"],
                    marker_color=f"rgba({_rn},{_gn},{_bn},0.72)", opacity=0.85,
                ))
            _bsz_n   = (_vmax_n - _vmin_n) / n_bins if n_bins > 0 and _vmax_n > _vmin_n else 1.0
            _xbins_n = dict(start=_vmin_n, end=_vmax_n + _bsz_n, size=_bsz_n)
            for _tr_n in fig_h.data:
                _tr_n.xbins = _xbins_n
            fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                            annotation_text=f"3σ {sig3:.1f}ms")
            fig_h.update_layout(xaxis_title="継続時間[ms]", yaxis_title="頻度",
                                 barmode="overlay", height=280, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            if _stat_rows_n:
                st.dataframe(pd.DataFrame(_stat_rows_n), hide_index=True, width="stretch")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey_n}",
                      help="ヒストグラムのビン数を手動調整")
        else:
            # ── 通常モード ────────────────────────────────────────
            st.markdown("**継続時間ヒストグラム**")

            # 基準CSV オーバーレイ用データを取得
            _ref_plot_n = None
            _ref_key_n  = st.session_state.get("ref_csv_key", "")
            _act_key_n  = st.session_state.get("active_csv", "")
            _store_n    = st.session_state.get("csv_store", {})
            if _ref_key_n and _ref_key_n != _act_key_n and _ref_key_n in _store_n:
                try:
                    _ref_steps_n = st.session_state.get(pk(pname, "steps_list"), [])
                    _ref_res_n   = cached_analyze_v2(
                        _store_n[_ref_key_n]["df"], trigger_col, edge,
                        json.dumps(_ref_steps_n)
                    )
                    if dur_col in _ref_res_n.columns:
                        _rv_n = _ref_res_n[dur_col].dropna().values
                        if len(_rv_n) > 0:
                            _ref_plot_n = _rv_n
                except Exception:
                    pass

            # ref データが取得できた場合は範囲を統合してビン幅を揃える
            if _ref_plot_n is not None:
                _vmin_n = min(_vmin_n, float(_ref_plot_n.min()))
                _vmax_n = max(_vmax_n, float(_ref_plot_n.max()))
                _bsz_n  = (_vmax_n - _vmin_n) / n_bins if n_bins > 0 and _vmax_n > _vmin_n else 1.0
                _xbins_n = dict(start=_vmin_n, end=_vmax_n + _bsz_n, size=_bsz_n)

            fig_h   = go.Figure()
            if _ref_plot_n is not None:
                fig_h.add_trace(go.Histogram(
                    x=_ref_plot_n,
                    xbins=_xbins_n,
                    name="基準データ分布",
                    marker_color="rgba(245,158,11,0.52)", opacity=0.88,
                ))

            if threshold > 0:
                below = durs[durs <= threshold]
                above = durs[durs >  threshold]
                if len(below):
                    fig_h.add_trace(go.Histogram(x=below, xbins=_xbins_n, name="閾値以内",
                                                 marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                if len(above):
                    fig_h.add_trace(go.Histogram(x=above, xbins=_xbins_n, name="閾値超過",
                                                 marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                fig_h.add_vline(x=threshold, line_dash="dash", line_color="#7c3aed",
                                annotation_text=f"閾値 {threshold}ms")
            else:
                fig_h.add_trace(go.Histogram(x=durs, xbins=_xbins_n,
                                             name="現在データ",
                                             marker_color="rgba(59,130,246,0.72)", opacity=0.88))
            fig_h.add_vline(x=sig3, line_dash="dot", line_color="#9ca3af",
                            annotation_text=f"3σ {sig3:.1f}ms")
            fig_h.update_layout(xaxis_title="継続時間[ms]", yaxis_title="頻度",
                                 barmode="overlay", height=260, margin=dict(t=8, b=32),
                                 showlegend=True)
            st.plotly_chart(fig_h, width="stretch", key=f"hist_{pname}_{name}")
            st.slider("ビン数", 3, 60, n_bins, key=f"_bins_{_bkey_n}",
                      help="ヒストグラムのビン数を手動調整（Freedman-Diaconis による自動算出が既定値）")
            sc = st.columns(4)
            sc[0].metric("N",        len(durs))
            sc[1].metric("平均継続",  f"{mean_d:.1f}ms")
            sc[2].metric("σ",       f"{std_d:.1f}ms")
            sc[3].metric("3σ上限",   f"{sig3:.1f}ms")
            if threshold > 0:
                rate = np.mean(durs <= threshold) * 100
                st.caption(f"閾値達成率: **{rate:.1f}%**")

    with w_col:
        st.markdown(f"**波形重ね（{var}）**")
        if not var or var not in df.columns:
            st.warning("変数が設定されていません")
            return
        try:
            waveforms = cached_waveforms(df, trigger_col, edge, (var,))
        except Exception:
            st.warning("波形データを取得できませんでした")
            return

        half_win   = max(mean_d * 0.8, 30.0)
        view_start = max(0.0, mean_s - half_win * 0.5)
        view_end   = mean_s + mean_d + half_win

        _MAX_WAVE = 60   # 表示サイクル上限
        fig_w  = go.Figure()
        _shown = 0
        all_slices = []
        for cyc in waveforms:
            mask = ((cyc["time_offset_ms"] >= view_start) &
                    (cyc["time_offset_ms"] <= view_end))
            sl = cyc[mask]
            if len(sl) < 2:
                continue
            t = sl["time_offset_ms"].values
            v = sl[var].values
            all_slices.append((t, v))    # 平均計算用（全サイクル使用）
            if _shown < _MAX_WAVE:
                fig_w.add_trace(go.Scatter(x=t, y=v, mode="lines", showlegend=False,
                                           line=dict(color="rgba(100,100,200,0.12)", width=1)))
                _shown += 1
        # 平均ライン（数値なので簡易的に各時刻の平均を取る）
        if all_slices:
            ct = np.linspace(view_start, view_end, 200)
            mv = [np.mean([np.interp(t, ts, vs) for ts, vs in all_slices]) for t in ct]
            fig_w.add_trace(go.Scatter(x=ct, y=mv, mode="lines",
                                       line=dict(color="royalblue", width=2.5), name="平均"))
        fig_w.add_vline(x=mean_s, line_dash="dash", line_color="green",
                        annotation_text=f"条件成立 {mean_s:.1f}ms")
        fig_w.add_vline(x=mean_s + mean_d, line_dash="dash", line_color="red",
                        annotation_text=f"条件終了 {mean_s+mean_d:.1f}ms")
        fig_w.add_hrect(y0=value - 0.3, y1=value + 0.3,
                        fillcolor="yellow", opacity=0.15, line_width=0,
                        annotation_text=f"{op} {value}")
        fig_w.update_layout(xaxis_title="サイクル開始からの経過時間 [ms]",
                            yaxis_title=var, height=260,
                            margin=dict(t=8, b=32), showlegend=True,
                            legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_w, width="stretch", key=f"wave_{pname}_{name}")


# ═══════════════════════════════════════════════════════════════
# ダイアログ群（最小分割）
# ─ cycle_settings_dialog : サイクル設定のみ（⚙️）
# ─ add_step_dialog        : ステップ追加（＋）
# ─ edit_step_dialog       : ステップ編集（ステップをクリック）
# ═══════════════════════════════════════════════════════════════

# ── ① サイクル設定（最小限）──────────────────────────────────

@st.dialog("⚙️ サイクル設定", width="small")
def cycle_settings_dialog(pname: str, bool_cols: list, df: pd.DataFrame):
    st.markdown(f"**{pname}**")

    _, del_col = st.columns([4, 1])
    with del_col:
        if st.button("🗑 削除", type="secondary", width="stretch",
                     key=f"_cyc_del_{pname}"):
            del st.session_state["processes"][pname]
            if st.session_state.get("_expand_new") == pname:
                st.session_state["_expand_new"] = None
            st.rerun()

    _cur_t = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    if _cur_t not in bool_cols:
        _cur_t = bool_cols[0]
    trigger_col = st.selectbox(
        "基準変数", bool_cols, index=bool_cols.index(_cur_t),
        key=pk(pname, "trigger"),
    )
    _cur_e = st.session_state.get(pk(pname, "edge"), "RISE")
    if _cur_e not in ["RISE", "FALL"]:
        _cur_e = "RISE"
    st.radio(
        "エッジ", ["RISE", "FALL"], index=["RISE", "FALL"].index(_cur_e),
        format_func=lambda x: "↑ 立ち上がり" if x == "RISE" else "↓ 立ち下がり",
        key=pk(pname, "edge"), horizontal=True,
    )
    st.number_input(
        "タクト目標 [ms]", min_value=0,
        value=st.session_state.get(pk(pname, "takt"), 0),
        step=10, key=pk(pname, "takt"),
    )
    try:
        cs = cached_detect_cycles(df, trigger_col, _cur_e)
        n  = len(cs)
        c1, c2 = st.columns(2)
        c1.metric("サイクル数", n)
        if n > 1:
            t0 = df.loc[cs[0], "Timestamp"]; t1 = df.loc[cs[-1], "Timestamp"]
            c2.metric("平均周期", f"{(t1-t0).total_seconds()*1000/(n-1):.0f}ms")
    except Exception:
        pass


# ── 全サイクル横断で RISE 時刻を収集するヘルパー ──────────────
def _collect_rise_times(df: pd.DataFrame, bool_cols: list, trigger_col: str,
                        edge: str, cs: list) -> dict:
    """全サイクルを横断して各 Bool 変数の最初の RISE 時刻（中央値）を返す。
    返り値: {var: median_t_ms}（RISE なし変数は除外）
    """
    accum: dict = {}
    for ci, csi in enumerate(cs):
        cei = cs[ci + 1] if ci + 1 < len(cs) else df.index[-1]
        cdf = df.loc[csi:cei]
        t0  = df.loc[csi, "Timestamp"]
        for var in bool_cols:
            if var == trigger_col:
                continue
            try:
                t = find_edge_time(cdf, var, "RISE", t0)
                if t is not None and t >= 0:
                    accum.setdefault(var, []).append(t)
            except Exception:
                pass
    return {var: float(np.median(ts)) for var, ts in accum.items()}


def _collect_bool_events(
    df: pd.DataFrame, bool_cols: list, trigger_col: str, cs: list
) -> "tuple[dict, dict, dict]":
    """全サイクルを横断して Bool 変数の全イベントを収集する（統合版）。

    Returns:
        rise_times  : {var: median_t_ms}   RISE 検出変数（LOW→HIGH）
        on_at_start : {var: median_dur_ms} サイクル開始時に既に HIGH の変数
        fall_times  : {var: median_t_ms}   FALL 検出変数（HIGH→LOW、LOW スタートのもの）
    """
    from analyzer import normalize_bool_series as _nbs_ev

    rise_acc: dict = {}   # {var: [t_ms]}
    on_acc:   dict = {}   # {var: [dur_ms]}
    fall_acc: dict = {}   # {var: [t_ms]}

    for ci, csi in enumerate(cs):
        cei  = cs[ci + 1] if ci + 1 < len(cs) else df.index[-1]
        cdf  = df.loc[csi:cei]
        if len(cdf) == 0:
            continue
        try:
            _t0ns = int(pd.Timestamp(df.loc[csi, "Timestamp"]).value)
            _ts   = cdf["Timestamp"].to_numpy("datetime64[ns]").astype(np.int64)
        except Exception:
            continue

        for var in bool_cols:
            if var == trigger_col or var not in cdf.columns:
                continue
            try:
                _bv = _nbs_ev(cdf[var]).values.astype(np.int8)
                _diff = np.empty_like(_bv); _diff[0] = 0
                _diff[1:] = _bv[1:] - _bv[:-1]

                if _bv[0] == 1:
                    # ── サイクル開始時 HIGH ────────────────────────────
                    _fall_pos = np.nonzero(_diff == -1)[0]
                    if len(_fall_pos) > 0:
                        # 途中で FALL → ON 持続時間 = FALL 時刻 - 開始
                        on_acc.setdefault(var, []).append(
                            (int(_ts[_fall_pos[0]]) - _t0ns) / 1e6)
                    else:
                        # サイクル末まで HIGH → 全サイクル長を持続時間として記録
                        on_acc.setdefault(var, []).append(
                            (_ts[-1] - _t0ns) / 1e6)
                else:
                    # ── サイクル開始時 LOW ─────────────────────────────
                    _rise_pos = np.nonzero(_diff == 1)[0]
                    if len(_rise_pos) > 0:
                        # RISE あり
                        rise_acc.setdefault(var, []).append(
                            (int(_ts[_rise_pos[0]]) - _t0ns) / 1e6)
                    else:
                        # RISE なし → FALL があれば FALL 候補
                        _fall_pos2 = np.nonzero(_diff == -1)[0]
                        if len(_fall_pos2) > 0:
                            fall_acc.setdefault(var, []).append(
                                (int(_ts[_fall_pos2[0]]) - _t0ns) / 1e6)
            except Exception:
                pass

    rise_times  = {v: float(np.median(ts)) for v, ts in rise_acc.items()}
    on_at_start = {v: float(np.median(ds)) for v, ds in on_acc.items()}
    fall_times  = {v: float(np.median(ts)) for v, ts in fall_acc.items()}
    return rise_times, on_at_start, fall_times


# ── ① 自動ステップ候補（1サイクルCSV 時）────────────────────

@st.dialog("🔍 ステップを自動検出", width="large")
def auto_step_dialog(pname: str, bool_cols: list, df: pd.DataFrame):
    """1サイクル内の Bool RISE イベントを時系列順に列挙し、一括追加する。"""
    trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    edge        = st.session_state.get(pk(pname, "edge"), "RISE")
    steps       = list(st.session_state.get(pk(pname, "steps_list"), []))
    added_vars  = {s.get("variable", s.get("start_var", "")) for s in steps}

    st.caption("サイクル内の Bool 変数の状態変化をタイミング順に表示します。"
               "チェックを入れて「一括追加」してください。")

    # サイクル検出
    try:
        cs = cached_detect_cycles(df, trigger_col, edge)
    except Exception:
        cs = []

    if len(cs) == 0:
        st.error("サイクルが検出できません。トリガー設定を確認してください。")
        return

    # 全サイクル横断で全イベント収集（RISE / ON開始 / FALL）
    _rise_times, _on_at_st, _fall_times = _collect_bool_events(
        df, bool_cols, trigger_col, cs)

    # 全候補リスト: RISE → ON開始 → FALL の順
    _all_ev = (
        [(v, t, "RISE")  for v, t in sorted(_rise_times.items(), key=lambda x: x[1])]
        + [(v, d, "ON開始") for v, d in sorted(_on_at_st.items(),  key=lambda x: x[1])]
        + [(v, t, "FALL") for v, t in sorted(_fall_times.items(), key=lambda x: x[1])]
    )

    if not _all_ev:
        st.warning("変数が検出できませんでした。トリガー設定を確認してください。")
        st.caption("💡 エッジを RISE/FALL で切り替えるとサイクルが検出される場合があります。")
        return

    st.caption(
        f"**{len(_all_ev)} 変数**を検出しました　"
        f"🟢 RISE: {len(_rise_times)} 件　"
        f"🔵 ON開始: {len(_on_at_st)} 件　"
        f"🔴 FALL: {len(_fall_times)} 件"
    )

    # チェックボックス一覧
    _sel_key = f"_auto_sel_{pname}"
    if _sel_key not in st.session_state:
        # RISE のみデフォルト選択
        st.session_state[_sel_key] = {
            v: (v not in added_vars and kind == "RISE")
            for v, _, kind in _all_ev
        }

    _sel_all, _clr_all = st.columns(2)
    with _sel_all:
        if st.button("全選択", key=f"_asel_all_{pname}", width="stretch"):
            st.session_state[_sel_key] = {v: (v not in added_vars) for v, _, _ in _all_ev}
    with _clr_all:
        if st.button("全解除", key=f"_asel_none_{pname}", width="stretch"):
            st.session_state[_sel_key] = {v: False for v, _, _ in _all_ev}

    st.divider()
    _checks: dict = {}
    for _vi, (var, t_ms, kind) in enumerate(_all_ev):
        _already = var in added_vars
        _col_chk, _col_lbl = st.columns([1, 9])
        with _col_chk:
            _default = st.session_state[_sel_key].get(
                var, not _already and kind == "RISE")
            _checks[var] = st.checkbox(
                " ", value=_default, key=f"_asel_{pname}_{_vi}",
                disabled=_already,
            )
        with _col_lbl:
            if _already:
                st.markdown(f"`{var}` ✅ 追加済")
            elif kind == "RISE":
                st.markdown(f"`{var}` &nbsp; 🟢 RISE &nbsp; **{t_ms:.1f} ms**")
            elif kind == "ON開始":
                st.markdown(f"`{var}` &nbsp; 🔵 ON開始 &nbsp; 持続 **{t_ms:.1f} ms**")
            else:
                st.markdown(f"`{var}` &nbsp; 🔴 FALL &nbsp; **{t_ms:.1f} ms**")

    st.divider()
    _add_mode = st.radio(
        "追加モード",
        ["RISE 時刻（単一変数）", "ON 期間（1 の間）"],
        index=1,
        horizontal=True,
        key=f"_auto_mode_{pname}",
    )
    _mode_val = "single" if "RISE" in _add_mode else "on_period"

    # sort key: RISE→ rise_times, ON開始→ on_at_st, FALL→ fall_times
    _ev_t_map = {v: t for v, t, _ in _all_ev}
    _to_add = [v for v, chk in _checks.items() if chk and v not in added_vars]
    if st.button(
        f"＋ {len(_to_add)} 件を一括追加" if _to_add else "＋ 追加（未選択）",
        disabled=not _to_add,
        type="primary",
        width="stretch",
        key=f"_auto_add_{pname}",
    ):
        _sorted_to_add = sorted(_to_add, key=lambda v: _ev_t_map.get(v, float("inf")))
        for v in _sorted_to_add:
            steps.append({
                "name":     v,
                "color":    _default_color(len(steps)),
                "mode":     _mode_val,
                "variable": v,
                "edge":     "RISE",
            })
        st.session_state[pk(pname, "steps_list")] = steps
        st.session_state.pop(_sel_key, None)
        st.toast(f"✅ {len(_sorted_to_add)} ステップを追加しました（RISE 順）")
        st.rerun()


# ── ② ステップ追加（＋ ボタン）──────────────────────────────

@st.dialog("＋ ステップを追加", width="large")
def add_step_dialog(pname: str, bool_cols: list, df: pd.DataFrame):
    trigger_col   = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    edge          = st.session_state.get(pk(pname, "edge"), "RISE")
    steps         = list(st.session_state.get(pk(pname, "steps_list"), []))
    cand_vars     = [v for v in bool_cols if v != trigger_col]

    # 工程周期
    proc_cycle_ms = 1000.0
    try:
        cs = cached_detect_cycles(df, trigger_col, edge)
        if len(cs) > 1:
            t0 = df.loc[cs[0], "Timestamp"]; t1 = df.loc[cs[-1], "Timestamp"]
            proc_cycle_ms = (t1 - t0).total_seconds() * 1000 / (len(cs) - 1)
    except Exception:
        pass

    var_periods = {}
    try:
        var_periods = cached_variable_periods(df, tuple(bool_cols))
    except Exception:
        pass

    added_vars = set()
    for s in steps:
        if s.get("mode", "single") == "single":
            added_vars.add(s.get("variable", ""))
        else:
            added_vars.add(s.get("start_var", ""))
            added_vars.add(s.get("end_var", ""))

    def _period_key(v):
        p = var_periods.get(v)
        return abs(p - proc_cycle_ms) if p else float("inf")

    cand_sorted = sorted(cand_vars, key=_period_key)

    # 検索ボックス（キーバージョン方式でリセット）
    srch_ver = st.session_state.get(f"_srch_ver_{pname}", 0)
    if st.session_state.pop(f"_clear_srch_{pname}", False):
        srch_ver += 1
        st.session_state[f"_srch_ver_{pname}"] = srch_ver

    search_txt = st.text_input(
        "🔍 変数名でフィルタ",
        key=f"step_srch_{pname}_v{srch_ver}",
        placeholder="変数名を入力...",
    )

    cand_filtered = (
        [v for v in cand_sorted if search_txt.lower() in v.lower()]
        if search_txt else cand_sorted
    )

    if cand_filtered:
        lbl = "検索結果:" if search_txt else f"候補（周期 {proc_cycle_ms:.0f}ms に近い順）:"
        st.caption(lbl)

        def _fmt(v):
            p = var_periods.get(v)
            tag = " [追加済]" if v in added_vars else (f"  {p:.0f}ms" if p else "")
            return f"{v}{tag}"

        selected_vars = st.multiselect(
            "追加する変数を選択（複数選択可）",
            options=cand_filtered,
            default=[],
            format_func=_fmt,
            key=f"multi_add_{pname}_v{srch_ver}",
            label_visibility="collapsed",
            placeholder="クリックして変数を選択...",
        )

        add_mode_lbl = st.radio(
            "追加モード",
            ["RISE 時刻（単一変数）", "ON 期間（1 の間）"],
            index=1,
            horizontal=True,
            key=f"bulk_add_mode_{pname}",
        )
        add_mode = "single" if "RISE" in add_mode_lbl else "on_period"

        _new = [v for v in selected_vars if v not in added_vars]
        btn_label = f"＋ {len(_new)} 件を追加" if _new else "＋ 追加"
        if st.button(
            btn_label,
            disabled=(not _new),
            type="primary",
            key=f"bulk_add_{pname}",
            width="stretch",
        ):
            for v in _new:
                if add_mode == "on_period":
                    steps.append({
                        "name":     v,
                        "color":    _default_color(len(steps)),
                        "mode":     "on_period",
                        "variable": v,
                    })
                else:
                    steps.append({
                        "name":     v,
                        "color":    _default_color(len(steps)),
                        "mode":     "single",
                        "variable": v,
                        "edge":     "RISE",
                    })
            st.session_state[pk(pname, "steps_list")] = steps
            st.session_state[f"_clear_srch_{pname}"] = True
            st.toast(f"+ {len(_new)}件のステップを追加しました", icon="✅")
            st.rerun()

    if steps:
        st.divider()
        st.caption("追加済み: " + "　→　".join(s["name"] for s in steps))

    # JSON インポート
    with st.expander("📂 設定JSONを読み込む", expanded=False):
        jf = st.file_uploader("JSON", type=["json"], key=f"cfg_json_{pname}",
                               label_visibility="collapsed")
        if jf:
            try:
                loaded = json.load(jf)
                new_list = []
                for i2, s in enumerate(loaded.get("steps", [])):
                    if "mode" in s:
                        new_list.append(s)
                    else:
                        v2 = s.get("variable", "")
                        new_list.append({
                            "name": s.get("label", v2), "color": s.get("color", _default_color(i2)),
                            "mode": "single", "variable": v2, "edge": "RISE",
                        })
                st.session_state[pk(pname, "steps_list")] = new_list
                st.session_state[pk(pname, "takt")] = loaded.get("takt_target_ms", 0)
                st.rerun()
            except Exception as e:
                st.error(f"読み込みエラー: {e}")


# ── ③ ステップ編集（ステップをクリック）────────────────────

@st.dialog("ステップ編集", width="small")
def edit_step_dialog(pname: str, step_idx: int, bool_cols: list, df: pd.DataFrame):
    steps = list(st.session_state.get(pk(pname, "steps_list"), []))
    if step_idx >= len(steps):
        st.error("ステップが見つかりません"); return

    step        = steps[step_idx]
    trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    var_list    = [v for v in bool_cols if v != trigger_col]
    mode        = step.get("mode", "single")
    name        = step.get("name", "")
    color       = step.get("color", _default_color(step_idx))

    # 名前・色・削除（最上部）
    nc1, nc2, nc3 = st.columns([4, 1, 1])
    with nc1:
        new_name  = st.text_input("ステップ名", value=name,
                                   key=f"_ename_{pname}_{step_idx}",
                                   placeholder="表示名を入力…")
    with nc2:
        new_color = st.color_picker("色", value=color,
                                     key=f"_ecolor_{pname}_{step_idx}")
    with nc3:
        st.markdown("<div style='padding-top:28px'></div>", unsafe_allow_html=True)
        if st.button("🗑", width="stretch", key=f"_edel_{pname}_{step_idx}"):
            steps.pop(step_idx)
            st.session_state[pk(pname, "steps_list")] = steps
            st.toast(f"🗑 {name} を削除しました")
            st.rerun()

    # モード切替
    _mode_opts  = ["単一変数", "ON期間", "開始/終了", "数値条件"]
    _mode_idx   = {"single": 0, "on_period": 1, "range": 2, "numeric": 3}.get(mode, 0)
    new_mode_lbl = st.radio(
        "モード", _mode_opts,
        index=_mode_idx, horizontal=True,
        key=f"_emode_{pname}_{step_idx}",
    )
    new_mode = {"単一変数": "single", "ON期間": "on_period", "開始/終了": "range", "数値条件": "numeric"}[new_mode_lbl]

    # 変数設定
    col_types = st.session_state.get("col_types", {})
    num_cols = [c for c, t in col_types.items() if t == "numeric"]

    if new_mode == "single":
        cur_var  = step.get("variable", step.get("start_var", var_list[0] if var_list else ""))
        cur_edge = step.get("edge", "RISE")
        v_i = var_list.index(cur_var) if cur_var in var_list else 0
        new_var = st.selectbox("変数", var_list, index=v_i,
                               key=f"_evar_{pname}_{step_idx}")
        new_edge_lbl = st.radio("エッジ", ["RISE↑", "FALL↓"],
                                index=0 if cur_edge == "RISE" else 1, horizontal=True,
                                key=f"_eedge_{pname}_{step_idx}")
        new_edge = "RISE" if "RISE" in new_edge_lbl else "FALL"
        if new_var:
            st.plotly_chart(make_mini_chart(df, new_var, 48), width="stretch",
                            key=f"_emini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "single", "variable": new_var, "edge": new_edge}

    elif new_mode == "on_period":
        cur_var = step.get("variable", step.get("start_var", var_list[0] if var_list else ""))
        v_i     = var_list.index(cur_var) if cur_var in var_list else 0
        new_var = st.selectbox("変数", var_list, index=v_i,
                               key=f"_eopvar_{pname}_{step_idx}")
        st.caption("選択した変数が 1 の間（RISE → FALL）を計測します")
        if new_var:
            st.plotly_chart(make_mini_chart(df, new_var, 48), width="stretch",
                            key=f"_eopmini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "on_period", "variable": new_var}

    elif new_mode == "range":
        cur_sv = step.get("start_var", step.get("variable", var_list[0] if var_list else ""))
        cur_se = step.get("start_edge", "RISE")
        cur_ev = step.get("end_var",   step.get("variable", var_list[0] if var_list else ""))
        cur_ee = step.get("end_edge", "FALL")
        sv_i   = var_list.index(cur_sv) if cur_sv in var_list else 0
        ev_i   = var_list.index(cur_ev) if cur_ev in var_list else 0

        c1, c2 = st.columns(2)
        with c1:
            new_sv = st.selectbox("開始変数", var_list, index=sv_i,
                                  key=f"_esv_{pname}_{step_idx}")
        with c2:
            new_se = "RISE" if "RISE" in st.radio(
                "開始エッジ", ["RISE↑", "FALL↓"],
                index=0 if cur_se == "RISE" else 1, horizontal=True,
                key=f"_ese_{pname}_{step_idx}") else "FALL"
        c3, c4 = st.columns(2)
        with c3:
            new_ev = st.selectbox("終了変数", var_list, index=ev_i,
                                  key=f"_eev_{pname}_{step_idx}")
        with c4:
            new_ee = "RISE" if "RISE" in st.radio(
                "終了エッジ", ["RISE↑", "FALL↓"],
                index=0 if cur_ee == "RISE" else 1, horizontal=True,
                key=f"_eee_{pname}_{step_idx}") else "FALL"
        upd = {"mode": "range",
               "start_var": new_sv, "start_edge": new_se,
               "end_var":   new_ev, "end_edge":   new_ee}

    else:  # numeric
        cur_nvar  = step.get("variable", num_cols[0] if num_cols else "")
        cur_op    = step.get("op", "==")
        cur_val   = float(step.get("value", 0))
        nv_i      = num_cols.index(cur_nvar) if cur_nvar in num_cols else 0
        _ops      = ["==", ">=", "<=", ">", "<"]

        cn1, cn2, cn3 = st.columns([3, 1, 2])
        with cn1:
            new_nvar = st.selectbox("数値変数", num_cols if num_cols else ["（数値列なし）"],
                                    index=nv_i, key=f"_envar_{pname}_{step_idx}")
        with cn2:
            new_op = st.selectbox("条件", _ops,
                                  index=_ops.index(cur_op) if cur_op in _ops else 0,
                                  key=f"_eop_{pname}_{step_idx}", label_visibility="hidden")
        with cn3:
            new_val = st.number_input("値", value=cur_val, step=1.0,
                                      key=f"_eval_{pname}_{step_idx}", label_visibility="hidden")
        if new_nvar and new_nvar in df.columns:
            st.caption(f"「{new_nvar} {new_op} {new_val:.0f}」が True の区間を計測します")
            st.plotly_chart(make_mini_chart(df, new_nvar, 48), width="stretch",
                            key=f"_enmini_{pname}_{step_idx}",
                            config={"displayModeBar": False})
        upd = {"mode": "numeric", "variable": new_nvar, "op": new_op, "value": new_val}

    # 波形監視変数（アナログ）
    st.markdown("**📈 波形監視変数**")
    num_cols_wave = [c for c, t in col_types.items() if t == "numeric"]
    cur_wvars     = [v for v in step.get("waveform_vars", []) if v in num_cols_wave]
    new_wvars     = st.multiselect(
        "波形監視変数（アナログ）",
        options=num_cols_wave,
        default=cur_wvars,
        key=f"_ewvars_{pname}_{step_idx}",
        placeholder="アナログ変数を選択（複数可）…",
        label_visibility="collapsed",
        help="ステップ開始を基準に各サイクルの波形を重ね合わせ表示します",
    )
    upd["waveform_vars"] = new_wvars

    # 並べ替え
    r1, r2, _ = st.columns([1, 1, 3])
    with r1:
        if st.button("↑ 上へ", disabled=step_idx == 0,
                     key=f"_eup_{pname}_{step_idx}", width="stretch"):
            steps[step_idx], steps[step_idx-1] = steps[step_idx-1], steps[step_idx]
            st.session_state[pk(pname, "steps_list")] = steps
            st.rerun()
    with r2:
        if st.button("↓ 下へ", disabled=step_idx == len(steps)-1,
                     key=f"_edn_{pname}_{step_idx}", width="stretch"):
            steps[step_idx], steps[step_idx+1] = steps[step_idx+1], steps[step_idx]
            st.session_state[pk(pname, "steps_list")] = steps
            st.rerun()

    # 変更を自動保存
    upd["name"]  = new_name or name
    upd["color"] = new_color
    updated = {**step, **upd}
    if updated != step:
        steps[step_idx] = updated
        st.session_state[pk(pname, "steps_list")] = steps

    # JSON エクスポート（折りたたみ）
    with st.expander("💾 設定JSONを保存", expanded=False):
        all_steps = st.session_state.get(pk(pname, "steps_list"), [])
        cfg = {
            "process_name": pname,
            "trigger_col":  st.session_state.get(pk(pname, "trigger"), ""),
            "edge":         st.session_state.get(pk(pname, "edge"), "RISE"),
            "takt_target_ms": st.session_state.get(pk(pname, "takt"), 0),
            "steps":        all_steps,
        }
        st.download_button(
            "ダウンロード",
            data=json.dumps(cfg, ensure_ascii=False, indent=2),
            file_name=f"{pname}_config.json", mime="application/json",
            key=f"_edl_{pname}_{step_idx}", width="stretch",
        )




# ── ④ 基準値登録・編集 ──────────────────────────────────────────

@st.dialog("基準値登録・編集", width="small")
def baseline_dialog(pname: str, step_stats: list, result_df: pd.DataFrame):
    """現在の解析結果から基準値を作成・手動編集する"""
    existing = st.session_state.get(pk(pname, "baseline"), {})

    src_lbl = st.radio(
        "算出方法", ["平均値", "最小値", "特定サイクル", "手動入力"],
        horizontal=True, key=f"_bl_src_{pname}",
    )

    # 特定サイクル選択
    _sel_cyc_row = None
    if src_lbl == "特定サイクル" and "サイクル#" in result_df.columns:
        _cyc_nums = result_df["サイクル#"].dropna().astype(int).tolist()
        _sel_cyc = st.selectbox(
            "サイクルを選択", _cyc_nums,
            format_func=lambda x: f"サイクル {x}",
            key=f"_bl_cyc_sel_{pname}",
        )
        _cyc_mask = result_df["サイクル#"].astype(int) == _sel_cyc
        _sel_cyc_row = result_df[_cyc_mask].iloc[0] if _cyc_mask.any() else None

    new_vals = {}
    st.divider()

    for s in step_stats:
        name = s["name"]
        mode = s["mode"]
        ex   = existing.get(name, {})

        if mode == "single":
            dcol = f"{name}_遅れ[ms]"
            vals = result_df[dcol].dropna().values if dcol in result_df.columns else np.array([])
            if len(vals) == 0:
                continue
            if src_lbl == "平均値":
                default = float(np.mean(vals))
            elif src_lbl == "最小値":
                default = float(np.min(vals))
            elif src_lbl == "特定サイクル":
                default = (float(_sel_cyc_row[dcol])
                           if _sel_cyc_row is not None and dcol in _sel_cyc_row and pd.notna(_sel_cyc_row[dcol])
                           else float(np.mean(vals)))
            else:
                default = float(ex.get("ref_ms", np.mean(vals)))

            st.caption(f"**{name}** — 遅れ時間基準")
            ref_ms = st.number_input(
                "基準値 [ms]", value=default, step=0.5,
                key=f"_bl_{pname}_{name}_rm",
                disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                label_visibility="collapsed",
            )
            new_vals[name] = {
                "mode":   mode,
                "ref_ms": ref_ms,
                "std_ms": float(np.std(vals)),
            }

        else:  # range / on_period / numeric
            dcol  = f"{name}_dur[ms]"
            scol  = f"{name}_start[ms]"
            dvals = result_df[dcol].dropna().values if dcol in result_df.columns else np.array([])
            svals = result_df[scol].dropna().values if scol in result_df.columns else np.array([])
            if len(dvals) == 0:
                continue
            if src_lbl == "平均値":
                default_dur   = float(np.mean(dvals))
                default_start = float(np.mean(svals)) if len(svals) > 0 else 0.0
            elif src_lbl == "最小値":
                default_dur   = float(np.min(dvals))
                default_start = float(np.min(svals)) if len(svals) > 0 else 0.0
            elif src_lbl == "特定サイクル":
                default_dur = (float(_sel_cyc_row[dcol])
                               if _sel_cyc_row is not None and dcol in _sel_cyc_row and pd.notna(_sel_cyc_row[dcol])
                               else float(np.mean(dvals)))
                default_start = (float(_sel_cyc_row[scol])
                                 if _sel_cyc_row is not None and scol in _sel_cyc_row and pd.notna(_sel_cyc_row[scol])
                                 else float(np.mean(svals)) if len(svals) > 0 else 0.0)
            else:
                default_dur   = float(ex.get("ref_dur_ms",   np.mean(dvals)))
                default_start = float(ex.get("ref_start_ms", float(np.mean(svals)) if len(svals) > 0 else 0.0))

            st.caption(f"**{name}** — 開始 / 継続時間基準")
            c1, c2 = st.columns(2)
            with c1:
                ref_start = st.number_input(
                    "開始 [ms]", value=default_start, step=0.5,
                    key=f"_bl_{pname}_{name}_rs",
                    disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                )
            with c2:
                ref_dur = st.number_input(
                    "継続 [ms]", value=default_dur, step=0.5,
                    key=f"_bl_{pname}_{name}_rd",
                    disabled=(src_lbl not in ("手動入力", "特定サイクル")),
                )
            new_vals[name] = {
                "mode":         mode,
                "ref_start_ms": ref_start,
                "ref_dur_ms":   ref_dur,
                "std_dur_ms":   float(np.std(dvals)),
                "std_start_ms": float(np.std(svals)) if len(svals) > 0 else 0.0,
            }
        st.divider()

    n_cyc = len(result_df)
    _btn_label = (
        f"✅ 登録（サイクル {st.session_state.get(f'_bl_cyc_sel_{pname}', '?')} の値）"
        if src_lbl == "特定サイクル" and _sel_cyc_row is not None
        else f"✅ 登録（{n_cyc} サイクルの{src_lbl}）"
    )
    if st.button(_btn_label, type="primary", width="stretch"):
        st.session_state[pk(pname, "baseline")] = new_vals
        st.session_state[pk(pname, "baseline_meta")] = {
            "source":    src_lbl,
            "n_cycles":  n_cyc,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.toast("✅ 基準値を登録しました")
        st.rerun()

    if existing:
        if st.button("🗑 基準値をクリア", width="stretch"):
            for k in [pk(pname, "baseline"), pk(pname, "baseline_meta")]:
                st.session_state.pop(k, None)
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# Session State 初期化
# ═══════════════════════════════════════════════════════════════

if "processes"   not in st.session_state:
    st.session_state["processes"]   = {}
if "_expand_new" not in st.session_state:
    st.session_state["_expand_new"] = None


# ── 複数CSV 一括基準値算出ダイアログ ───────────────────────────

@st.dialog("📂 複数CSV から基準値を一括算出", width="large")
def bulk_baseline_dialog(pname: str, steps_list: list):
    """複数の「良品」CSVを選択し、全ステップの平均基準値を一括算出・登録する。"""
    csv_store   = st.session_state.get("csv_store", {})
    trigger_col = st.session_state.get(pk(pname, "trigger"), "")
    edge        = st.session_state.get(pk(pname, "edge"), "RISE")

    if len(csv_store) < 1:
        st.warning("CSVが読み込まれていません。サイドバーからCSVをアップロードしてください。")
        return

    st.caption("基準値の算出に使う CSV を選択してください（複数選択可）")
    _all_keys   = list(csv_store.keys())
    _label_map  = {
        k: ("📌 " if csv_store[k].get("is_ref") else "📊 ") + csv_store[k].get("label", k)
        for k in _all_keys
    }
    _selected = st.multiselect(
        "良品CSV を選択",
        _all_keys,
        default=_all_keys,
        format_func=lambda k: _label_map[k],
        key="_bulk_bl_sel",
        label_visibility="collapsed",
    )
    if not _selected:
        st.info("CSVを1件以上選択してください")
        return

    steps_json = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
    _accum: dict = {}   # {step_name: {"vals": [], "mode": str}}
    _n_total_cyc = 0

    _prog = st.progress(0, text="解析中…")
    for _bi, _bck in enumerate(_selected):
        _prog.progress((_bi + 1) / len(_selected), text=f"解析中… {_bck}")
        _bcdf = csv_store[_bck].get("df", pd.DataFrame())
        if _bcdf.empty:
            continue
        try:
            _bcrd = cached_analyze_v2(_bcdf, trigger_col, edge, steps_json)
            if _bcrd is None or len(_bcrd) == 0:
                continue
            _, _bcss = build_gantt_v2(_bcrd, steps_list, 0)
            _n_total_cyc += len(_bcrd)
            for _bcs in _bcss:
                _sn  = _bcs["name"]
                _sm  = _bcs["mode"]
                _col = f"{_sn}_遅れ[ms]" if _sm == "single" else f"{_sn}_dur[ms]"
                if _col in _bcrd.columns:
                    _vals = _bcrd[_col].dropna().tolist()
                    _accum.setdefault(_sn, {"vals": [], "mode": _sm})["vals"].extend(_vals)
        except Exception:
            continue
    _prog.empty()

    if not _accum:
        st.error("解析できたステップがありません。トリガー設定を確認してください。")
        return

    # プレビューテーブル
    _prev_rows = []
    for _sn, _sd in _accum.items():
        _v = np.array(_sd["vals"])
        if len(_v) == 0:
            continue
        _prev_rows.append({
            "ステップ":     _sn,
            "モード":      "単一" if _sd["mode"] == "single" else "範囲",
            "平均[ms]":   round(float(np.mean(_v)), 2),
            "σ[ms]":     round(float(np.std(_v)), 2),
            "サンプル数":  len(_v),
        })

    st.markdown(f"**基準値プレビュー**（合計 **{_n_total_cyc}** サイクル / **{len(_selected)}** ファイル）")
    st.dataframe(pd.DataFrame(_prev_rows), hide_index=True, width="stretch")

    if st.button("✅ この基準値を登録する", type="primary", width="stretch",
                 key="_bulk_bl_reg"):
        _new_bl = {}
        for _sn, _sd in _accum.items():
            _v = np.array(_sd["vals"])
            if len(_v) == 0:
                continue
            _mean = float(np.mean(_v))
            _std  = float(np.std(_v))
            if _sd["mode"] == "single":
                _new_bl[_sn] = {
                    "mode":   "single",
                    "ref_ms": round(_mean, 3),
                    "std_ms": round(_std, 3),
                }
            else:
                _new_bl[_sn] = {
                    "mode":          _sd["mode"],
                    "ref_dur_ms":    round(_mean, 3),
                    "std_dur_ms":    round(_std, 3),
                    "ref_start_ms":  0.0,
                    "std_start_ms":  0.0,
                }
        st.session_state[pk(pname, "baseline")] = _new_bl
        st.session_state[pk(pname, "baseline_meta")] = {
            "source":     f"{len(_selected)} 件のCSV",
            "n_cycles":   _n_total_cyc,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.success("✅ 基準値を登録しました")
        st.rerun()


# ── セットアップウィザード ────────────────────────────────────

def _render_wizard(pname: str, step: int, bool_cols: list, df: pd.DataFrame):
    """4ステップのセットアップウィザードをメイン領域に描画する。"""
    trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
    edge        = st.session_state.get(pk(pname, "edge"), "RISE")
    steps_list  = list(st.session_state.get(pk(pname, "steps_list"), []))

    _STEPS = ["① トリガー確認", "② ステップ自動検出", "③ ガントプレビュー", "④ 設定保存"]
    _N = len(_STEPS)

    # ヘッダー
    st.markdown(f"## 🚀 セットアップウィザード　—　**{pname}**")
    st.progress((step - 1) / (_N - 1), text=f"**{_STEPS[step-1]}** （{step}/{_N}）")
    st.divider()

    # ─────────────────────────────────────────────────────────────
    if step == 1:
        st.markdown("### ① トリガー変数を確認")
        st.caption("「サイクル開始」を示すBool変数を選択してください。選択した瞬間に自動で適用されます。")

        _tr_new = st.selectbox(
            "トリガー変数", bool_cols,
            index=bool_cols.index(trigger_col) if trigger_col in bool_cols else 0,
            key="_wiz_trigger",
        )
        _ed_new = st.radio("エッジ", ["RISE", "FALL"], horizontal=True, key="_wiz_edge")

        # 即時反映（ボタン不要）
        if _tr_new != trigger_col or _ed_new != edge:
            st.session_state[pk(pname, "trigger")] = _tr_new
            st.session_state[pk(pname, "edge")]    = _ed_new
            trigger_col = _tr_new
            edge        = _ed_new

        # サイクル数確認
        try:
            _cs = cached_detect_cycles(df, trigger_col, edge)
            _n  = len(_cs)
        except Exception:
            _n = 0

        # 全 Bool 変数のサイクル数をプレビューして最適なトリガー候補を強調
        _cyc_candidates = []
        for _bc in bool_cols:
            try:
                _bc_n = len(cached_detect_cycles(df, _bc, "RISE"))
                _cyc_candidates.append((_bc, _bc_n))
            except Exception:
                pass
        _best_n = max((n for _, n in _cyc_candidates), default=0)
        if _cyc_candidates and _best_n > 0:
            st.caption("💡 **候補**: サイクル数が多い変数がトリガーに向いています")
            _cand_rows = [{"変数": bc, "RISEサイクル数": n,
                           "推奨": "⭐" if n == _best_n else ""}
                          for bc, n in sorted(_cyc_candidates, key=lambda x: -x[1])[:8]]
            st.dataframe(pd.DataFrame(_cand_rows), hide_index=True, width="stretch")

        if _n == 0:
            st.error("サイクルが検出できません。変数・エッジを変更してください。")
        elif _n <= 10:
            st.info(f"✅ {_n} サイクル検出。基準サンプルとして登録できます。")
        else:
            st.success(f"✅ {_n} サイクル検出。")

    # ─────────────────────────────────────────────────────────────
    elif step == 2:
        st.markdown("### ② ステップ候補を選択")
        st.caption("サイクル内の Bool 変数の状態変化を一覧します。"
                   "チェックを入れて一括追加してください。")

        try:
            _cs2 = cached_detect_cycles(df, trigger_col, edge)
        except Exception:
            _cs2 = []

        if not _cs2:
            st.error("サイクルが検出できません。前ステップでトリガーを確認してください。")
        else:
            st.caption(f"全 {len(_cs2)} サイクルを横断して変数を検索中...")
            # 全サイクル横断でイベント収集（共有関数）
            _rise_times2, _on_at_start, _fall_times2 = _collect_bool_events(
                df, bool_cols, trigger_col, _cs2)

            # 全候補をまとめてリスト化: RISE → ON開始 → FALL
            _all_cands = (
                [(v, t, "RISE")  for v, t in sorted(_rise_times2.items(),  key=lambda x: x[1])]
                + [(v, d, "ON開始") for v, d in sorted(_on_at_start.items(), key=lambda x: x[1])]
                + [(v, t, "FALL") for v, t in sorted(_fall_times2.items(),  key=lambda x: x[1])]
            )
            _added = {s.get("variable", s.get("start_var", "")) for s in steps_list}

            if not _all_cands:
                st.warning("変数が検出できませんでした。「次へ →」でスキップして設定モードから手動追加できます。")
                st.caption("💡 ヒント: トリガー変数が正しく設定されているか確認してください。"
                           "エッジを RISE/FALL で切り替えるとサイクルが検出される場合があります。")
            else:
                st.caption(
                    f"**{len(_all_cands)} 変数**を検出しました　"
                    f"🟢 RISE: {len(_rise_times2)} 件　"
                    f"🔵 ON開始: {len(_on_at_start)} 件　"
                    f"🔴 FALL: {len(_fall_times2)} 件"
                )

                _sel_key2 = f"_wiz_sel_{pname}"
                if _sel_key2 not in st.session_state:
                    # RISE のみデフォルト選択（ON開始・FALL は任意）
                    st.session_state[_sel_key2] = {
                        v: (v not in _added and kind == "RISE")
                        for v, _, kind in _all_cands
                    }

                _ca, _cb = st.columns(2)
                with _ca:
                    if st.button("全選択", key="_wiz_all", width="stretch"):
                        st.session_state[_sel_key2] = {v: (v not in _added) for v, _, _ in _all_cands}
                with _cb:
                    if st.button("全解除", key="_wiz_none", width="stretch"):
                        st.session_state[_sel_key2] = {v: False for v, _, _ in _all_cands}

                _checks2: dict = {}
                for _wi, (var, t_ms, kind) in enumerate(_all_cands):
                    _already2 = var in _added
                    _cc, _cl  = st.columns([1, 9])
                    with _cc:
                        _checks2[var] = st.checkbox(
                            " ",
                            value=st.session_state[_sel_key2].get(
                                var, not _already2 and kind == "RISE"),
                            key=f"_wiz_chk_{pname}_{_wi}", disabled=_already2,
                        )
                    with _cl:
                        if _already2:
                            st.markdown(f"`{var}` ✅ 追加済")
                        elif kind == "RISE":
                            st.markdown(f"`{var}` &nbsp; 🟢 RISE &nbsp; **{t_ms:.1f} ms**")
                        elif kind == "ON開始":
                            st.markdown(f"`{var}` &nbsp; 🔵 ON開始 &nbsp; 持続 **{t_ms:.1f} ms**")
                        else:  # FALL
                            st.markdown(f"`{var}` &nbsp; 🔴 FALL &nbsp; **{t_ms:.1f} ms**")

                _wiz_mode2 = st.radio(
                    "追加モード",
                    ["RISE 時刻（単一変数）", "ON 期間（1 の間）"],
                    index=1, horizontal=True, key="_wiz_add_mode2",
                )
                _wiz_mode2_val = "single" if "RISE" in _wiz_mode2 else "on_period"

                _to_add2 = [v for v, _, _ in _all_cands
                            if _checks2.get(v, False) and v not in _added]
                if st.button(
                    f"＋ {len(_to_add2)} 件を追加してプレビューへ",
                    disabled=not _to_add2,
                    type="primary", key="_wiz_add2", width="stretch",
                ):
                    _sort_map2 = {v: t for v, t, _ in _all_cands}
                    _sorted_add = sorted(_to_add2, key=lambda v: _sort_map2.get(v, 9999))
                    for v in _sorted_add:
                        steps_list.append({
                            "name": v, "color": _default_color(len(steps_list)),
                            "mode": _wiz_mode2_val, "variable": v, "edge": "RISE",
                        })
                    st.session_state[pk(pname, "steps_list")] = steps_list
                    st.session_state.pop(_sel_key2, None)
                    st.session_state["wizard_step"] = 3
                    st.rerun()

    # ─────────────────────────────────────────────────────────────
    elif step == 3:
        st.markdown("### ③ ガントプレビュー")
        if not steps_list:
            st.warning("ステップが未設定です。前ステップでステップを追加してください。")
        else:
            try:
                _sj3 = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
                _rd3 = cached_analyze_v2(df, trigger_col, edge, _sj3)
                _fg3, _ss3 = build_gantt_v2(_rd3, steps_list, int(st.session_state.get(pk(pname, "takt"), 0)))
                if _fg3:
                    st.plotly_chart(_fg3, width="stretch", key="_wiz_gantt")
                    st.caption(f"✅ {len(_rd3)} サイクル / {len(_ss3)} ステップ")

                    # ── 基準登録 ─────────────────────────────────────
                    _bl_exist = st.session_state.get(pk(pname, "baseline"), {})
                    _n_rd3    = len(_rd3)
                    if _bl_exist:
                        st.success("📐 基準値登録済み — 上書きする場合は下のボタンを押してください。")
                    else:
                        st.info(f"💡 このCSV（{_n_rd3} サイクル平均）を基準サンプルとして登録できます"
                                "（比較時の差分表示・ヒストグラムに使用）。")

                    if st.button(
                        f"📐 このCSV（{_n_rd3} サイクル平均）を基準として登録",
                        key="_wiz_bl_reg",
                        type="secondary",
                        width="stretch",
                    ):
                        # 各ステップの平均値をそのまま基準値として登録
                        _new_bl: dict = {}
                        for _bs in _ss3:
                            _bsmode = _bs.get("mode", "single")
                            if _bsmode == "single":
                                _new_bl[_bs["name"]] = {
                                    "mode":    "single",
                                    "ref_ms":  round(_bs["abs_mean"], 3),
                                    "std_ms":  round(_bs["abs_std"], 3),
                                }
                            else:
                                _new_bl[_bs["name"]] = {
                                    "mode":         _bsmode,
                                    "ref_start_ms": round(_bs.get("abs_start", 0), 3),
                                    "ref_dur_ms":   round(_bs["mean"], 3),
                                    "std_dur_ms":   round(_bs["abs_std"], 3),
                                    "std_start_ms": 0.0,
                                }
                        st.session_state[pk(pname, "baseline")] = _new_bl
                        st.session_state[pk(pname, "baseline_meta")] = {
                            "source":     "ウィザード（1サイクル平均）",
                            "n_cycles":   len(_rd3),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.toast("✅ 基準値を登録しました")
                        st.rerun()
                else:
                    st.warning("ガントを描画できません。")
            except Exception as _we:
                st.error(f"解析エラー: {_we}")

        if st.button("この設定でOK → 保存へ", type="primary", key="_wiz_ok3", width="stretch"):
            st.session_state["wizard_step"] = 4
            st.rerun()

    # ─────────────────────────────────────────────────────────────
    elif step == 4:
        st.markdown("### ④ 設定を保存")
        st.success("セットアップが完了しました。設定をJSONに保存してください。")
        st.caption(f"工程: **{pname}** / ステップ: {len(steps_list)} 件")

        # JSON 生成（エクスポートロジックを再利用）
        _det_conds4 = {
            k: v for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith(f"wvol_{pname}_")
            and isinstance(v, (bool, int, float, str, list, dict, type(None)))
        }
        _glob_det_conds4 = {
            k: v for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith("wvol___global_")
            and isinstance(v, (bool, int, float, str, list, dict, type(None)))
        }
        _exp4 = {
            "version": "1.2",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source_csv": st.session_state.get("uploaded_filename", ""),
            "processes": {
                pname: {
                    "trigger":        trigger_col,
                    "edge":           edge,
                    "takt_target_ms": int(st.session_state.get(pk(pname, "takt"), 0)),
                    "steps":          steps_list,
                    "baseline":       st.session_state.get(pk(pname, "baseline"), {}),
                    "baseline_meta":  st.session_state.get(pk(pname, "baseline_meta"), {}),
                    "wv_baseline":    st.session_state.get(pk(pname, "wv_baseline"), {}),
                    "wv_xy_baseline": st.session_state.get(pk(pname, "wv_xy_baseline"), {}),
                    "det_conditions": _det_conds4,
                }
            },
            "global_det_conditions": _glob_det_conds4,
        }
        st.download_button(
            "📥 設定をJSONで保存",
            data=json.dumps(_exp4, ensure_ascii=False, indent=2),
            file_name=f"apb_{pname}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            width="stretch",
            key="_wiz_dl4",
        )
        if st.button("✅ ウィザードを終了して解析を開始", type="primary",
                     key="_wiz_done4", width="stretch"):
            st.session_state["wizard_step"]  = 0
            st.session_state.pop("wizard_pname", None)
            st.rerun()

    # ─ ナビゲーション ────────────────────────────────────────────
    st.divider()
    _nav_prev, _nav_cancel, _nav_next = st.columns([2, 2, 2])
    with _nav_prev:
        if st.button("← 前へ", disabled=(step == 1), width="stretch", key="_wiz_prev"):
            st.session_state["wizard_step"] = step - 1
            st.rerun()
    with _nav_cancel:
        if st.button("✖ キャンセル", width="stretch", key="_wiz_cancel"):
            st.session_state["wizard_step"] = 0
            st.session_state.pop("wizard_pname", None)
            st.rerun()
    with _nav_next:
        if step < _N:
            if st.button("次へ →", type="primary", width="stretch", key="_wiz_next"):
                st.session_state["wizard_step"] = step + 1
                st.rerun()


# ── CSVファイル名からラベルを自動パース ─────────────────────────
import re as _re_label
def _parse_csv_label(filename: str) -> str:
    """ファイル名から自動ラベルを生成（例: 20260418_am_line1_good.csv → 2026-04-18 朝番 L1 良品）"""
    stem = _re_label.sub(r"\.csv$", "", filename, flags=_re_label.IGNORECASE)
    # 日付パターン
    _d = _re_label.search(r"(\d{4})(\d{2})(\d{2})", stem)
    date_str = f"{_d.group(1)}-{_d.group(2)}-{_d.group(3)}" if _d else ""
    stem2 = _re_label.sub(r"\d{8}", "", stem)
    # シフト
    shift_map = {"am": "朝番", "pm": "夕番", "night": "夜番", "day": "日勤"}
    shift_str = ""
    for k, v in shift_map.items():
        if _re_label.search(rf"\b{k}\b", stem2, _re_label.IGNORECASE):
            shift_str = v; stem2 = _re_label.sub(rf"\b{k}\b", "", stem2, flags=_re_label.IGNORECASE); break
    # ライン
    _l = _re_label.search(r"line(\d+)", stem2, _re_label.IGNORECASE)
    line_str = f"L{_l.group(1)}" if _l else ""
    # 品質
    qual_map = {"good": "良品", "ng": "NG", "ok": "良品", "fail": "NG"}
    qual_str = ""
    for k, v in qual_map.items():
        if _re_label.search(rf"\b{k}\b", stem2, _re_label.IGNORECASE):
            qual_str = v; break
    parts = [p for p in [date_str, shift_str, line_str, qual_str] if p]
    return " ".join(parts) if parts else stem


# ═══════════════════════════════════════════════════════════════
# 設定JSONヘルパー
# ═══════════════════════════════════════════════════════════════

def _apply_settings_json(loaded: dict) -> str:
    """設定JSONをsession_stateに適用する。成功時はトースト用メッセージを返す。"""
    if "processes" not in loaded:
        raise ValueError("有効な設定JSONではありません（'processes'キーが見つかりません）")
    _new_procs: dict = {}
    for _ip, _ipd in loaded["processes"].items():
        _new_procs[_ip] = {
            "trigger_col":    _ipd.get("trigger", ""),
            "edge":           _ipd.get("edge", "RISE"),
            "takt_target_ms": _ipd.get("takt_target_ms", 0),
            "steps":          _ipd.get("steps", []),
        }
        st.session_state[pk(_ip, "trigger")]    = _ipd.get("trigger", "")
        st.session_state[pk(_ip, "edge")]       = _ipd.get("edge", "RISE")
        st.session_state[pk(_ip, "takt")]       = int(_ipd.get("takt_target_ms", 0))
        st.session_state[pk(_ip, "steps_list")] = _ipd.get("steps", [])
        if _ipd.get("baseline"):
            st.session_state[pk(_ip, "baseline")]      = _ipd["baseline"]
        if _ipd.get("baseline_meta"):
            st.session_state[pk(_ip, "baseline_meta")] = _ipd["baseline_meta"]
        if _ipd.get("wv_baseline"):
            st.session_state[pk(_ip, "wv_baseline")]   = _ipd["wv_baseline"]
        if _ipd.get("wv_xy_baseline"):
            st.session_state[pk(_ip, "wv_xy_baseline")] = _ipd["wv_xy_baseline"]
        for _dc_k, _dc_v in _ipd.get("det_conditions", {}).items():
            st.session_state[_dc_k] = _dc_v
    st.session_state["processes"]   = _new_procs
    st.session_state["_expand_new"] = next(iter(_new_procs), None)
    # __global 波形検査条件を復元
    _n_glob_det = 0
    for _gc_k, _gc_v in loaded.get("global_det_conditions", {}).items():
        st.session_state[_gc_k] = _gc_v
        _n_glob_det += 1
    _n_det = sum(len(v.get("det_conditions", {})) for v in loaded["processes"].values())
    _msg = f"✅ 設定を読み込みました（工程 {len(_new_procs)} 件・波形検出条件 {_n_det} 件"
    if _n_glob_det:
        _msg += f"・波形検査独立条件 {_n_glob_det} 件"
    return _msg + "）"


# ═══════════════════════════════════════════════════════════════
# サイドバー: データ読み込み
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📂 データ")

    # ─── 表示モード切替 ──────────────────────────────────────────
    _vm_labels = ["⚙️ 設定", "👁️ 監視", "📊 品質分析"]
    view_mode = st.radio(
        "表示モード",
        _vm_labels,
        horizontal=True,
        key="view_mode",
        label_visibility="collapsed",
    )
    # ウィザード開始ボタン（設定モード時のみ表示）
    if view_mode == "⚙️ 設定" and st.session_state.get("processes"):
        _wiz_pnames = list(st.session_state["processes"].keys())
        if len(_wiz_pnames) == 1:
            _wiz_target = _wiz_pnames[0]
        else:
            _wiz_target = st.selectbox(
                "ウィザード対象工程", _wiz_pnames, key="_wiz_pname_sel",
                label_visibility="collapsed",
            )
        if st.button("🚀 セットアップウィザード", width="stretch",
                     help="ガイド付きで初期設定を進めます",
                     key="_wiz_start_btn"):
            st.session_state["wizard_step"]  = 1
            st.session_state["wizard_pname"] = _wiz_target
            st.rerun()
    st.divider()

    if "csv_store" not in st.session_state:
        st.session_state["csv_store"]    = {}
    if "ref_csv_key"  not in st.session_state:
        st.session_state["ref_csv_key"]  = None   # 基準CSV キー

    use_sample = st.toggle("サンプルデータを使用", value=True, key="use_sample")

    # ════════════════════════════════════════════════════════════
    # ① 基準CSV（2〜3サイクル程度の型定義用CSV）
    # ════════════════════════════════════════════════════════════
    st.markdown("##### 📌 ① 基準CSV（型定義用）")

    if use_sample:
        # ① 基準CSV: sample_baseline_3cyc.csv (3サイクル、ウィザード用)
        _ref_sample_name = "sample_baseline_3cyc.csv"
        _ref_sample_path = os.path.join(os.path.dirname(__file__), _ref_sample_name)
        # フォールバック: 3サイクル版がなければ従来版を使用
        if not os.path.exists(_ref_sample_path):
            _ref_sample_name = "sample_playback.csv"
            _ref_sample_path = os.path.join(os.path.dirname(__file__), _ref_sample_name)
        _sample_key = _ref_sample_name
        try:
            df, col_types = cached_load_sample(_ref_sample_path)
            _n_ref_cyc = len(df) // 600  # 概算サイクル数
            _is_first_load = _sample_key not in st.session_state["csv_store"]
            st.session_state["csv_store"][_sample_key] = {
                "df": df, "col_types": col_types,
                "label": f"基準サンプル（{_ref_sample_name}）", "is_ref": True,
            }
            st.session_state["ref_csv_key"] = _sample_key
            # active_csv は初回のみ基準CSVに設定する（ユーザーが比較CSVに切替後は上書きしない）
            _cur_active = st.session_state.get("active_csv")
            if _is_first_load or _cur_active not in st.session_state["csv_store"]:
                st.session_state["active_csv"] = _sample_key
            st.caption(f"📌 {_ref_sample_name}（{len(df):,}行）")
            # 少サイクルならウィザード提案フラグ（初回のみ）
            if _n_ref_cyc <= 10 and not st.session_state.get("_ref_sample_loaded"):
                st.session_state["_suggest_wizard"] = True
                st.session_state["_ref_sample_loaded"] = True
        except FileNotFoundError:
            st.error(f"{_ref_sample_name} が見つかりません")
            st.stop()

        # ② 比較CSV: sample_playback.csv (30サイクル) を自動追加
        _cmp_sample_name = "sample_playback.csv"
        _cmp_sample_path = os.path.join(os.path.dirname(__file__), _cmp_sample_name)
        if (_cmp_sample_name != _ref_sample_name
                and os.path.exists(_cmp_sample_path)
                and _cmp_sample_name not in st.session_state["csv_store"]):
            try:
                _cdf, _cct = cached_load_sample(_cmp_sample_path)
                st.session_state["csv_store"][_cmp_sample_name] = {
                    "df": _cdf, "col_types": _cct,
                    "label": "比較サンプル（sample_playback.csv）", "is_ref": False,
                }
            except Exception:
                pass
    else:
        _ref_up = st.file_uploader(
            "基準CSV（2〜10サイクル推奨）", type=["csv"], key="upload_ref",
            help="ステップ定義・基準値登録に使う少サイクルCSVをアップロードしてください",
        )
        if _ref_up:
            _ref_key = _ref_up.name
            if _ref_key not in st.session_state["csv_store"] or \
               not st.session_state["csv_store"][_ref_key].get("is_ref"):
                try:
                    _rdf  = load_csv(_ref_up)
                    _rct  = detect_bool_columns(_rdf)
                    _rlbl = _parse_csv_label(_ref_up.name)
                    st.session_state["csv_store"][_ref_key] = {
                        "df": _rdf, "col_types": _rct, "label": _rlbl, "is_ref": True,
                    }
                    st.session_state["ref_csv_key"] = _ref_key
                    st.session_state["active_csv"]  = _ref_key
                    st.toast(f"✅ 基準CSV: {_ref_up.name}")
                    # 少サイクルならウィザード自動提案フラグ
                    try:
                        _rprocs = st.session_state.get("processes", {})
                        _rtrig  = (list(_rprocs.values())[0].get("trigger_col", "")
                                   if _rprocs else "")
                        _rn = len(cached_detect_cycles(_rdf,
                            st.session_state.get(list(_rprocs.keys())[0] + "__trigger", _rtrig) if _rprocs else "",
                            "RISE")) if _rprocs else 0
                    except Exception:
                        _rn = 0
                    if _rn <= 10:
                        st.session_state["_suggest_wizard"] = True
                except Exception as _re:
                    st.error(f"読み込みエラー: {_re}")
                    st.stop()
            # 基準CSVをアクティブに
            st.session_state["ref_csv_key"] = _ref_key
            st.session_state["active_csv"]  = _ref_key
            _rentry   = st.session_state["csv_store"][_ref_key]
            df        = _rentry["df"]
            col_types = _rentry["col_types"]
            _rlbl     = _rentry.get("label", _ref_key)
            st.success(f"📌 {_rlbl}  ({len(df):,}行)")
            st.session_state["df"] = df
            st.session_state["col_types"] = col_types
            st.session_state["uploaded_filename"] = _ref_key
        else:
            # 既存の基準CSVがあればそれを使う
            _rk = st.session_state.get("ref_csv_key")
            if _rk and _rk in st.session_state["csv_store"]:
                _rentry   = st.session_state["csv_store"][_rk]
                df        = _rentry["df"]
                col_types = _rentry["col_types"]
                st.session_state["df"] = df
                st.session_state["col_types"] = col_types
                st.session_state["active_csv"] = _rk
                st.caption(f"📌 {_rentry.get('label', _rk)}")
            else:
                st.info("基準CSVをアップロードしてください（2〜10サイクル推奨）")
                st.stop()

    # ════════════════════════════════════════════════════════════
    # ② 比較CSV（複数・日常監視・傾向解析用）
    # ════════════════════════════════════════════════════════════
    st.markdown("##### 📊 ② 比較CSV（監視・傾向解析用）")
    _cmp_ups = st.file_uploader(
        "比較CSV（複数選択可）", type=["csv"],
        key="upload_cmp", accept_multiple_files=True,
        help="日常データや複数ロットのCSVを追加して傾向比較できます",
    )
    if _cmp_ups:
        _cmp_names = {uf.name for uf in _cmp_ups}
        for uf in _cmp_ups:
            if uf.name not in st.session_state["csv_store"]:
                try:
                    _cdf = load_csv(uf)
                    _cct = detect_bool_columns(_cdf)
                    _clbl = _parse_csv_label(uf.name)
                    st.session_state["csv_store"][uf.name] = {
                        "df": _cdf, "col_types": _cct, "label": _clbl, "is_ref": False,
                    }
                    st.toast(f"✅ {uf.name}")
                except Exception as _ce:
                    st.error(f"{uf.name}: {_ce}")
        # アップローダーから消えたファイルは削除（基準CSV・フォルダ読み込み分は残す）
        for _ck in [k for k in list(st.session_state["csv_store"].keys())
                    if k not in _cmp_names
                    and not st.session_state["csv_store"][k].get("is_ref", False)
                    and st.session_state["csv_store"][k].get("source") != "folder"]:
            del st.session_state["csv_store"][_ck]

    # ════════════════════════════════════════════════════════════
    # 📁 フォルダ一括読み込み（ローカル）
    # ════════════════════════════════════════════════════════════
    with st.expander("📁 フォルダ一括読み込み（ローカル）", expanded=False):
        _fl_root = st.text_input(
            "ルートフォルダパス",
            key="folder_root",
            placeholder=r"C:\APB\logs",
            help="基準/比較/設定ファイルをサブフォルダに整理したルートフォルダのパス",
        )
        _fl_auto = st.toggle(
            "パス変更時に自動スキャン",
            key="folder_auto_scan",
        )
        _fl_show_adv = st.checkbox("⚙️ サブフォルダ名・件数の設定", key="_fl_show_adv")
        if _fl_show_adv:
            _fca, _fcb, _fcc = st.columns(3)
            with _fca:
                st.text_input("基準", value="baseline", key="folder_sub_ref",
                              help="基準CSV用サブフォルダ名")
            with _fcb:
                st.text_input("比較", value="data",     key="folder_sub_cmp",
                              help="比較CSV用サブフォルダ名")
            with _fcc:
                st.text_input("設定", value="config",   key="folder_sub_cfg",
                              help="設定JSON用サブフォルダ名")
            st.number_input(
                "比較CSV 最大件数（新しい順）", min_value=1, max_value=500,
                value=20, step=10, key="folder_max_cmp",
            )

        _fl_do_scan = st.button("🔄 フォルダをスキャン", key="_fl_scan_btn",
                                width="stretch")

        # 自動スキャン: パスが変わったとき
        _fl_last_root = st.session_state.get("folder_last_root", "")
        if _fl_auto and _fl_root and _fl_root != _fl_last_root:
            _fl_do_scan = True

        if _fl_do_scan and _fl_root:
            _fl_root_n  = os.path.normpath(_fl_root)
            _fl_sub_ref = st.session_state.get("folder_sub_ref", "baseline")
            _fl_sub_cmp = st.session_state.get("folder_sub_cmp", "data")
            _fl_sub_cfg = st.session_state.get("folder_sub_cfg", "config")
            _fl_max_cmp = int(st.session_state.get("folder_max_cmp", 20))

            if not os.path.isdir(_fl_root_n):
                st.error(f"フォルダが見つかりません: {_fl_root_n}")
            else:
                # ── 基準CSV ────────────────────────────────────────
                _fl_ref_dir = os.path.join(_fl_root_n, _fl_sub_ref)
                if os.path.isdir(_fl_ref_dir):
                    _fl_ref_csvs = sorted(
                        glob.glob(os.path.join(_fl_ref_dir, "*.csv")),
                        key=os.path.getmtime, reverse=True,
                    )
                    if _fl_ref_csvs:
                        _fl_rp  = _fl_ref_csvs[0]   # 最新1件
                        _fl_rk  = _fl_rp             # フルパスをキーに
                        if _fl_rk not in st.session_state["csv_store"]:
                            try:
                                _fl_rdf, _fl_rct = cached_load_sample(_fl_rp)
                                _fl_is_ref = not use_sample
                                st.session_state["csv_store"][_fl_rk] = {
                                    "df": _fl_rdf, "col_types": _fl_rct,
                                    "label": f"📌基準({os.path.basename(_fl_rp)})",
                                    "is_ref": _fl_is_ref, "source": "folder",
                                }
                                if _fl_is_ref:
                                    st.session_state["ref_csv_key"] = _fl_rk
                                    st.session_state["active_csv"]  = _fl_rk
                                st.toast(f"📌 基準CSV読み込み: {os.path.basename(_fl_rp)}")
                            except Exception as _fe:
                                st.warning(f"基準CSV読み込みエラー: {_fe}")
                    else:
                        st.toast(f"⚠️ {_fl_sub_ref}/ にCSVが見つかりません")

                # ── 比較CSV ────────────────────────────────────────
                _fl_cmp_dir = os.path.join(_fl_root_n, _fl_sub_cmp)
                if os.path.isdir(_fl_cmp_dir):
                    _fl_cmp_csvs = sorted(
                        glob.glob(os.path.join(_fl_cmp_dir, "*.csv")),
                        key=os.path.getmtime, reverse=True,
                    )[:_fl_max_cmp]
                    _fl_n_new = 0
                    for _fl_cp in _fl_cmp_csvs:
                        if _fl_cp not in st.session_state["csv_store"]:
                            try:
                                _fl_cdf, _fl_cct = cached_load_sample(_fl_cp)
                                st.session_state["csv_store"][_fl_cp] = {
                                    "df": _fl_cdf, "col_types": _fl_cct,
                                    "label": _parse_csv_label(os.path.basename(_fl_cp)),
                                    "is_ref": False, "source": "folder",
                                }
                                _fl_n_new += 1
                            except Exception:
                                pass
                    if _fl_n_new > 0:
                        st.toast(f"📊 比較CSV {_fl_n_new}件 読み込み")
                    elif _fl_cmp_csvs:
                        st.toast("📊 比較CSV: すでに読み込み済み")
                    else:
                        st.toast(f"⚠️ {_fl_sub_cmp}/ にCSVが見つかりません")

                # ── 設定JSON ───────────────────────────────────────
                _fl_cfg_dir = os.path.join(_fl_root_n, _fl_sub_cfg)
                _fl_jsons_new: list = []
                if os.path.isdir(_fl_cfg_dir):
                    _fl_jsons_new = sorted(
                        glob.glob(os.path.join(_fl_cfg_dir, "*.json")),
                        key=os.path.getmtime, reverse=True,
                    )
                    st.session_state["folder_jsons_found"] = _fl_jsons_new
                    if len(_fl_jsons_new) == 1:
                        # 1件のみ → 自動適用
                        try:
                            with open(_fl_jsons_new[0], encoding="utf-8") as _jf:
                                _jd = json.load(_jf)
                            _jmsg = _apply_settings_json(_jd)
                            st.toast(f"⚙️ 設定JSON自動適用: {os.path.basename(_fl_jsons_new[0])}")
                        except Exception as _je:
                            st.warning(f"設定JSON読み込みエラー: {_je}")
                    elif len(_fl_jsons_new) > 1:
                        st.toast(f"⚙️ 設定JSON {len(_fl_jsons_new)}件 → 下記で選択してください")

                st.session_state["folder_last_root"] = _fl_root
                st.rerun()

        # 設定JSON選択（複数ある場合）
        _fl_jsons_st = st.session_state.get("folder_jsons_found", [])
        if len(_fl_jsons_st) > 1:
            _fl_sel = st.selectbox(
                "設定JSONを選択",
                _fl_jsons_st,
                format_func=os.path.basename,
                key="_fl_json_sel",
            )
            if st.button("⚙️ この設定を適用", key="_fl_json_apply", width="stretch"):
                try:
                    with open(_fl_sel, encoding="utf-8") as _jf:
                        _jd = json.load(_jf)
                    _jmsg = _apply_settings_json(_jd)
                    st.toast(_jmsg)
                    st.rerun()
                except Exception as _je:
                    st.error(f"設定JSONエラー: {_je}")
        elif len(_fl_jsons_st) == 1:
            st.caption(f"⚙️ 設定JSON自動適用済み: {os.path.basename(_fl_jsons_st[0])}")

        # フォルダ読み込み済み統計
        _fl_folder_cnt = sum(
            1 for v in st.session_state["csv_store"].values()
            if v.get("source") == "folder"
        )
        if _fl_folder_cnt > 0:
            st.caption(f"📁 フォルダから {_fl_folder_cnt}件 読み込み中")
        if st.session_state.get("folder_last_root"):
            if st.button("🗑️ フォルダ読み込みをリセット", key="_fl_reset_btn"):
                for _fl_del in [k for k, v in st.session_state["csv_store"].items()
                                if v.get("source") == "folder"]:
                    del st.session_state["csv_store"][_fl_del]
                st.session_state.pop("folder_last_root", None)
                st.session_state.pop("folder_jsons_found", None)
                st.rerun()

        # ── 一括保存 ────────────────────────────────────────────
        st.divider()
        st.markdown("**💾 現在のデータを保存**")
        _fl_save_root = _fl_root or st.session_state.get("folder_last_root", "")
        if not _fl_save_root:
            st.caption("⬆️ 上記のフォルダパスを入力してから保存できます")
        else:
            _fl_sub_ref_s = st.session_state.get("folder_sub_ref", "baseline")
            _fl_sub_cmp_s = st.session_state.get("folder_sub_cmp", "data")
            _fl_sub_cfg_s = st.session_state.get("folder_sub_cfg", "config")
            _fl_store_all = st.session_state["csv_store"]
            _fl_n_ref = sum(1 for v in _fl_store_all.values() if v.get("is_ref"))
            _fl_n_cmp = len(_fl_store_all) - _fl_n_ref
            st.caption(
                f"保存先: `{os.path.normpath(_fl_save_root)}`\n\n"
                f"基準CSV {_fl_n_ref}件 → `{_fl_sub_ref_s}/`　"
                f"比較CSV {_fl_n_cmp}件 → `{_fl_sub_cmp_s}/`　"
                f"設定JSON → `{_fl_sub_cfg_s}/`"
            )
            if st.button("💾 フォルダに一括保存", key="_fl_save_btn", width="stretch"):
                try:
                    _fl_sroot = os.path.normpath(_fl_save_root)
                    for _sd in [_fl_sub_ref_s, _fl_sub_cmp_s, _fl_sub_cfg_s]:
                        os.makedirs(os.path.join(_fl_sroot, _sd), exist_ok=True)

                    # CSV保存
                    _fl_n_saved = 0
                    for _sk, _sv in _fl_store_all.items():
                        _ssub = _fl_sub_ref_s if _sv.get("is_ref") else _fl_sub_cmp_s
                        # ファイル名を決定（フルパスキーはbasename、それ以外はそのまま）
                        if _sv.get("source_path"):
                            _sfn = os.path.basename(_sv["source_path"])
                        elif os.sep in _sk or "/" in _sk:
                            _sfn = os.path.basename(_sk)
                        else:
                            _sfn = _sk
                        if not _sfn.lower().endswith(".csv"):
                            _sfn += ".csv"
                        _sv["df"].to_csv(
                            os.path.join(_fl_sroot, _ssub, _sfn), index=False
                        )
                        _fl_n_saved += 1

                    # 設定JSON保存
                    _fl_json_saved = False
                    if st.session_state.get("processes"):
                        _sexp = {}
                        for _sep in st.session_state["processes"]:
                            _sdet = {k: v for k, v in st.session_state.items()
                                     if isinstance(k, str) and k.startswith(f"wvol_{_sep}_")
                                     and isinstance(v, (bool, int, float, str, list, dict, type(None)))}
                            _sexp[_sep] = {
                                "trigger":        st.session_state.get(pk(_sep, "trigger"), ""),
                                "edge":           st.session_state.get(pk(_sep, "edge"), "RISE"),
                                "takt_target_ms": st.session_state.get(pk(_sep, "takt"), 0),
                                "steps":          st.session_state.get(pk(_sep, "steps_list"), []),
                                "baseline":       st.session_state.get(pk(_sep, "baseline"), {}),
                                "baseline_meta":  st.session_state.get(pk(_sep, "baseline_meta"), {}),
                                "wv_baseline":    st.session_state.get(pk(_sep, "wv_baseline"), {}),
                                "wv_xy_baseline": st.session_state.get(pk(_sep, "wv_xy_baseline"), {}),
                                "det_conditions": _sdet,
                            }
                        _sglob_det = {k: v for k, v in st.session_state.items()
                                      if isinstance(k, str) and k.startswith("wvol___global_")
                                      and isinstance(v, (bool, int, float, str, list, dict, type(None)))}
                        _sjson_str = json.dumps(
                            {"version": "1.2",
                             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                             "source_csv": st.session_state.get("uploaded_filename", ""),
                             "processes": _sexp,
                             "global_det_conditions": _sglob_det},
                            ensure_ascii=False, indent=2,
                        )
                        _sjfn  = f"apb_settings_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
                        _sjpath = os.path.join(_fl_sroot, _fl_sub_cfg_s, _sjfn)
                        with open(_sjpath, "w", encoding="utf-8") as _sjf:
                            _sjf.write(_sjson_str)
                        _fl_json_saved = True

                    _msg = f"💾 CSV {_fl_n_saved}件"
                    if _fl_json_saved:
                        _msg += " + 設定JSON"
                    st.toast(_msg + f" を保存しました → {_fl_sroot}")
                    # 保存先をfolder_last_rootにセット（次回スキャンで即読み込める）
                    st.session_state["folder_last_root"] = _fl_save_root
                except Exception as _se:
                    st.error(f"保存エラー: {_se}")

    # 全CSV一覧（基準 + 比較）からアクティブを選択
    _all_keys = list(st.session_state["csv_store"].keys())
    if len(_all_keys) > 1:
        _csv_store_sb = st.session_state["csv_store"]

        def _fmt_csv(k):
            _e = _csv_store_sb.get(k, {})
            _lbl = _e.get("label", k)
            return ("📌 " + _lbl) if _e.get("is_ref") else ("📊 " + _lbl)

        _non_ref_cnt = sum(
            1 for k in _all_keys if not _csv_store_sb.get(k, {}).get("is_ref")
        )
        if _non_ref_cnt >= 1:
            _compare_mode_sb = st.toggle(
                "🔀 比較モード",
                key="compare_mode",
                help="複数CSVを同時に重ねて比較表示します",
            )
        else:
            st.session_state.setdefault("compare_mode", False)
            _compare_mode_sb = False

        if _compare_mode_sb:
            # 比較モード: multiselect で複数CSV を選択
            _cmp_default = [k for k in st.session_state.get("compare_csv_keys", [])
                            if k in _all_keys]
            if not _cmp_default:
                _cmp_default = _all_keys   # デフォルトで全選択
            st.multiselect(
                "比較するCSV（複数選択）",
                _all_keys,
                default=_cmp_default,
                key="compare_csv_keys",
                format_func=_fmt_csv,
            )
            # スキーマ操作は基準CSVのデータを使用
            _ref_k_sb = st.session_state.get("ref_csv_key", _all_keys[0])
            if _ref_k_sb not in _all_keys:
                _ref_k_sb = _all_keys[0]
            _ref_entry_sb     = _csv_store_sb[_ref_k_sb]
            df                = _ref_entry_sb["df"]
            col_types         = _ref_entry_sb["col_types"]
            st.session_state["df"]         = df
            st.session_state["col_types"]  = col_types
            st.session_state["active_csv"] = _ref_k_sb
            st.caption(
                f"🔀 {len(st.session_state.get('compare_csv_keys', []))}件 のCSVを比較中"
            )
        else:
            # 通常モード: ラジオでアクティブCSVを選択
            _active_now = st.session_state.get("active_csv", _all_keys[0])
            if _active_now not in _all_keys:
                _active_now = _all_keys[0]
            _active_idx = _all_keys.index(_active_now)
            st.radio(
                "表示中CSV",
                _all_keys,
                index=_active_idx,
                key="active_csv",
                format_func=_fmt_csv,
            )
            _chosen   = st.session_state["active_csv"]
            _entry    = _csv_store_sb[_chosen]
            df        = _entry["df"]
            col_types = _entry["col_types"]
            st.session_state["df"]        = df
            st.session_state["col_types"] = col_types
            st.session_state["uploaded_filename"] = _chosen

    bool_cols = [c for c, t in col_types.items() if t == "bool"]
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]

    # A-3: データ概要カード
    with st.container():
        _n_bool = len(bool_cols)
        _n_num  = len(num_cols)
        if "Timestamp" in df.columns and len(df) > 0:
            _ts0 = str(df["Timestamp"].iloc[0])[:19]
            _ts1 = str(df["Timestamp"].iloc[-1])[:19]
            st.caption(f"📅 {_ts0}  〜  {_ts1}")
        st.caption(f"🔀 Bool列: **{_n_bool}**　　📊 アナログ列: **{_n_num}**")

    # スキーマ変更検知 → 工程を自動登録（列構成が変わった場合のみリセット）
    schema_hash = hashlib.md5(str(sorted(col_types.keys())).encode()).hexdigest()[:8]
    if st.session_state.get("_schema_hash") != schema_hash:
        st.session_state["_schema_hash"] = schema_hash
        st.session_state["processes"]   = {}
        st.session_state["_expand_new"] = None
        starts = [c for c in bool_cols if any(
            kw in c.lower() for kw in ["start", "begin", "trigger", "trig"]
        )]
        if not starts:
            starts = bool_cols[:1]
        for var in starts:
            pname_auto = var
            for sfx in ["_Start", "_start", "_Begin", "_begin", "_Trigger", "_trigger"]:
                pname_auto = pname_auto.replace(sfx, "")
            st.session_state["processes"][pname_auto] = {
                "trigger_col": var, "edge": "RISE", "takt_target_ms": 0, "steps": [],
            }
        if st.session_state["processes"]:
            st.session_state["_expand_new"] = list(st.session_state["processes"].keys())[0]
            # 基準CSVが少サイクルならウィザード自動提案
            try:
                _auto_pname = list(st.session_state["processes"].keys())[0]
                _auto_tr    = st.session_state["processes"][_auto_pname].get("trigger_col", bool_cols[0] if bool_cols else "")
                _auto_n     = len(cached_detect_cycles(df, _auto_tr, "RISE")) if _auto_tr else 0
                if _auto_n <= 10:
                    st.session_state["_suggest_wizard"] = True
                    st.session_state["_suggest_wizard_pname"] = _auto_pname
            except Exception:
                pass

    for pname, pinfo in st.session_state["processes"].items():
        init_proc_widgets(pname, pinfo, bool_cols)

    # ウィザード自動提案バナー
    if st.session_state.pop("_suggest_wizard", False):
        _sw_pname = st.session_state.get("_suggest_wizard_pname", "")
        try:
            _sw_n = len(cached_detect_cycles(df,
                st.session_state.get(pk(_sw_pname, "trigger"), bool_cols[0] if bool_cols else ""),
                "RISE"))
        except Exception:
            _sw_n = 0
        if _sw_n <= 10 and _sw_pname:
            st.info(f"🔰 **{_sw_n} サイクル**の基準CSVを検出しました。"
                    "ウィザードでステップを自動設定しますか？")
            if st.button("🚀 ウィザードを開始", key="_wiz_auto_start", width="stretch",
                         type="primary"):
                st.session_state["wizard_step"]  = 1
                st.session_state["wizard_pname"] = _sw_pname
                st.rerun()

    # ─── 設定ファイル 保存 / 読み込み ───────────────────────────
    st.divider()
    st.markdown("#### 💾 設定ファイル")

    # 保存（現在の全工程設定をJSONに）
    if st.session_state.get("processes"):
        _exp_procs = {}
        for _ep in st.session_state["processes"]:
            # 波形検出条件（wvol_{pname}_* のsession_stateキーを全収集）
            _det_conds = {
                k: v for k, v in st.session_state.items()
                if isinstance(k, str) and k.startswith(f"wvol_{_ep}_")
                and isinstance(v, (bool, int, float, str, list, dict, type(None)))
            }
            _exp_procs[_ep] = {
                "trigger":        st.session_state.get(pk(_ep, "trigger"), ""),
                "edge":           st.session_state.get(pk(_ep, "edge"), "RISE"),
                "takt_target_ms": st.session_state.get(pk(_ep, "takt"), 0),
                "steps":          st.session_state.get(pk(_ep, "steps_list"), []),
                "baseline":       st.session_state.get(pk(_ep, "baseline"), {}),
                "baseline_meta":  st.session_state.get(pk(_ep, "baseline_meta"), {}),
                "wv_baseline":    st.session_state.get(pk(_ep, "wv_baseline"), {}),
                "wv_xy_baseline": st.session_state.get(pk(_ep, "wv_xy_baseline"), {}),
                "det_conditions": _det_conds,
            }
        _exp_glob_det = {
            k: v for k, v in st.session_state.items()
            if isinstance(k, str) and k.startswith("wvol___global_")
            and isinstance(v, (bool, int, float, str, list, dict, type(None)))
        }
        _exp_json = json.dumps(
            {"version": "1.2",
             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
             "source_csv": st.session_state.get("uploaded_filename", ""),
             "processes": _exp_procs,
             "global_det_conditions": _exp_glob_det},
            ensure_ascii=False, indent=2,
        )
        st.download_button(
            "📥 設定を保存 (JSON)",
            data=_exp_json,
            file_name=f"apb_settings_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            width="stretch",
            key="_settings_export_btn",
        )

    # 読み込み
    _imp_file = st.file_uploader(
        "📤 設定JSONを読み込む", type=["json"],
        key="_settings_import", label_visibility="visible",
    )
    if _imp_file is not None:
        try:
            _imp = json.load(_imp_file)
            _msg = _apply_settings_json(_imp)
            st.toast(_msg)
            st.rerun()
        except Exception as _imp_e:
            st.error(f"読み込みエラー: {_imp_e}")

    # ─── ページナビゲーション ─────────────────────────────────
    st.divider()
    st.page_link("pages/4_ロガー設定ナビ.py",   label="📋 ロガー設定ナビ")
    st.page_link("pages/0_このツールについて.py", label="ℹ️ このツールについて")


# ═══════════════════════════════════════════════════════════════
# メインコンテンツ
# ═══════════════════════════════════════════════════════════════

st.title("🏭 APB タイミング解析")

processes = st.session_state["processes"]
view_mode = st.session_state.get("view_mode", "⚙️ 設定")

_active_key   = st.session_state.get("active_csv", "")
_active_label = (st.session_state.get("csv_store", {})
                 .get(_active_key, {}).get("label", _active_key))

hc1, hc2 = st.columns([6, 1])
with hc1:
    _csv_badge = f"  |  📂 {_active_label}" if _active_label else ""
    st.caption(f"登録済み工程: {len(processes)}件{_csv_badge}")
with hc2:
    if view_mode == "⚙️ 設定":
        if st.button("＋ 工程を追加", type="primary", width="stretch"):
            add_process_dialog(bool_cols)

if not processes:
    st.info("「＋ 工程を追加」から工程を登録してください。")
    st.stop()

# ── セットアップウィザード優先表示 ─────────────────────────────
_wizard_step  = st.session_state.get("wizard_step", 0)
_wizard_pname = st.session_state.get("wizard_pname", "")
if _wizard_step > 0 and _wizard_pname in processes:
    _render_wizard(_wizard_pname, _wizard_step, bool_cols, df)
    st.stop()

# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═
# ページタブ
# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═# ═

_page_tabs = st.tabs(["⚙️ 画面設定", "📈 傾向解析", "🔍 波形検査"])

with _page_tabs[0]:

    # ═══════════════════════════════════════════════════════════════
    # 品質分析モード: 工程横断 Cpk サマリーテーブル
    # ═══════════════════════════════════════════════════════════════
    if view_mode == "📊 品質分析":
        st.markdown("### 📊 工程横断 Cpk サマリー")
        _cpk_rows = []
        for _qpn in processes:
            _qsl  = st.session_state.get(pk(_qpn, "steps_list"), [])
            _qtr  = st.session_state.get(pk(_qpn, "trigger"), bool_cols[0])
            _qed  = st.session_state.get(pk(_qpn, "edge"), "RISE")
            _qtk  = int(st.session_state.get(pk(_qpn, "takt"), 0))
            _qbl  = st.session_state.get(pk(_qpn, "baseline"), {})
            if not _qsl:
                continue
            try:
                _qrj = json.dumps(_qsl, ensure_ascii=False, sort_keys=True)
                _qrd = cached_analyze_v2(df, _qtr, _qed, _qrj)
                if _qrd is None or len(_qrd) == 0:
                    continue
                _, _qss = build_gantt_v2(_qrd, _qsl, _qtk)
            except Exception:
                continue
            for _qs in _qss:
                _qmean = _qs["abs_mean"]
                _qstd  = _qs["abs_std"]
                # Cpk: USL = takt_target, LSL = 0
                if _qtk > 0 and _qstd > 0:
                    _qcpk = round(min((_qtk - _qmean) / (3 * _qstd),
                                      _qmean / (3 * _qstd)), 2)
                    _qcpk_str = (f"🟢 {_qcpk}" if _qcpk >= 1.33
                                 else f"🟡 {_qcpk}" if _qcpk >= 1.0
                                 else f"🔴 {_qcpk}")
                else:
                    _qcpk_str = "—"
                # vs 基準
                _qbl_step = _qbl.get(_qs["name"], {})
                if _qbl_step:
                    _qbl_ref = (
                        _qbl_step.get("ref_start_ms", 0) + _qbl_step.get("ref_dur_ms", 0)
                        if _qs.get("mode") != "single"
                        else _qbl_step.get("ref_ms", 0)
                    )
                    _qdelta = round(_qmean - _qbl_ref, 1)
                    _qd_str = f"{_qdelta:+.1f}"
                else:
                    _qd_str = "—"
                _cpk_rows.append({
                    "工程":      _qpn,
                    "ステップ":  _qs["name"],
                    "平均[ms]":  round(_qmean, 1),
                    "σ[ms]":    round(_qstd, 1),
                    "min[ms]":  round(_qs.get("abs_min", 0), 1),
                    "max[ms]":  round(_qs.get("abs_max", 0), 1),
                    "Cpk":      _qcpk_str,
                    "vs基準[ms]": _qd_str,
                })
        if _cpk_rows:
            _cpk_df = pd.DataFrame(_cpk_rows)
            _cpk_dl, _cpk_tbl = st.columns([1, 9])
            with _cpk_dl:
                st.download_button(
                    "📊 CSV",
                    data=_cpk_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"cpk_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    width="stretch",
                    key="_cpk_dl_btn",
                )
            with _cpk_tbl:
                st.dataframe(_cpk_df, hide_index=True, width="stretch")
            if not any(st.session_state.get(pk(p, "takt"), 0) > 0 for p in processes):
                st.caption("💡 Cpk を表示するにはサイクル設定でタクト目標を設定してください")
        else:
            st.info("解析可能なステップがありません。まず設定モードでステップを追加してください。")

        # ── 良品条件一覧（登録済み基準値テーブル）──────────────────
        st.markdown("### 📐 良品条件一覧（登録済み基準値）")
        _bl_list_rows = []
        for _blpn in processes:
            _blbl  = st.session_state.get(pk(_blpn, "baseline"), {})
            _blmeta = st.session_state.get(pk(_blpn, "baseline_meta"), {})
            _blsl  = st.session_state.get(pk(_blpn, "steps_list"), [])
            if not _blbl:
                continue
            for _blstep in _blsl:
                _blsn = _blstep.get("name", "")
                _blse = _blbl.get(_blsn, {})
                if not _blse:
                    continue
                _blmode = _blse.get("mode", "single")
                if _blmode == "single":
                    _blref  = _blse.get("ref_ms", 0)
                    _blstd  = _blse.get("std_ms", 0)
                    _blrefl = f"{_blref:.1f}"
                else:
                    _blref  = _blse.get("ref_dur_ms", 0)
                    _blstd  = _blse.get("std_dur_ms", 0)
                    _blrefl = f"{_blref:.1f} (持続)"
                _bl_list_rows.append({
                    "工程":       _blpn,
                    "ステップ":   _blsn,
                    "モード":     "単一" if _blmode == "single" else "範囲",
                    "基準値[ms]": _blrefl,
                    "±σ[ms]":    f"±{_blstd:.1f}",
                    "登録元":     _blmeta.get("source", "—"),
                    "登録日時":   _blmeta.get("created_at", "—"),
                })
        if _bl_list_rows:
            _bl_list_df = pd.DataFrame(_bl_list_rows)
            _bll_dl, _bll_tbl = st.columns([1, 9])
            with _bll_dl:
                st.download_button(
                    "📊 CSV",
                    data=_bl_list_df.to_csv(index=False, encoding="utf-8-sig"),
                    file_name=f"baseline_conditions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    width="stretch",
                    key="_bll_dl_btn",
                )
            with _bll_tbl:
                st.dataframe(_bl_list_df, hide_index=True, width="stretch")
        else:
            st.info("基準値が未登録です。設定モードで各工程の基準値を登録してください。")
        st.divider()

    # ═══════════════════════════════════════════════════════════════
    # 工程タイムライン概要（アコーディオン形式）
    # ═══════════════════════════════════════════════════════════════

    _proc_items = []
    for _pn in processes:
        _sl = st.session_state.get(pk(_pn, "steps_list"), [])
        if not _sl:
            continue
        _tr = st.session_state.get(pk(_pn, "trigger"), bool_cols[0])
        _ed = st.session_state.get(pk(_pn, "edge"), "RISE")
        _tk = int(st.session_state.get(pk(_pn, "takt"), 0))
        try:
            _rj = json.dumps(_sl, ensure_ascii=False, sort_keys=True)
            _rd = cached_analyze_v2(df, _tr, _ed, _rj)
            if _rd is None or len(_rd) == 0:
                continue
            _, _ss = build_gantt_v2(_rd, _sl, 0)
            if not _ss:
                continue
            _total  = max((s["abs_mean"] for s in _ss), default=0.0)
            _cyc_ms = 0.0
            try:
                _cs = cached_detect_cycles(df, _tr, _ed)
                if len(_cs) > 1:
                    _t0 = df.loc[_cs[0],  "Timestamp"]
                    _t1 = df.loc[_cs[-1], "Timestamp"]
                    _cyc_ms = (_t1 - _t0).total_seconds() * 1000 / (len(_cs) - 1)
            except Exception:
                pass
            _proc_items.append({
                "proc": _pn, "total": _total, "takt": _tk,
                "cyc": _cyc_ms, "n_steps": len(_sl),
                "trigger": _tr, "edge": _ed,
                "steps_list": _sl, "result_df": _rd, "step_stats": _ss,
            })
        except Exception:
            continue

    if _proc_items:
        _max_total = max(p["total"] for p in _proc_items)
        _x_max = max(
            _max_total,
            max((p["cyc"]  for p in _proc_items if p["cyc"]  > 0), default=0),
            max((p["takt"] for p in _proc_items if p["takt"] > 0), default=0),
        ) * 1.06 or 1.0

        _sum_hdr, _ea, _ca = st.columns([8, 1, 1])
        with _sum_hdr:
            st.markdown("**⏱ 工程タイムライン概要**")
        with _ea:
            if st.button("▼ 全展開", key="_expand_all", width="stretch"):
                for _p in _proc_items:
                    st.session_state[f"_sum_exp_{_p['proc']}"] = True
                st.rerun()
        with _ca:
            if st.button("▶ 全折畳", key="_collapse_all", width="stretch"):
                for _p in _proc_items:
                    st.session_state[f"_sum_exp_{_p['proc']}"] = False
                st.rerun()
        st.caption(
            "▶ をクリックするとステップ詳細ガントが展開されます　"
            "｜　ガントバーをダブルクリックするとヒストグラムが別タブで開きます"
        )

        for _i, _item in enumerate(_proc_items):
            _clr    = STEP_COLORS[_i % len(STEP_COLORS)]
            _is_bn  = (_item["total"] == _max_total)
            _is_exp = st.session_state.get(f"_sum_exp_{_item['proc']}", False)
            _over   = _item["takt"] > 0 and _item["total"] > _item["takt"]

            # ─ 行レイアウト: [▶] [タイムラインバー全体] ────────────
            _rc0, _rc_bar = st.columns([0.55, 11.45])

            with _rc0:
                if st.button(
                    "▼" if _is_exp else "▶",
                    key=f"_sum_tog_{_item['proc']}",
                    width="stretch",
                ):
                    st.session_state[f"_sum_exp_{_item['proc']}"] = not _is_exp
                    st.rerun()

            with _rc_bar:
                _bp = min(99.5, _item["total"] / _x_max * 100)
                _cp = min(99.5, _item["cyc"]   / _x_max * 100) if _item["cyc"]  > 0 else 0
                _tp = min(99.5, _item["takt"]  / _x_max * 100) if _item["takt"] > 0 else -1
                _bdr = "2px solid crimson" if _over else "none"
                _bn_badge = " 🔴" if _is_bn else ""

                if _item["takt"] > 0:
                    _delta = _item["total"] - _item["takt"]
                    _dc = "crimson" if _delta > 0 else "seagreen"
                    _takt_label = (f' &nbsp;<span style="color:{_dc};font-size:11px;">'
                                   f'({_delta:+.0f}ms)</span>')
                else:
                    _takt_label = ""

                _ch = (f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_cp:.1f}%;'
                       f'background:{_clr};opacity:0.13;border-radius:4px;"></div>') if _cp > 0 else ""
                _th = (f'<div style="position:absolute;left:{_tp:.1f}%;top:0;bottom:0;'
                       f'width:2px;background:rgba(210,0,0,0.8);z-index:10;"></div>') if _tp >= 0 else ""

                # ─ プロセス行
                _html = '<div style="position:relative;font-size:13px;">'
                _html += (
                    f'<div style="position:relative;height:28px;margin-bottom:2px;">'
                    f'<div style="position:absolute;left:0;top:0;right:0;bottom:0;'
                    f'background:#f0f0f0;border-radius:4px;border:{_bdr};overflow:hidden;">'
                    f'{_ch}'
                    f'<div style="position:absolute;left:0;top:0;bottom:0;width:{_bp:.1f}%;'
                    f'background:{_clr};border-radius:4px;opacity:0.85;"></div>'
                    f'</div>'
                    f'{_th}'
                    f'<div style="position:absolute;left:{min(99, _bp+0.5):.1f}%;top:50%;'
                    f'transform:translateY(-50%);white-space:nowrap;padding-left:6px;font-weight:bold;">'
                    f'{_item["proc"]}{_bn_badge} '
                    f'<span style="font-weight:normal;color:#666;">{_item["total"]:.0f}ms</span>'
                    f'{_takt_label}'
                    f'</div>'
                    f'</div>'
                )

                # ─ ステップ行（展開時）
                if _is_exp:
                    for _s in _item["step_stats"]:
                        _sl = _s["start"] / _x_max * 100
                        _sw = max(0.5, _s["mean"] / _x_max * 100)
                        _pct = _s["mean"] / (_item["total"] or 1) * 100
                        _sc  = _s["color"]
                        _html += (
                            f'<div style="position:relative;height:22px;margin-bottom:1px;">'
                            f'<div style="position:absolute;left:{_sl:.1f}%;width:{_sw:.1f}%;'
                            f'top:4px;height:14px;background:{_sc};border-radius:3px;opacity:0.75;"></div>'
                            f'<div style="position:absolute;left:{min(99, _sl+_sw+0.5):.1f}%;top:3px;'
                            f'white-space:nowrap;padding-left:3px;font-size:12px;color:#444;">'
                            f'{_s["name"]} '
                            f'<span style="color:#888;">{_s["mean"]:.0f}ms</span> '
                            f'<span style="color:{_sc};font-weight:600;">({_pct:.0f}%)</span>'
                            f'</div>'
                            f'</div>'
                        )

                _html += '</div>'
                st.markdown(_html, unsafe_allow_html=True)

            # ─ ボトルネックメッセージ（展開時）
            if _is_exp and _item["step_stats"]:
                _max_st = max(_item["step_stats"], key=lambda s: s["mean"])
                _pct_max = _max_st["mean"] / (_item["total"] or 1) * 100
                st.caption(
                    f"📌 最長ステップ: **{_max_st['name']}**  "
                    f"{_max_st['mean']:.1f} ms　工程比 {_pct_max:.0f}%"
                )
                if _item["takt"] > 0 and _item["total"] > _item["takt"]:
                    _need = _item["total"] - _item["takt"]
                    st.caption(
                        f"⚠️ タクト目標達成には **{_need:.0f} ms** の短縮が必要。"
                        f"まず **{_max_st['name']}** を改善してください。"
                    )
                elif _item["takt"] > 0:
                    _margin = _item["takt"] - _item["total"]
                    st.caption(f"✅ タクト目標を **{_margin:.0f} ms** 下回っています。")

            st.markdown('<hr style="margin:4px 0 8px 0;opacity:0.2;">', unsafe_allow_html=True)

        st.markdown("")


    # ═══════════════════════════════════════════════════════════════
    # 工程ごとのセクション
    # ═══════════════════════════════════════════════════════════════

    for pname, pinfo in list(processes.items()):

        trigger_col = st.session_state.get(pk(pname, "trigger"), bool_cols[0])
        if trigger_col not in bool_cols:
            trigger_col = bool_cols[0]
            st.session_state[pk(pname, "trigger")] = trigger_col

        edge        = st.session_state.get(pk(pname, "edge"), "RISE")
        takt_target = st.session_state.get(pk(pname, "takt"), 0)
        edge_s      = "↑" if edge == "RISE" else "↓"
        steps_list  = st.session_state.get(pk(pname, "steps_list"), [])

        try:
            n_cyc = len(cached_detect_cycles(df, trigger_col, edge))
        except Exception:
            n_cyc = 0

        step_lbl = f"{len(steps_list)}ステップ" if steps_list else "ステップ未設定"
        _is_new  = (pname == st.session_state.get("_expand_new"))

        with st.expander(
            f"🏭  **{pname}**　　{trigger_col} {edge_s}　|　{n_cyc}サイクル　|　{step_lbl}",
            expanded=_is_new,
        ):
            # ── ヘッダ: 工程名変更 ＋ サイクル設定 ＋ ステップ追加 ──
            if view_mode == "⚙️ 設定":
                # 1サイクル時は「🔍 自動検出」ボタンを追加
                _is_setup_mode = n_cyc <= 10  # 少サイクルは初期セットアップと判断
                if _is_setup_mode:
                    st.info(f"🔰 **セットアップモード**: {n_cyc} サイクル検出。"
                            "「🔍 ステップ自動検出」で変数を自動的に候補として表示できます。",
                            icon=None)
                    name_col, cyc_col, add_col, auto_col = st.columns([3, 1, 1.5, 1.5])
                else:
                    name_col, cyc_col, add_col = st.columns([4, 1, 1.5])
                    auto_col = None
                with name_col:
                    new_name = st.text_input(
                        "工程名", value=pname,
                        key=f"rename_input_{pname}",
                        label_visibility="collapsed",
                        placeholder="工程名を変更",
                    )
                with cyc_col:
                    if st.button("⚙️", key=f"open_cyc_{pname}",
                                 width="stretch", help="サイクル設定"):
                        cycle_settings_dialog(pname, bool_cols, df)
                with add_col:
                    if st.button("＋ ステップを追加", key=f"open_add_{pname}",
                                 width="stretch", type="primary"):
                        add_step_dialog(pname, bool_cols, df)
                if auto_col is not None:
                    with auto_col:
                        if st.button("🔍 自動検出", key=f"open_auto_{pname}",
                                     width="stretch"):
                            auto_step_dialog(pname, bool_cols, df)

                new_name_stripped = new_name.strip()
                if (new_name_stripped and
                        new_name_stripped != pname and
                        new_name_stripped not in processes):
                    rename_process(pname, new_name_stripped)
                    st.toast(f"✏️ 工程名を {pname} → {new_name_stripped} に変更しました")
                    st.rerun()
            else:
                st.caption(f"📌 {pname}　　{trigger_col} {edge_s}　|　{n_cyc}サイクル")

            # ── ステップチップ（クリックで編集ダイアログ）──────────
            # ボタン自体をステップ色のチップとして表示。
            # CSS :has() で直前マーカー span を起点にボタンを着色。
            if steps_list and view_mode == "⚙️ 設定":
                n_chips = len(steps_list)
                chip_cols = st.columns(min(6, n_chips))
                for ci, step in enumerate(steps_list):
                    clr   = step.get("color", _default_color(ci))
                    lbl   = step.get("name", f"Step{ci+1}")
                    mode  = step.get("mode", "single")
                    icon  = "↔️" if mode == "range" else "→"
                    with chip_cols[ci % min(6, n_chips)]:
                        # マーカー span + CSS injection で、直後ボタンを着色
                        # DOM: stElementContainer > stMarkdown > stMarkdownContainer > span#id
                        #       ↕ (adjacent sibling)
                        #      stElementContainer > stButton > ... > button
                        safe_id = "cm_" + "".join(
                            c if c.isalnum() else "_" for c in f"{pname}_{ci}"
                        )
                        _is_sel = (lbl == st.session_state.get(f"sel_step_{pname}", ""))
                        _sel_css = (
                            f'    background: linear-gradient(90deg,{clr}55 0%,{clr}22 55%) !important;'
                            f'    box-shadow: 0 0 0 2px {clr}88 !important;'
                        ) if _is_sel else (
                            f'    background: linear-gradient(90deg,{clr}28 0%,transparent 55%) !important;'
                        )
                        st.markdown(
                            f'<span id="{safe_id}"></span>'
                            f'<style>'
                            f'[data-testid="stElementContainer"]:has(#{safe_id}) + '
                            f'[data-testid="stElementContainer"] button'
                            f' {{ border-left: 5px solid {clr} !important;'
                            f'    padding-left: 10px !important;'
                            f'{_sel_css}}}'
                            f'</style>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            f"{icon} {lbl}",
                            key=f"chip_{pname}_{ci}",
                            width="stretch",
                            help="クリックして編集",
                        ):
                            edit_step_dialog(pname, ci, bool_cols, df)

            # ── メインビュー ──────────────────────────────────────
            result_df = pd.DataFrame()

            if not steps_list:
                if view_mode == "⚙️ 設定":
                    st.info("「＋ ステップを追加」からステップ変数を追加するとガントチャートが表示されます")
                else:
                    st.info("ステップが未設定です。設定モードでステップを追加してください。")
            else:
                # ステップ JSON をキャッシュキーに使用
                steps_json = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
                try:
                    result_df = cached_analyze_v2(df, trigger_col, edge, steps_json)
                except Exception as e:
                    st.error(f"解析エラー: {e}")
                    result_df = pd.DataFrame()

                if result_df is None or len(result_df) == 0:
                    st.warning("サイクルデータが取得できません。設定の基準変数を確認してください。")
                else:
                    fig_gantt, step_stats = build_gantt_v2(result_df, steps_list, takt_target)

                    # ── 比較モード: 複数CSV を結合した Gantt を再構築 ──────────
                    _is_compare = (
                        st.session_state.get("compare_mode", False)
                        and len(st.session_state.get("compare_csv_keys", [])) >= 1
                    )
                    _cmp_entries: list = []
                    if _is_compare:
                        _cmp_keys_g  = st.session_state.get("compare_csv_keys", [])
                        _cmp_store_g = st.session_state.get("csv_store", {})
                        for _ci_g, _ck_g in enumerate(_cmp_keys_g):
                            if _ck_g not in _cmp_store_g:
                                continue
                            _ce_g       = _cmp_store_g[_ck_g]
                            _ce_color_g = _CMP_PALETTE[_ci_g % len(_CMP_PALETTE)]
                            try:
                                _ce_rd_g = cached_analyze_v2(
                                    _ce_g["df"], trigger_col, edge, steps_json
                                )
                            except Exception:
                                continue
                            if _ce_rd_g is None or len(_ce_rd_g) == 0:
                                continue
                            _cmp_entries.append({
                                "key":       _ck_g,
                                "label":     _ce_g.get("label", _ck_g),
                                "df":        _ce_g["df"],
                                "result_df": _ce_rd_g,
                                "color":     _ce_color_g,
                            })
                        if _cmp_entries:
                            _combined_rd = pd.concat(
                                [e["result_df"] for e in _cmp_entries],
                                ignore_index=True,
                            )
                            fig_gantt, step_stats = build_gantt_v2(
                                _combined_rd, steps_list, takt_target
                            )
                            _n_total_cmp = sum(len(e["result_df"]) for e in _cmp_entries)
                            st.caption(
                                f"🔀 比較モード: **{len(_cmp_entries)}件** のCSV / "
                                f"計 **{_n_total_cmp}** サイクル"
                            )
                    # step detail 等で参照できるようセッションに保存
                    st.session_state[f"_cmp_entries_{pname}"] = _cmp_entries

                    # ── 基準値オーバーレイ（比較CSV閲覧時・単CSV時のみ）────────
                    _csv_store_g = st.session_state.get("csv_store", {})
                    _ref_key_g   = st.session_state.get("ref_csv_key", "")
                    _act_key_g   = st.session_state.get("active_csv", "")
                    _is_cmp_g    = bool(
                        not _is_compare
                        and fig_gantt and step_stats
                        and _ref_key_g and _act_key_g
                        and _act_key_g != _ref_key_g
                    )

                    if _is_cmp_g:
                        _bl_g = st.session_state.get(pk(pname, "baseline"), {})
                        _bl_source_g = "登録済み"

                        # 基準未登録 → 基準サンプルCSV から算出
                        if not _bl_g and _ref_key_g in _csv_store_g:
                            try:
                                _ref_df_g  = _csv_store_g[_ref_key_g]["df"]
                                _ref_res_g = cached_analyze_v2(
                                    _ref_df_g, trigger_col, edge, steps_json)
                                if _ref_res_g is not None and len(_ref_res_g) > 0:
                                    _, _ref_ss_g = build_gantt_v2(
                                        _ref_res_g, steps_list, 0)
                                    _bl_g = {}
                                    for _rs_g in _ref_ss_g:
                                        _rn_g = _rs_g["name"]
                                        _rm_g = _rs_g["mode"]
                                        if _rm_g == "single":
                                            _bl_g[_rn_g] = {
                                                "mode":   _rm_g,
                                                "ref_ms": _rs_g["mean"],
                                            }
                                        else:
                                            _bl_g[_rn_g] = {
                                                "mode":         _rm_g,
                                                "ref_start_ms": _rs_g.get(
                                                    "abs_start", _rs_g["start"]),
                                                "ref_dur_ms":   _rs_g["mean"],
                                            }
                                    _bl_source_g = "基準CSV平均"
                            except Exception:
                                pass

                        if _bl_g:
                            # 各ステップ行に赤い縦線を add_shape で描画
                            # カテゴリ軸では y0/y1 にカテゴリインデックス(整数)を使用
                            _any_shape = False
                            for _gi, _ss_g in enumerate(step_stats):
                                _sn_g = _ss_g["name"]
                                _sb_g = _bl_g.get(_sn_g, {})
                                if not _sb_g:
                                    continue
                                _bmode = _sb_g.get("mode", "single")
                                _half  = 0.38   # バー幅 0.5 の内側に収める

                                if _bmode == "single":
                                    _rx = float(_sb_g.get("ref_ms", 0.0))
                                    fig_gantt.add_shape(
                                        type="line",
                                        x0=_rx, x1=_rx,
                                        y0=_gi - _half, y1=_gi + _half,
                                        xref="x", yref="y",
                                        line=dict(color="red", width=4),
                                    )
                                    # ホバー用の不可視 scatter（tooltip）
                                    fig_gantt.add_trace(go.Scatter(
                                        x=[_rx], y=[_sn_g],
                                        mode="markers",
                                        marker=dict(size=12, color="rgba(0,0,0,0)",
                                                    line=dict(width=0)),
                                        hovertemplate=(
                                            f"📐 基準({_bl_source_g}): {_rx:.1f} ms"
                                            "<extra></extra>"),
                                        showlegend=False,
                                    ))
                                else:
                                    _rs_x = float(_sb_g.get("ref_start_ms", 0.0))
                                    _rd_g = float(_sb_g.get("ref_dur_ms", 0.0))
                                    _re_x = _rs_x + _rd_g
                                    # 開始側：細め破線
                                    fig_gantt.add_shape(
                                        type="line",
                                        x0=_rs_x, x1=_rs_x,
                                        y0=_gi - _half, y1=_gi + _half,
                                        xref="x", yref="y",
                                        line=dict(color="red", width=2, dash="dash"),
                                    )
                                    # 終了側：太め実線
                                    fig_gantt.add_shape(
                                        type="line",
                                        x0=_re_x, x1=_re_x,
                                        y0=_gi - _half, y1=_gi + _half,
                                        xref="x", yref="y",
                                        line=dict(color="red", width=4),
                                    )
                                    fig_gantt.add_trace(go.Scatter(
                                        x=[_rs_x, _re_x], y=[_sn_g, _sn_g],
                                        mode="markers",
                                        marker=dict(size=12, color="rgba(0,0,0,0)",
                                                    line=dict(width=0)),
                                        hovertemplate=(
                                            f"📐 基準({_bl_source_g})<br>"
                                            f"開始: {_rs_x:.1f} ms / "
                                            f"終了: {_re_x:.1f} ms（継続 {_rd_g:.1f} ms）"
                                            "<extra></extra>"),
                                        showlegend=False,
                                    ))
                                _any_shape = True

                            # 凡例エントリ（shape は凡例に出ないため dummy trace）
                            if _any_shape:
                                fig_gantt.add_trace(go.Scatter(
                                    x=[None], y=[None], mode="lines",
                                    line=dict(color="red", width=4),
                                    name=f"📐 基準値（{_bl_source_g}）",
                                    showlegend=True,
                                ))
                                fig_gantt.update_layout(showlegend=True)

                    # ── 時系列自動並び替え ──────────────────────────
                    if step_stats and view_mode == "⚙️ 設定":
                        _sort_col, _ = st.columns([2, 8])
                        with _sort_col:
                            if st.button(
                                "⇅ 時系列で自動整列",
                                key=f"auto_sort_{pname}",
                                help="ステップをトリガーからの経過時刻順に並び替えます",
                            ):
                                def _abs_t(stat):
                                    if stat.get("mode") == "range":
                                        return stat.get("abs_start", stat.get("abs_mean", 0))
                                    return stat.get("abs_mean", 0)
                                sort_map = {s["name"]: _abs_t(s) for s in step_stats}
                                sorted_steps = sorted(
                                    steps_list,
                                    key=lambda sl: sort_map.get(sl.get("name", ""), 0),
                                )
                                st.session_state[pk(pname, "steps_list")] = sorted_steps
                                st.toast("⇅ ステップを時系列順に並び替えました")
                                st.rerun()

                    # ── NG サマリー（監視 / 品質分析モード）────────────
                    if view_mode in ("👁️ 監視", "📊 品質分析") and step_stats:
                        if _is_compare and _cmp_entries:
                            # 比較モード: CSV ごとに 1 行ずつ表示
                            for _ce_ng in _cmp_entries:
                                try:
                                    _, _ce_ss_ng = build_gantt_v2(
                                        _ce_ng["result_df"], steps_list, takt_target
                                    )
                                    _ce_out = detect_outliers_iqr(
                                        _ce_ng["result_df"], _ce_ss_ng
                                    )
                                except Exception:
                                    _ce_out = []
                                _ce_lbl = _ce_ng["label"]
                                _ce_n   = len(_ce_ng["result_df"])
                                if _ce_out:
                                    _ng_msg_ce = "  |  ".join(
                                        f"**{o['name']}** {len(o['cycles'])}サイクルNG"
                                        for o in _ce_out
                                    )
                                    st.error(f"🚨 **{_ce_lbl}**: {_ng_msg_ce}")
                                else:
                                    st.success(
                                        f"✅ **{_ce_lbl}**: 全{_ce_n}サイクル正常"
                                    )
                        else:
                            _outliers_sum = detect_outliers_iqr(result_df, step_stats)
                            if _outliers_sum:
                                _ng_msg = "  |  ".join(
                                    f"**{o['name']}** {len(o['cycles'])}サイクルNG"
                                    for o in _outliers_sum
                                )
                                st.error(f"🚨 NG検出: {_ng_msg}")
                            else:
                                st.success(
                                    f"✅ {pname}: 全{len(result_df)}サイクル正常"
                                )

                    if fig_gantt and view_mode != "📊 品質分析":
                        # ── ヒストグラム詳細ページ用コンテキストを保存 ─────
                        st.session_state[f"_hist_ctx_{pname}"] = {
                            "step_stats": step_stats,
                            "steps_list": steps_list,
                            "result_df":  result_df,
                            "baseline":   st.session_state.get(pk(pname, "baseline"), {}),
                        }

                        # ── ヒストグラム URL マップ（ダブルクリック遷移用）──
                        _hist_urls = {
                            s["name"]: (
                                f"/histogram"
                                f"?proc={urllib.parse.quote(pname)}"
                                f"&step_name={urllib.parse.quote(s['name'])}"
                            )
                            for s in step_stats
                        }
                        _safe_pn = "".join(c if c.isalnum() else "_" for c in pname)
                        _gm_id   = f"gm_{_safe_pn}"

                        # ── マーカー div（JS がガントチャートを特定するため）──
                        st.markdown(f'<div id="{_gm_id}"></div>', unsafe_allow_html=True)

                        # ── ガントチャート（クリックでステップ詳細連動）──
                        gantt_event = st.plotly_chart(
                            fig_gantt, width="stretch",
                            key=f"gantt_{pname}", on_select="rerun",
                        )
                        try:
                            pts = (gantt_event.selection.get("points", [])
                                   if gantt_event and gantt_event.selection else [])
                            if pts:
                                clicked_y  = pts[0].get("y", "")
                                step_names = [s["name"] for s in step_stats]
                                if clicked_y in step_names:
                                    st.session_state[f"sel_step_{pname}"] = clicked_y
                        except Exception:
                            pass

                        # ── ダブルクリック → ヒストグラムを別タブで開く ──────
                        # マーカー div の次の stElementContainer 内 Plotly チャートを特定し
                        # 2 回クリック（< 450ms）で window.open('/histogram?...')
                        _urls_js = json.dumps(_hist_urls, ensure_ascii=False)
                        _flag    = f"_dbl_{_safe_pn}"
                        st.components.v1.html(f"""<script>
    (function(){{
      var URLS={_urls_js};
      var MID="{_gm_id}";
      // ダブルクリック状態を親ウィンドウに保持
      // → Streamlit の rerun でiframeがリロードされても状態が消えない
      if(!window.parent._dbl) window.parent._dbl={{}};
      if(!window.parent._dbl[MID]) window.parent._dbl[MID]={{t:0,n:""}};
      var _st=window.parent._dbl[MID];

      function findGd(){{
        var m=window.parent.document.getElementById(MID);
        if(!m) return null;
        var c=m;
        while(c&&(!c.getAttribute||c.getAttribute('data-testid')!=='stElementContainer'))
          c=c.parentElement;
        if(!c) return null;
        var sib=c.nextElementSibling;
        while(sib){{
          var gd=sib.querySelector('.js-plotly-plot');
          if(gd) return gd;
          sib=sib.nextElementSibling;
        }}
        return null;
      }}
      function attach(){{
        var gd=findGd();
        if(!gd||gd['{_flag}']) return;
        gd['{_flag}']=true;
        gd.on('plotly_click',function(ev){{
          if(!ev||!ev.points||!ev.points.length) return;
          var name=ev.points[0].data.name||"";
          var now=Date.now();
          // 500ms以内に同じステップを2回クリック → ヒストグラムページへ遷移
          if(now-_st.t<500&&name===_st.n&&URLS[name]){{
            window.parent.location.href=URLS[name];
            _st.t=0;
          }} else {{
            _st.t=now; _st.n=name;
          }}
        }});
      }}
      attach();
      setTimeout(attach,400);
      setTimeout(attach,1500);
    }})();
    </script>""", height=0)

                        # ── ステップ統計テーブル（B-3: Cpk含む）──────────
                        total_t = sum(s["mean"] for s in step_stats) or 1.0
                        rows = []
                        for s in step_stats:
                            mean_v = s["mean"]
                            std_v  = s["abs_std"]
                            # B-3: Cpk 計算 (USL=takt_target, LSL=0)
                            if takt_target > 0 and std_v > 0:
                                cpk = min(
                                    (takt_target - mean_v) / (3 * std_v),
                                    mean_v / (3 * std_v)
                                )
                                if cpk >= 1.33:
                                    cpk_str = f"🟢 {cpk:.2f}"
                                elif cpk >= 1.0:
                                    cpk_str = f"🟡 {cpk:.2f}"
                                else:
                                    cpk_str = f"🔴 {cpk:.2f}"
                            else:
                                cpk_str = "—"
                            row = {
                                "ステップ名": s["name"],
                                "モード":    "範囲" if s["mode"] == "range" else "単一",
                                "平均[ms]":  round(mean_v, 2),
                                "σ[ms]":    round(std_v, 2),
                                "min[ms]":  round(s.get("abs_min", s["min"]), 2),
                                "max[ms]":  round(s.get("abs_max", s["max"]), 2),
                                "タクト比率": f"{mean_v/total_t*100:.1f}%",
                                "Cpk":      cpk_str,
                            }
                            rows.append(row)
                        _stats_df = pd.DataFrame(rows)
                        _dl_col, _tbl_col = st.columns([1, 7])
                        with _dl_col:
                            st.download_button(
                                "📊 CSV",
                                data=result_df.to_csv(index=False, encoding="utf-8-sig"),
                                file_name=f"{pname}_cycles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                width="stretch",
                                key=f"dl_result_{pname}",
                                help="サイクル解析結果をCSVでダウンロード",
                            )
                        with _tbl_col:
                            st.dataframe(_stats_df, hide_index=True, width="stretch")
                        if takt_target == 0:
                            st.caption("💡 Cpk を表示するにはサイクル設定でタクト目標を設定してください")

                        # ── B-2: IQR外れ値検出 警告カード ───────────────
                        _outliers = detect_outliers_iqr(result_df, step_stats)
                        if _outliers:
                            for _ov in _outliers:
                                _cyc_str = ", ".join(str(c) for c in _ov["cycles"])
                                st.warning(
                                    f"⚠️ **{_ov['name']}**: サイクル {_cyc_str} で外れ値検出"
                                    f"（IQR境界: {_ov['lo']:.1f}〜{_ov['hi']:.1f} ms）"
                                )

                        if view_mode == "⚙️ 設定":
                            # ── 基準値管理 ──────────────────────────────────
                            st.divider()
                            _bl      = st.session_state.get(pk(pname, "baseline"), {})
                            _bl_meta = st.session_state.get(pk(pname, "baseline_meta"), {})
                            bl_l, bl_r = st.columns([7, 3])
                            with bl_l:
                                if _bl:
                                    st.success(
                                        f"📐 基準値登録済み — {_bl_meta.get('source','?')} / "
                                        f"{_bl_meta.get('n_cycles','?')} サイクル "
                                        f"({_bl_meta.get('created_at','')})"
                                    )
                                else:
                                    st.caption("📐 基準値を登録するとヒストグラムが差分表示になります")
                            with bl_r:
                                _bl_c1, _bl_c2 = st.columns(2)
                                with _bl_c1:
                                    if st.button(
                                        "📐 登録・編集",
                                        key=f"bl_btn_{pname}",
                                        width="stretch",
                                        disabled=not step_stats,
                                        help="現在のCSVの統計から基準値を登録",
                                    ):
                                        baseline_dialog(pname, step_stats, result_df)
                                with _bl_c2:
                                    _has_multi = len(st.session_state.get("csv_store", {})) > 1
                                    if st.button(
                                        "📂 複数CSV",
                                        key=f"bl_bulk_btn_{pname}",
                                        width="stretch",
                                        disabled=not (step_stats and _has_multi),
                                        help="複数の良品CSVから一括算出",
                                    ):
                                        bulk_baseline_dialog(pname, steps_list)

                            st.divider()

                            # ── ステップ詳細 ──────────────────────────────
                            st.markdown("**ステップ詳細**")
                            step_names = [s["name"] for s in step_stats]
                            cur_sel = st.session_state.get(f"sel_step_{pname}", step_names[0] if step_names else "")
                            if cur_sel not in step_names and step_names:
                                cur_sel = step_names[0]
                            sel_name = st.selectbox(
                                "詳細を見るステップ", step_names,
                                index=step_names.index(cur_sel) if cur_sel in step_names else 0,
                                key=f"sel_step_{pname}",
                                label_visibility="collapsed",
                            )
                            sel_stat = next((s for s in step_stats if s["name"] == sel_name), None)
                            sel_step = next((s for s in steps_list if s.get("name") == sel_name), None)
                            if sel_stat and sel_step:
                                render_step_detail(df, trigger_col, edge,
                                                   sel_stat, sel_step, pname, result_df)

            # ── 複数CSV 比較テーブル（監視モード）──────────────────
            _csv_store = st.session_state.get("csv_store", {})
            _store_keys = [k for k in _csv_store if k != "sample_playback.csv" or len(_csv_store) == 1]
            if view_mode == "👁️ 監視" and len(_csv_store) > 1 and steps_list:
                st.divider()
                st.markdown("**📊 CSV 横断比較**")
                _steps_json_cmp = json.dumps(steps_list, ensure_ascii=False, sort_keys=True)
                _bl_base = st.session_state.get(pk(pname, "baseline"), {})
                _cmp_rows = []
                _all_csv_step_means: dict = {}  # {csv_key: {step_name: mean_ms}}

                for _ck, _centry in _csv_store.items():
                    _cdf = _centry.get("df", pd.DataFrame())
                    _clbl = _centry.get("label", _ck)
                    if _cdf.empty:
                        continue
                    try:
                        _crd = cached_analyze_v2(_cdf, trigger_col, edge, _steps_json_cmp)
                    except Exception:
                        continue
                    if _crd is None or len(_crd) == 0:
                        continue
                    _, _css = build_gantt_v2(_crd, steps_list, takt_target)
                    _step_means = {s["name"]: s["abs_mean"] for s in _css}
                    _all_csv_step_means[_clbl] = _step_means

                if _all_csv_step_means:
                    _snames = [s["name"] for s in step_stats]
                    _cmp_cols = list(_all_csv_step_means.keys())
                    _cmp_data = {"ステップ": _snames}
                    for _clbl, _smeans in _all_csv_step_means.items():
                        _cmp_data[_clbl] = [round(_smeans.get(sn, float("nan")), 1) for sn in _snames]
                    # 全CSV平均と基準差分
                    _cmp_data["全体平均[ms]"] = []
                    _cmp_data["vs 基準[ms]"] = []
                    for _si, sn in enumerate(_snames):
                        _vals = [_all_csv_step_means[c].get(sn) for c in _all_csv_step_means
                                 if _all_csv_step_means[c].get(sn) is not None]
                        _avg = round(float(np.mean(_vals)), 1) if _vals else float("nan")
                        _cmp_data["全体平均[ms]"].append(_avg)
                        # vs 基準（baseline は {step_name: {ref_dur_ms, ref_start_ms}} 形式）
                        _bl_step = _bl_base.get(sn, {}) if _bl_base else {}
                        if _bl_step and not np.isnan(_avg):
                            _s_mode = step_stats[_si].get("mode", "single") if _si < len(step_stats) else "single"
                            _bl_ref = (
                                _bl_step.get("ref_start_ms", 0) + _bl_step.get("ref_dur_ms", 0)
                                if _s_mode != "single"
                                else _bl_step.get("ref_ms", 0)
                            )
                            _cmp_data["vs 基準[ms]"].append(round(_avg - _bl_ref, 1))
                        else:
                            _cmp_data["vs 基準[ms]"].append(None)
                    st.dataframe(pd.DataFrame(_cmp_data), hide_index=True, width="stretch")
                    if not _bl_base:
                        st.caption("💡 基準値を登録すると「vs 基準」列が表示されます")

                    # ── 比較バーチャート ─────────────────────────────
                    if len(_all_csv_step_means) >= 1 and _snames:
                        st.markdown("**📊 ステップ別平均時刻 比較チャート**")
                        _cmp_colors = [
                            "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
                            "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
                        ]
                        _fig_cmp = go.Figure()
                        for _ci, (_clbl2, _smeans2) in enumerate(_all_csv_step_means.items()):
                            _fig_cmp.add_trace(go.Bar(
                                name=_clbl2,
                                x=_snames,
                                y=[_smeans2.get(sn, 0) for sn in _snames],
                                marker_color=_cmp_colors[_ci % len(_cmp_colors)],
                                opacity=0.82,
                            ))
                        # 基準値をダイヤモンドマーカー付き折れ線で重ねる
                        if _bl_base:
                            _bl_refs2 = []
                            for _si2, sn2 in enumerate(_snames):
                                _bls2 = _bl_base.get(sn2, {})
                                if _bls2:
                                    _sm2 = step_stats[_si2].get("mode", "single") if _si2 < len(step_stats) else "single"
                                    _blr2 = (
                                        _bls2.get("ref_start_ms", 0) + _bls2.get("ref_dur_ms", 0)
                                        if _sm2 != "single"
                                        else _bls2.get("ref_ms", 0)
                                    )
                                    _bl_refs2.append(_blr2)
                                else:
                                    _bl_refs2.append(None)
                            _fig_cmp.add_trace(go.Scatter(
                                x=_snames,
                                y=_bl_refs2,
                                mode="markers+lines",
                                name="📐 基準値",
                                line=dict(color="#8e44ad", width=2, dash="dot"),
                                marker=dict(size=11, symbol="diamond", color="#8e44ad"),
                            ))
                        _fig_cmp.update_layout(
                            barmode="group",
                            height=320,
                            margin=dict(t=20, b=40, l=60, r=20),
                            yaxis_title="時刻 [ms]",
                            legend=dict(
                                orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1,
                            ),
                        )
                        st.plotly_chart(_fig_cmp, width="stretch",
                                        key=f"cmp_bar_{pname}")

            # ── 詳細解析タブ（補助）─── 全モードで表示、監視モードは最初から開く ──
            if steps_list and len(result_df) > 0:
                _dtab_expanded = (view_mode == "👁️ 監視")
                _dtab_label = "📊 ヒストグラム" if view_mode == "👁️ 監視" else "📊 詳細解析タブ"
                with st.expander(_dtab_label, expanded=_dtab_expanded):
                    delay_cols = [c for c in result_df.columns
                                  if c.endswith("_遅れ[ms]") or c.endswith("_dur[ms]")]
                    thr_tab = st.number_input(
                        "閾値[ms]（0=なし）", min_value=0.0, value=0.0, step=0.5,
                        key=f"thr_tab_{pname}",
                    )
                    try:
                        cycle_starts = cached_detect_cycles(df, trigger_col, edge)
                    except Exception:
                        cycle_starts = []

                    # 監視モードではヒストグラムを先頭タブに
                    if view_mode == "👁️ 監視":
                        tabs = st.tabs(["ヒストグラム", "サイクル一覧", "時系列波形", "📈 トレンド"])
                        _tab_hist, _tab_cyc, _tab_ts, _tab_trend = tabs
                    else:
                        tabs = st.tabs(["サイクル一覧", "ヒストグラム", "時系列波形", "📈 トレンド"])
                        _tab_cyc, _tab_hist, _tab_ts, _tab_trend = tabs

                    with _tab_cyc:
                        st.dataframe(result_df, width="stretch")
                        st.download_button("CSVダウンロード",
                                           result_df.to_csv(index=False, encoding="utf-8-sig"),
                                           f"{pname}_cycles.csv", key=f"dl_csv_{pname}")

                    with _tab_hist:
                        # 基準データ（ref CSV）を取得 ─ 比較時のオーバーレイ用
                        _h_ref_key   = st.session_state.get("ref_csv_key", "")
                        _h_act_key   = st.session_state.get("active_csv", "")
                        _h_csv_store = st.session_state.get("csv_store", {})
                        _h_has_ref   = (
                            bool(_h_ref_key)
                            and _h_ref_key != _h_act_key
                            and _h_ref_key in _h_csv_store
                        )
                        _h_ref_result: dict = {}      # {col: ndarray}
                        if _h_has_ref:
                            try:
                                _h_ref_df = _h_csv_store[_h_ref_key]["df"]
                                _h_ref_res = cached_analyze_v2(
                                    _h_ref_df, trigger_col, edge, json.dumps(steps_list)
                                )
                                for _hc in delay_cols:
                                    if _hc in _h_ref_res.columns:
                                        _hv = _h_ref_res[_hc].dropna().values
                                        if len(_hv) > 0:
                                            _h_ref_result[_hc] = _hv
                            except Exception:
                                _h_has_ref = False

                        _h_baseline = st.session_state.get(pk(pname, "baseline"), {})

                        for col in delay_cols:
                            vn    = col.replace("_遅れ[ms]", "").replace("_dur[ms]", " (所要時間)")
                            dl    = result_df[col].dropna().values
                            if len(dl) == 0:
                                continue
                            _bkey_t = f"{pname}_{col}_t"
                            nb  = calc_nice_bins(dl, _bkey_t)
                            _vmin_t = float(dl.min()); _vmax_t = float(dl.max())
                            _bsz_t  = (_vmax_t - _vmin_t) / nb if nb > 0 and _vmax_t > _vmin_t else 1.0
                            _xbins_t = dict(start=_vmin_t, end=_vmax_t + _bsz_t, size=_bsz_t)
                            st_ = calc_statistics(dl)
                            sg3 = st_.get("3σ上限[ms]", 0)

                            # 基準値（登録済み mean / std）
                            _bl_step_h = col.replace("_遅れ[ms]", "").replace("_dur[ms]", "")
                            _bl_entry_h = _h_baseline.get(_bl_step_h, {})
                            _bl_ref_ms  = _bl_entry_h.get("ref_ms")
                            _bl_std_ms  = _bl_entry_h.get("std_ms", 0.0)

                            fig = go.Figure()

                            # ── 基準CSV の分布オーバーレイ ─────────────────
                            if _h_has_ref and col in _h_ref_result:
                                _ref_dl = _h_ref_result[col]
                                # 範囲を統合してビン幅を揃える
                                _vmin_t = min(_vmin_t, float(_ref_dl.min()))
                                _vmax_t = max(_vmax_t, float(_ref_dl.max()))
                                _bsz_t  = (_vmax_t - _vmin_t) / nb if nb > 0 and _vmax_t > _vmin_t else 1.0
                                _xbins_t = dict(start=_vmin_t, end=_vmax_t + _bsz_t, size=_bsz_t)
                                fig.add_trace(go.Histogram(
                                    x=_ref_dl,
                                    xbins=_xbins_t,
                                    name="基準データ分布",
                                    marker_color="rgba(30,120,220,0.55)",
                                    opacity=0.85,
                                ))

                            # ── 現在データの分布 ────────────────────────────
                            if thr_tab > 0:
                                bl = dl[dl <= thr_tab]
                                ab = dl[dl >  thr_tab]
                                if len(bl):
                                    fig.add_trace(go.Histogram(x=bl, xbins=_xbins_t, name="閾値以内",
                                                               marker_color="rgba(59,130,246,0.72)", opacity=0.88))
                                if len(ab):
                                    fig.add_trace(go.Histogram(x=ab, xbins=_xbins_t, name="閾値超過",
                                                               marker_color="rgba(239,68,68,0.72)", opacity=0.88))
                                fig.add_vline(x=thr_tab, line_dash="dash", line_color="#7c3aed",
                                              annotation_text=f"閾値 {thr_tab}ms")
                            else:
                                fig.add_trace(go.Histogram(x=dl, xbins=_xbins_t,
                                                           name="現在データ",
                                                           marker_color="rgba(59,130,246,0.72)", opacity=0.88))

                            # ── 垂直線: 3σ上限 ──────────────────────────────
                            fig.add_vline(x=sg3, line_dash="dot", line_color="#9ca3af",
                                          annotation_text=f"3σ {sg3:.1f}ms")

                            # ── 垂直線: 基準平均（ダークアンバー）＋ ±1σ 帯 ─
                            if _bl_ref_ms is not None:
                                if _bl_std_ms > 0:
                                    fig.add_vrect(
                                        x0=_bl_ref_ms - _bl_std_ms,
                                        x1=_bl_ref_ms + _bl_std_ms,
                                        fillcolor="rgba(245,158,11,0.12)",
                                        line_width=0,
                                    )
                                fig.add_vline(
                                    x=_bl_ref_ms,
                                    line_color="#b45309", line_width=2.5,
                                    annotation_text=f"基準 {_bl_ref_ms:.1f}ms",
                                    annotation_font_color="#b45309",
                                    annotation_position="top left",
                                )

                            fig.update_layout(title=vn, xaxis_title="時間[ms]",
                                              barmode="overlay", height=260, margin=dict(t=32))
                            st.plotly_chart(fig, width="stretch", key=f"t2_{pname}_{vn}")
                            st.slider("ビン数", 3, 60, nb, key=f"_bins_{_bkey_t}",
                                      help="ヒストグラムのビン数を手動調整（Freedman-Diaconis による自動算出が既定値）")
                            if st_:
                                sc = st.columns(min(6, len(st_)))
                                for i, (k, v) in enumerate(list(st_.items())[:6]):
                                    sc[i].metric(k, v)
                            st.markdown("---")

                    with _tab_ts:
                        all_vars_disp = steps_all_vars(steps_list, bool_cols)
                        ts_b = st.multiselect("Bool変数", bool_cols,
                                              default=all_vars_disp[:3],
                                              key=f"ts_b_{pname}")
                        ts_n = st.multiselect("数値変数（第2Y軸）", num_cols,
                                              default=[], key=f"ts_n_{pname}")
                        if ts_b or ts_n:
                            fig = go.Figure()
                            for var in ts_b:
                                fig.add_trace(go.Scatter(
                                    x=df["Timestamp"],
                                    y=normalize_bool_series(df[var]),
                                    name=var, fill="tozeroy", mode="lines",
                                    line=dict(width=1)))
                            for var in ts_n:
                                fig.add_trace(go.Scatter(
                                    x=df["Timestamp"], y=df[var],
                                    name=var, mode="lines", yaxis="y2"))
                            if ts_n:
                                fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
                            for idx_c in cycle_starts[:50]:
                                if idx_c in df.index:
                                    fig.add_vline(x=df.loc[idx_c, "Timestamp"],
                                                  line=dict(color="green", width=0.5, dash="dash"))
                            fig.update_layout(xaxis_title="時刻", height=420)
                            st.plotly_chart(fig, width="stretch", key=f"t3_{pname}")

                    with _tab_trend:
                        # B-1: トレンドチャート
                        _trend_step_stats = []
                        try:
                            _, _trend_step_stats = build_gantt_v2(result_df, steps_list, 0)
                        except Exception:
                            pass
                        _trend_fig = build_trend_chart(result_df, _trend_step_stats, takt_target)
                        if _trend_fig:
                            st.caption(
                                "各サイクルの遅れ時間推移。赤丸は外れ値（|値-平均| > 2σ）、"
                                "点線は mean+3σ 上限。"
                            )
                            st.plotly_chart(_trend_fig, width="stretch",
                                            key=f"trend_{pname}")
                        else:
                            st.info("トレンドを表示するにはステップを追加してください")

        st.markdown("")

    if False:  # 工程間タイムライン比較（削除）
        with st.expander("", expanded=False):
            proc_names = list(processes.keys())
            selected_procs = st.multiselect(
                "比較する工程",
                proc_names,
                default=proc_names,
                key="cmp_procs_sel",
            )

            if selected_procs:
                # 各工程のステップ統計を収集
                all_bars: list[dict] = []
                takt_lines: list[tuple] = []   # (takt_ms, proc_label)

                for cmp_pname in selected_procs:
                    cmp_steps  = st.session_state.get(pk(cmp_pname, "steps_list"), [])
                    if not cmp_steps:
                        continue
                    cmp_trig   = st.session_state.get(pk(cmp_pname, "trigger"), bool_cols[0])
                    cmp_edge   = st.session_state.get(pk(cmp_pname, "edge"), "RISE")
                    cmp_takt   = int(st.session_state.get(pk(cmp_pname, "takt"), 0))

                    try:
                        cmp_json = json.dumps(cmp_steps, ensure_ascii=False, sort_keys=True)
                        cmp_res  = cached_analyze_v2(df, cmp_trig, cmp_edge, cmp_json)
                    except Exception:
                        continue
                    if cmp_res is None or len(cmp_res) == 0:
                        continue

                    _, cmp_stats = build_gantt_v2(cmp_res, cmp_steps, 0)
                    for s in cmp_stats:
                        all_bars.append({
                            **s,
                            "label": f"{cmp_pname}　/　{s['name']}",
                            "proc":  cmp_pname,
                        })
                    if cmp_takt > 0:
                        takt_lines.append((cmp_takt, cmp_pname))

                if all_bars:
                    fig_cmp = go.Figure()
                    for b in all_bars:
                        is_range = (b.get("mode") == "range")
                        if is_range:
                            ht = (
                                f"<b>%{{y}}</b> [範囲]<br>"
                                f"開始: {b.get('abs_start', b['start']):.1f}ms<br>"
                                "長さ（平均）: %{customdata[1]:.1f} ms<br>"
                                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms"
                                "<extra></extra>"
                            )
                        else:
                            ht = (
                                "<b>%{y}</b><br>"
                                "区間長（平均）: %{customdata[1]:.1f} ms<br>"
                                "ばらつき: min %{customdata[2]:.1f} / max %{customdata[3]:.1f} ms"
                                "<extra></extra>"
                            )
                        fig_cmp.add_trace(go.Bar(
                            name=b["label"],
                            y=[b["label"]], x=[b["mean"]], base=[b["start"]],
                            orientation="h", width=0.6,
                            marker_color=b["color"],
                            marker_pattern_shape="/" if is_range else "",
                            customdata=[[b["start"], b["mean"], b["min"], b["max"]]],
                            hovertemplate=ht,
                        ))
                        # ばらつき帯
                        bw = b["max"] - b["min"]
                        if bw > 0:
                            fig_cmp.add_trace(go.Bar(
                                y=[b["label"]], x=[bw], base=[b["start"] + b["min"]],
                                orientation="h", width=0.6,
                                marker_color=b["color"], opacity=0.18,
                                showlegend=False, hoverinfo="skip",
                            ))

                    # タクト目標ライン（工程ごとに色分け）
                    takt_colors = ["red", "darkorange", "purple", "brown"]
                    for ti, (takt_ms, takt_proc) in enumerate(takt_lines):
                        fig_cmp.add_vline(
                            x=takt_ms,
                            line_dash="dash",
                            line_color=takt_colors[ti % len(takt_colors)],
                            annotation_text=f"{takt_proc} 目標 {takt_ms}ms",
                            annotation_position="top",
                        )

                    fig_cmp.update_layout(
                        barmode="overlay",
                        xaxis_title="サイクル開始からの経過時間 [ms]",
                        yaxis=dict(autorange="reversed"),
                        height=max(300, len(all_bars) * 50 + 120),
                        margin=dict(l=0, r=20, t=16, b=40),
                        showlegend=False,
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_cmp, width="stretch",
                                    key="cmp_gantt_multi")

                    # 工程別タクト消費サマリー
                    st.markdown("**工程別タクト消費サマリー**")
                    rows_cmp = []
                    for cmp_pname in selected_procs:
                        proc_bars = [b for b in all_bars if b["proc"] == cmp_pname]
                        if not proc_bars:
                            continue
                        total_ms  = sum(b["mean"] for b in proc_bars)
                        cmp_takt  = int(st.session_state.get(pk(cmp_pname, "takt"), 0))
                        usage_pct = f"{total_ms / cmp_takt * 100:.1f}%" if cmp_takt > 0 else "—"
                        rows_cmp.append({
                            "工程名":         cmp_pname,
                            "ステップ数":     len(proc_bars),
                            "合計時間[ms]":   round(total_ms, 1),
                            "タクト目標[ms]": cmp_takt if cmp_takt > 0 else "—",
                            "タクト消費率":   usage_pct,
                        })
                    if rows_cmp:
                        st.dataframe(pd.DataFrame(rows_cmp), hide_index=True,
                                     width="stretch")
                else:
                    st.info("比較対象の工程にステップが設定されていません。")

# ═══════════════════════════════════════════════════════════════
# 傾向解析タブ
# ═══════════════════════════════════════════════════════════════

with _page_tabs[1]:
    import re as _re

    st.subheader("📈 傾向解析")
    st.caption("複数時期のCSVを登録し、ステップ毎のタイミング傾向を可視化します")

    if not processes:
        st.info("「⚙️ 画面設定」タブで工程を設定してください")
    else:
        _tr_pname = st.selectbox(
            "工程を選択", list(processes.keys()), key="tr_page_proc"
        )
        _tr_trigger = st.session_state.get(pk(_tr_pname, "trigger"), bool_cols[0] if bool_cols else "")
        _tr_edge    = st.session_state.get(pk(_tr_pname, "edge"), "RISE")
        _tr_steps   = st.session_state.get(pk(_tr_pname, "steps_list"), [])
        _tr_bl      = st.session_state.get(pk(_tr_pname, "baseline"), {})

        if not _tr_steps:
            st.info("ステップが設定されていません。「⚙️ 画面設定」タブでステップを追加してください")
        else:
            # ── 解析対象CSV: サイドバーの csv_store から選択 ─────────
            _tr_csv_store = st.session_state.get("csv_store", {})
            if not _tr_csv_store:
                st.info("サイドバーの「② 比較CSV」にCSVをアップロードしてください")
            else:
                _tr_all_keys = list(_tr_csv_store.keys())
                _tr_sel = st.multiselect(
                    "解析対象CSV（複数選択可）",
                    _tr_all_keys,
                    default=_tr_all_keys,
                    format_func=lambda k: (
                        "📌 " + _tr_csv_store[k].get("label", k)
                        if _tr_csv_store[k].get("is_ref")
                        else "📊 " + _tr_csv_store[k].get("label", k)
                    ),
                    key=f"tr_sel_{_tr_pname}",
                )

            # ラベル管理（csv_store の label を初期値に）
            _tr_labels_key = f"_tr_labels_{_tr_pname}"
            if _tr_labels_key not in st.session_state:
                st.session_state[_tr_labels_key] = {}

            # 重複を除去（multiselect の session_state がズレたとき対策）
            _tr_sel_list = list(dict.fromkeys(locals().get("_tr_sel", [])))

            # ── 並び順を session_state で管理 ────────────────────
            _tr_order_key = f"tr_order_{_tr_pname}"
            _cur_ord = st.session_state.get(_tr_order_key, [])
            _synced  = [k for k in _cur_ord if k in _tr_sel_list]
            _synced += [k for k in _tr_sel_list if k not in _synced]
            st.session_state[_tr_order_key] = _synced
            _tr_sel_list = _synced

            if _tr_sel_list:
                with st.expander("ラベル設定（グラフX軸に使用）", expanded=False):
                    _n_cols = min(len(_tr_sel_list), 4)
                    _label_cols = st.columns(_n_cols)
                    for _ti, _tk in enumerate(_tr_sel_list):
                        if _tk not in st.session_state[_tr_labels_key]:
                            st.session_state[_tr_labels_key][_tk] = (
                                _tr_csv_store.get(_tk, {}).get("label", _tk)
                            )
                        # キーに特殊文字・長パスが含まれても衝突しないようハッシュ化
                        _tk_wkey = hashlib.md5(_tk.encode()).hexdigest()[:12]
                        with _label_cols[_ti % _n_cols]:
                            st.session_state[_tr_labels_key][_tk] = st.text_input(
                                os.path.basename(_tk) if os.sep in _tk else _tk,
                                value=st.session_state[_tr_labels_key][_tk],
                                key=f"tr_lbl_{_tr_pname}_{_tk_wkey}",
                            )

                # ── CSV 並び順変更 ────────────────────────────────────
                if len(_tr_sel_list) > 1:
                    with st.expander("↕️ CSV の並び順を変更", expanded=False):
                        for _oi in range(len(_tr_sel_list)):
                            _ok = _tr_sel_list[_oi]
                            _oc1, _oc2, _oc3 = st.columns([6, 1, 1])
                            _lbl_d = _tr_csv_store.get(_ok, {}).get(
                                "label", os.path.basename(_ok) if os.sep in _ok else _ok
                            )
                            with _oc1:
                                st.write(f"**{_oi + 1}.** {_lbl_d}")
                            with _oc2:
                                if _oi > 0:
                                    if st.button("↑", key=f"tr_up_{_tr_pname}_{_oi}"):
                                        _ord = st.session_state[_tr_order_key]
                                        _ord[_oi - 1], _ord[_oi] = _ord[_oi], _ord[_oi - 1]
                                        st.rerun()
                            with _oc3:
                                if _oi < len(_tr_sel_list) - 1:
                                    if st.button("↓", key=f"tr_dn_{_tr_pname}_{_oi}"):
                                        _ord = st.session_state[_tr_order_key]
                                        _ord[_oi], _ord[_oi + 1] = _ord[_oi + 1], _ord[_oi]
                                        st.rerun()

                # ── 波形検査 設定確認パネル ─────────────────────────────
                # ※ 傾向解析対象は時間軸検出点のみ（XY は今後対応）
                _wi_trig_pre = st.session_state.get(
                    "wi_trigger", bool_cols[0] if bool_cols else "（未設定）"
                )
                _wi_edge_pre = st.session_state.get("wi_edge", "RISE")
                _wi_pre_items: list = []    # {var, idx, type, id, warn}
                for _pv in num_cols:
                    _pvkey = f"wvol___global_{_pv}"
                    for _pi, _pdet in enumerate(
                            st.session_state.get(f"{_pvkey}_t_det_list", [])):
                        _pdid  = _pdet.get("id", "")
                        _ptype = _pdet.get("type", "")
                        if st.session_state.get(
                                f"{_pvkey}_{_pdid}_trend_on", False):
                            # 傾き変化点で閾値=0 は実際には検出しない
                            _pwarn = ""
                            if _ptype == "傾き変化点":
                                _pth = float(st.session_state.get(
                                    f"{_pvkey}_{_pdid}_thresh", 0.0))
                                if _pth <= 0.0:
                                    _pwarn = "⚠️ 閾値=0（検出されません）"
                            _wi_pre_items.append({
                                "var": _pv, "idx": _pi + 1,
                                "type": _ptype, "id": _pdid,
                                "warn": _pwarn,
                            })
                    # XY 検出点も傾向解析に対応したので _wi_pre_items に追加
                    for _pi, _pdet in enumerate(
                            st.session_state.get(f"{_pvkey}_xy_det_list", [])):
                        _pdid  = _pdet.get("id", "")
                        _ptype = _pdet.get("type", "")
                        if st.session_state.get(f"{_pvkey}_{_pdid}_trend_on", False):
                            _xvar_conf = st.session_state.get(f"{_pvkey}_xy_xvar", "（X未設定）")
                            _wi_pre_items.append({
                                "var": _pv, "idx": _pi + 1,
                                "type": _ptype, "id": _pdid,
                                "warn": "" if _xvar_conf != "（X未設定）" else "⚠️ X変数未選択",
                                "is_xy": True,
                            })

                _wi_pre_ok = sum(
                    1 for x in _wi_pre_items if not x["warn"])
                with st.expander(
                    f"🔍 波形検査 設定確認　"
                    f"トリガー: **{_wi_trig_pre}** / {_wi_edge_pre}　"
                    f"傾向送出: **{_wi_pre_ok}** 件"
                    + (f"（⚠️ {len(_wi_pre_items)-_wi_pre_ok} 件 閾値=0）"
                       if len(_wi_pre_items) > _wi_pre_ok else ""),
                    expanded=(len(_wi_pre_items) == 0),
                ):
                    if _wi_pre_items:
                        for _px in _wi_pre_items:
                            _icon = "📈" if not _px["warn"] else "⚠️"
                            st.markdown(
                                f"{_icon} **{_px['var']}** "
                                f"#{_px['idx']} {_px['type']}"
                                + (f"　{_px['warn']}" if _px["warn"] else "")
                            )
                        st.caption(
                            "設定を変更した場合は **「📊 傾向解析を実行」を再クリック** してください。"
                        )
                    else:
                        st.info(
                            "傾向解析に送出する波形検査の検出点がありません。\n\n"
                            "「🔍 波形検査」タブ → **「⏱ 時間軸」タブ** または "
                            "**「XY グラフ」タブ** で"
                            "検出点を追加し、**📈 傾向解析に出す** をオンに"
                            "してください。"
                        )

                # ── 解析実行 ────────────────────────────────────────
                st.divider()
                if st.button(
                    "📊 傾向解析を実行", type="primary",
                    width="stretch", key=f"tr_run_{_tr_pname}",
                ):
                    _tr_steps_json = json.dumps(_tr_steps)
                    _res_list = []
                    _prog = st.progress(0, text="解析中...")
                    for _ti, _tk in enumerate(_tr_sel_list):
                        _prog.progress(
                            (_ti + 1) / len(_tr_sel_list),
                            text=f"解析中 {_ti + 1}/{len(_tr_sel_list)}: {_tk}",
                        )
                        try:
                            _tdf = _tr_csv_store[_tk]["df"]   # ← csv_store から直接取得
                            _tres = cached_analyze_v2(
                                _tdf, _tr_trigger, _tr_edge, _tr_steps_json
                            )
                            _lbl = st.session_state[_tr_labels_key].get(_tk, _tk)
                            # 波形NG統計
                            _wv_stats_tr: dict = {}
                            for _step_cfg_tr in _tr_steps:
                                if _step_cfg_tr.get("waveform_vars"):
                                    _sn_tr = _step_cfg_tr.get("name", "")
                                    try:
                                        _wv_stats_tr[_sn_tr] = _compute_wv_ng(
                                            _tdf, _tr_trigger, _tr_edge,
                                            _step_cfg_tr, _tr_pname, _tres,
                                        )
                                    except Exception:
                                        pass
                            _wi_det_stats_tr: dict = {}
                            _wi_err_msg: str = ""
                            try:
                                _wi_trig = st.session_state.get(
                                    "wi_trigger",
                                    bool_cols[0] if bool_cols else "",
                                )
                                _wi_det_stats_tr = _compute_wi_det_stats_for_csv(
                                    _tdf,
                                    _wi_trig,
                                    st.session_state.get("wi_edge", "RISE"),
                                    [c for c in num_cols if c in _tdf.columns],
                                    st.session_state,
                                )
                            except Exception as _wi_exc:
                                _wi_err_msg = str(_wi_exc)
                            _res_list.append(
                                {"label": _lbl, "fname": _tk,
                                 "result": _tres, "wv_stats": _wv_stats_tr,
                                 "wi_det_stats": _wi_det_stats_tr,
                                 "wi_err": _wi_err_msg}
                            )
                        except Exception as _te:
                            st.warning(f"{_tk}: 解析失敗 ({_te})")
                    _prog.empty()
                    st.session_state[f"_tr_results_{_tr_pname}"] = _res_list
                    st.toast(f"✅ {len(_res_list)} ファイルの解析が完了しました")
                    st.rerun()

                # ── 結果表示 ────────────────────────────────────────
                _tr_res_list = st.session_state.get(f"_tr_results_{_tr_pname}", [])
                if _tr_res_list:
                    st.divider()

                    # ── 波形検査エラー表示 ────────────────────────────
                    _wi_errs = [(r["label"], r["wi_err"])
                                for r in _tr_res_list if r.get("wi_err")]
                    if _wi_errs:
                        with st.expander("⚠️ 波形検査解析エラー", expanded=True):
                            for _el, _em in _wi_errs:
                                st.error(f"**{_el}**: {_em}")

                    # ── 数式検出点 診断警告 ───────────────────────────
                    _fm_warn_all: list = []
                    for _r in _tr_res_list:
                        for _fw in _r.get("wi_det_stats", {}).get("__formula_warns__", []):
                            _entry = f"[{_r['label']}] {_fw}"
                            if _entry not in _fm_warn_all:
                                _fm_warn_all.append(_entry)
                    if _fm_warn_all:
                        with st.expander(
                            f"🧮 数式検出点 診断 ({len(_fm_warn_all)} 件)",
                            expanded=True,
                        ):
                            for _fw in _fm_warn_all:
                                st.warning(_fw)

                    # ── チャートタイプ切り替え ────────────────────────
                    _chart_mode = st.radio(
                        "表示形式",
                        ["📈 傾向チャート（平均 ± σ）", "📊 Xbar-R 管理図"],
                        horizontal=True,
                        key=f"tr_chart_mode_{_tr_pname}",
                    )

                    _tr_summary_rows = []

                    # ══════════════════════════════════════════════════
                    # A) 傾向チャート（従来）
                    # ══════════════════════════════════════════════════
                    if _chart_mode == "📈 傾向チャート（平均 ± σ）":
                        st.markdown("#### 📈 傾向チャート")
                        _tr_sigma = st.slider(
                            "異常判定 σ 倍数", 1.0, 5.0, 3.0, 0.5,
                            key=f"tr_sigma_{_tr_pname}",
                            help="基準値±Nσ を超えたポイントを 🔴 でマーク",
                        )

                        for _ts in _tr_steps:
                            _tsn  = _ts["name"]
                            _tsm  = _ts["mode"]
                            _tsc  = _ts.get("color", "#1f77b4")
                            _dcol = (
                                f"{_tsn}_遅れ[ms]" if _tsm == "single"
                                else f"{_tsn}_dur[ms]"
                            )

                            _tr_lbls, _tr_means, _tr_stds = [], [], []
                            for _r in _tr_res_list:
                                _rdf = _r["result"]
                                if _dcol not in _rdf.columns:
                                    continue
                                _rv = _rdf[_dcol].dropna().values
                                if len(_rv) == 0:
                                    continue
                                _tr_lbls.append(_r["label"])
                                _tr_means.append(float(np.mean(_rv)))
                                _tr_stds.append(float(np.std(_rv)))

                            if not _tr_lbls:
                                continue

                            _tbl  = _tr_bl.get(_tsn, {})
                            _tref = (
                                _tbl.get("ref_ms") if _tsm == "single"
                                else _tbl.get("ref_dur_ms")
                            )
                            _tbls = (
                                _tbl.get("std_ms", 0.0) if _tsm == "single"
                                else _tbl.get("std_dur_ms", 0.0)
                            )

                            _is_ng = [
                                bool(
                                    _tref is not None and _tbls > 0
                                    and abs(m - _tref) > _tr_sigma * _tbls
                                )
                                for m in _tr_means
                            ]

                            _sr = {"ステップ": _tsn, "モード": _tsm}
                            for _li, (_lbl, _m, _s) in enumerate(
                                zip(_tr_lbls, _tr_means, _tr_stds)
                            ):
                                _sr[_lbl] = f"{'🔴 ' if _is_ng[_li] else ''}{_m:.1f} ± {_s:.1f}"
                            _tr_summary_rows.append(_sr)

                            _fig_tr = go.Figure()
                            _hex = _tsc.lstrip("#")
                            _rgb = tuple(int(_hex[i:i+2], 16) for i in (0, 2, 4)) if len(_hex) == 6 else (31, 119, 180)
                            _fill_col = f"rgba({_rgb[0]},{_rgb[1]},{_rgb[2]},0.15)"
                            _fig_tr.add_trace(go.Scatter(
                                x=_tr_lbls + _tr_lbls[::-1],
                                y=(
                                    [m + s for m, s in zip(_tr_means, _tr_stds)]
                                    + [m - s for m, s in zip(_tr_means[::-1], _tr_stds[::-1])]
                                ),
                                fill="toself", fillcolor=_fill_col,
                                line=dict(width=0), showlegend=True, name="±1σ",
                            ))
                            _pt_colors  = ["#e74c3c" if ng else _tsc for ng in _is_ng]
                            _pt_symbols = ["x" if ng else "circle" for ng in _is_ng]
                            _fig_tr.add_trace(go.Scatter(
                                x=_tr_lbls, y=_tr_means,
                                mode="lines+markers",
                                line=dict(color=_tsc, width=2),
                                marker=dict(color=_pt_colors, size=10,
                                            symbol=_pt_symbols,
                                            line=dict(width=2, color="white")),
                                name="平均値",
                                hovertemplate="%{x}<br>平均: %{y:.2f} ms<br>σ: %{customdata:.2f} ms<extra></extra>",
                                customdata=_tr_stds,
                            ))
                            if _tref is not None:
                                _fig_tr.add_hline(
                                    y=_tref, line_dash="dash", line_color="#27ae60",
                                    annotation_text=f"基準 {_tref:.1f} ms",
                                    annotation_position="bottom right",
                                )
                                if _tbls > 0:
                                    _fig_tr.add_hrect(
                                        y0=_tref - _tr_sigma * _tbls,
                                        y1=_tref + _tr_sigma * _tbls,
                                        fillcolor="#27ae60", opacity=0.08, line_width=0,
                                        annotation_text=f"±{_tr_sigma:.0f}σ",
                                        annotation_position="top right",
                                    )
                            _fig_tr.update_layout(
                                title=dict(text=f"<b>{_tsn}</b>　傾向", font=dict(size=14)),
                                xaxis_title="時期", yaxis_title="[ms]",
                                height=290, margin=dict(t=48, b=40, l=60, r=20),
                                legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
                                hovermode="x unified",
                            )
                            st.plotly_chart(_fig_tr, width="stretch")

                    # ══════════════════════════════════════════════════
                    # B) Xbar-R 管理図
                    # ══════════════════════════════════════════════════
                    else:
                        st.markdown("#### 📊 Xbar-R 管理図")
                        st.caption(
                            "各時期のデータをサブグループとして管理限界線（UCL/CL/LCL）を算出。"
                            "n ≤ 25 は Xbar-R、n > 25 は Xbar-S（s̄ベース）で算出します。"
                        )

                        for _ts in _tr_steps:
                            _tsn  = _ts["name"]
                            _tsm  = _ts["mode"]
                            _tsc  = _ts.get("color", "#1f77b4")
                            _dcol = (
                                f"{_tsn}_遅れ[ms]" if _tsm == "single"
                                else f"{_tsn}_dur[ms]"
                            )

                            # ── サブグループデータ収集 ────────────────
                            _sg_lbls, _sg_xbar, _sg_r, _sg_s, _sg_n = [], [], [], [], []
                            for _r in _tr_res_list:
                                _rdf = _r["result"]
                                if _dcol not in _rdf.columns:
                                    continue
                                _rv = _rdf[_dcol].dropna().values
                                if len(_rv) < 2:
                                    continue
                                _sg_lbls.append(_r["label"])
                                _sg_xbar.append(float(np.mean(_rv)))
                                _sg_r.append(float(np.max(_rv) - np.min(_rv)))
                                _sg_s.append(float(np.std(_rv, ddof=1)))
                                _sg_n.append(len(_rv))

                            if len(_sg_lbls) < 2:
                                st.info(f"{_tsn}: データが 2 時期以上必要です")
                                continue

                            # ── サマリー行（共通）────────────────────
                            _sr = {"ステップ": _tsn, "モード": _tsm}
                            for _lbl, _m, _s in zip(_sg_lbls, _sg_xbar, _sg_s):
                                _sr[_lbl] = f"{_m:.1f} ± {_s:.1f}"
                            _tr_summary_rows.append(_sr)

                            # ── 管理限界計算 ──────────────────────────
                            _xbar_bar = float(np.mean(_sg_xbar))
                            _r_bar    = float(np.mean(_sg_r))
                            _s_bar    = float(np.mean(_sg_s))
                            _n_avg    = float(np.mean(_sg_n))
                            _n_rep    = int(round(_n_avg))

                            _A2, _D3, _D4, _c4, _B3, _B4 = _spc_consts(_n_rep)
                            _use_s = (_n_rep > 25)   # n>25 は S チャートの方が有効

                            if _use_s:
                                # Xbar-S 管理限界
                                _sigma_hat = _s_bar / _c4
                                _UCL_x = _xbar_bar + 3 * _sigma_hat / (_n_avg ** 0.5)
                                _LCL_x = _xbar_bar - 3 * _sigma_hat / (_n_avg ** 0.5)
                                _UCL_sub = _B4 * _s_bar
                                _LCL_sub = _B3 * _s_bar
                                _sub_vals  = _sg_s
                                _sub_label = "S（標準偏差）[ms]"
                                _sub_cl    = _s_bar
                                _cl_label  = f"S̄ = {_s_bar:.2f} ms"
                            else:
                                # Xbar-R 管理限界
                                _UCL_x = _xbar_bar + _A2 * _r_bar
                                _LCL_x = _xbar_bar - _A2 * _r_bar
                                _UCL_sub = _D4 * _r_bar
                                _LCL_sub = _D3 * _r_bar
                                _sub_vals  = _sg_r
                                _sub_label = "R（範囲）[ms]"
                                _sub_cl    = _r_bar
                                _cl_label  = f"R̄ = {_r_bar:.2f} ms"

                            # 管理外判定（UCL/LCL を超えたら NG）
                            _xbar_ng = [v > _UCL_x or v < _LCL_x for v in _sg_xbar]
                            _sub_ng  = [v > _UCL_sub or (_LCL_sub > 0 and v < _LCL_sub)
                                        for v in _sub_vals]

                            # ── サブプロット（上: Xbar / 下: R or S）────
                            _fig_xbr = make_subplots(
                                rows=2, cols=1,
                                shared_xaxes=True,
                                subplot_titles=(
                                    f"X̄ 管理図  （UCL={_UCL_x:.2f}  CL={_xbar_bar:.2f}  LCL={_LCL_x:.2f}）",
                                    f"{'S' if _use_s else 'R'} 管理図  "
                                    f"（UCL={_UCL_sub:.2f}  CL={_sub_cl:.2f}"
                                    + (f"  LCL={_LCL_sub:.2f}" if _LCL_sub > 0 else "  LCL=0") + "）",
                                ),
                                vertical_spacing=0.14,
                                row_heights=[0.6, 0.4],
                            )

                            # X̄ チャート ──────────────────────────────
                            _xc  = ["#e74c3c" if ng else _tsc for ng in _xbar_ng]
                            _xs  = ["x-thin" if ng else "circle" for ng in _xbar_ng]
                            _fig_xbr.add_trace(go.Scatter(
                                x=_sg_lbls, y=_sg_xbar,
                                mode="lines+markers",
                                name="X̄",
                                line=dict(color=_tsc, width=2),
                                marker=dict(color=_xc, size=11, symbol=_xs,
                                            line=dict(width=2, color="white")),
                                hovertemplate="%{x}<br>X̄ = %{y:.3f} ms<extra></extra>",
                            ), row=1, col=1)

                            # X̄ 管理限界線
                            for _yv, _dash, _col, _ann in [
                                (_UCL_x,    "dash",  "#e74c3c", f"UCL={_UCL_x:.2f}"),
                                (_xbar_bar, "solid", "#27ae60", f"X̄̄={_xbar_bar:.2f}"),
                                (_LCL_x,    "dash",  "#e74c3c", f"LCL={_LCL_x:.2f}"),
                            ]:
                                _fig_xbr.add_hline(
                                    y=_yv, line_dash=_dash, line_color=_col,
                                    line_width=1.5,
                                    annotation_text=_ann,
                                    annotation_position="right",
                                    row=1, col=1,
                                )

                            # ±1σ 帯（X̄ チャート）
                            _hex2 = _tsc.lstrip("#")
                            _rgb2 = tuple(int(_hex2[i:i+2], 16) for i in (0, 2, 4)) if len(_hex2) == 6 else (31, 119, 180)
                            _sigma_x = (_UCL_x - _xbar_bar) / 3
                            _fig_xbr.add_hrect(
                                y0=_xbar_bar - _sigma_x,
                                y1=_xbar_bar + _sigma_x,
                                fillcolor=f"rgba({_rgb2[0]},{_rgb2[1]},{_rgb2[2]},0.10)",
                                line_width=0,
                                annotation_text="±1σ帯",
                                annotation_position="right",
                                row=1, col=1,
                            )

                            # R / S チャート ──────────────────────────
                            _rc  = ["#e74c3c" if ng else "#7f8c8d" for ng in _sub_ng]
                            _fig_xbr.add_trace(go.Scatter(
                                x=_sg_lbls, y=_sub_vals,
                                mode="lines+markers",
                                name="S" if _use_s else "R",
                                line=dict(color="#7f8c8d", width=2),
                                marker=dict(color=_rc, size=10,
                                            symbol=["x-thin" if ng else "circle" for ng in _sub_ng],
                                            line=dict(width=2, color="white")),
                                hovertemplate="%{x}<br>" + ("S" if _use_s else "R") + " = %{y:.3f} ms<extra></extra>",
                            ), row=2, col=1)

                            # R/S 管理限界線
                            _sub_lines = [
                                (_UCL_sub, "dash",  "#e74c3c", f"UCL={_UCL_sub:.2f}"),
                                (_sub_cl,  "solid", "#27ae60", _cl_label),
                            ]
                            if _LCL_sub > 0:
                                _sub_lines.append(
                                    (_LCL_sub, "dash", "#e74c3c", f"LCL={_LCL_sub:.2f}")
                                )
                            for _yv, _dash, _col, _ann in _sub_lines:
                                _fig_xbr.add_hline(
                                    y=_yv, line_dash=_dash, line_color=_col,
                                    line_width=1.5,
                                    annotation_text=_ann,
                                    annotation_position="right",
                                    row=2, col=1,
                                )

                            # 基準値ライン（X̄ チャートに重ねる）
                            _tbl  = _tr_bl.get(_tsn, {})
                            _tref = (
                                _tbl.get("ref_ms") if _tsm == "single"
                                else _tbl.get("ref_dur_ms")
                            )
                            if _tref is not None:
                                _fig_xbr.add_hline(
                                    y=_tref,
                                    line_dash="dot", line_color="#8e44ad", line_width=2,
                                    annotation_text=f"基準値 {_tref:.1f} ms",
                                    annotation_position="left",
                                    row=1, col=1,
                                )

                            _fig_xbr.update_layout(
                                title=dict(
                                    text=(
                                        f"<b>{_tsn}</b>　"
                                        f"Xbar-{'S' if _use_s else 'R'} 管理図"
                                        + (f"　（n={_n_rep}、Xbar-S で算出）" if _use_s else f"　（n={_n_rep}）")
                                    ),
                                    font=dict(size=14),
                                ),
                                height=480,
                                margin=dict(t=60, b=48, l=80, r=120),
                                showlegend=True,
                                legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                                hovermode="x unified",
                            )
                            _fig_xbr.update_yaxes(title_text="X̄ [ms]", row=1, col=1)
                            _fig_xbr.update_yaxes(title_text=_sub_label, row=2, col=1)
                            _fig_xbr.update_xaxes(title_text="時期", row=2, col=1)

                            # NG 点サマリー + ウェスタン・エレクトリック・ルール
                            _xbar_ng_cnt = sum(_xbar_ng)
                            _sub_ng_cnt  = sum(_sub_ng)
                            _we_warns = []
                            _xv = _sg_xbar
                            # Rule 1: UCL/LCL 超え（既存 _xbar_ng）
                            if _xbar_ng_cnt > 0:
                                _we_warns.append(f"🔴 Rule1: X̄管理外 **{_xbar_ng_cnt}** 点（UCL/LCL超え）")
                            # Rule 2: 連続9点が中心線の同側
                            if len(_xv) >= 9:
                                for _ri in range(len(_xv) - 8):
                                    _seg = _xv[_ri:_ri + 9]
                                    if all(v > _xbar_bar for v in _seg) or all(v < _xbar_bar for v in _seg):
                                        _we_warns.append(f"🟠 Rule2: 連続9点が中心線の同側（{_sg_lbls[_ri]}〜）")
                                        break
                            # Rule 3: 連続6点単調増加 or 単調減少
                            if len(_xv) >= 6:
                                for _ri in range(len(_xv) - 5):
                                    _seg = _xv[_ri:_ri + 6]
                                    if all(_seg[j] < _seg[j+1] for j in range(5)):
                                        _we_warns.append(f"🟡 Rule3: 連続6点単調増加（{_sg_lbls[_ri]}〜）")
                                        break
                                    if all(_seg[j] > _seg[j+1] for j in range(5)):
                                        _we_warns.append(f"🟡 Rule3: 連続6点単調減少（{_sg_lbls[_ri]}〜）")
                                        break
                            # Rule 4: 連続14点交互増減
                            if len(_xv) >= 14:
                                for _ri in range(len(_xv) - 13):
                                    _seg = _xv[_ri:_ri + 14]
                                    if all((_seg[j] - _seg[j+1]) * (_seg[j+1] - _seg[j+2]) < 0
                                           for j in range(12)):
                                        _we_warns.append(f"🟡 Rule4: 連続14点交互増減（{_sg_lbls[_ri]}〜）")
                                        break
                            if _we_warns:
                                for _ww in _we_warns:
                                    st.warning(f"**{_tsn}**: {_ww}")
                            elif _sub_ng_cnt > 0:
                                st.warning(
                                    f"**{_tsn}**: "
                                    f"{'S' if _use_s else 'R'} 管理外 {_sub_ng_cnt} 点"
                                )

                            st.plotly_chart(_fig_xbr, width="stretch",
                                            key=f"xbr_{_tr_pname}_{_tsn}")

                    # ── サマリーテーブル ────────────────────────────
                    if _tr_summary_rows:
                        st.divider()
                        st.markdown("#### 📋 サマリーテーブル（平均 ± σ）")
                        st.caption("🔴 は基準値から±Nσ 以上の逸脱")
                        _tr_sum_df = pd.DataFrame(_tr_summary_rows)
                        st.dataframe(_tr_sum_df, hide_index=True, width="stretch")

                        # CSV ダウンロード
                        _tr_dl_rows = []
                        for _ts2 in _tr_steps:
                            _tsn2  = _ts2["name"]
                            _tsm2  = _ts2["mode"]
                            _dcol2 = (
                                f"{_tsn2}_遅れ[ms]" if _tsm2 == "single"
                                else f"{_tsn2}_dur[ms]"
                            )
                            for _r2 in _tr_res_list:
                                _rdf2 = _r2["result"]
                                if _dcol2 not in _rdf2.columns:
                                    continue
                                _rv2 = _rdf2[_dcol2].dropna().values
                                if len(_rv2) == 0:
                                    continue
                                _tr_dl_rows.append({
                                    "時期":         _r2["label"],
                                    "ステップ":       _tsn2,
                                    "サイクル数":     len(_rv2),
                                    "平均[ms]":      round(float(np.mean(_rv2)), 3),
                                    "標準偏差[ms]":   round(float(np.std(_rv2)), 3),
                                    "最小[ms]":      round(float(np.min(_rv2)), 3),
                                    "最大[ms]":      round(float(np.max(_rv2)), 3),
                                })
                        if _tr_dl_rows:
                            st.download_button(
                                "📥 傾向データCSVダウンロード",
                                pd.DataFrame(_tr_dl_rows).to_csv(
                                    index=False, encoding="utf-8-sig"
                                ),
                                f"{_tr_pname}_trend.csv",
                                key=f"tr_dl_{_tr_pname}",
                                width="stretch",
                            )

                    # ── 波形監視トレンド ────────────────────────────
                    _wv_tr_steps = [s for s in _tr_steps if s.get("waveform_vars")]
                    _has_wv_data = any(
                        _r.get("wv_stats") for _r in _tr_res_list
                    )
                    if _wv_tr_steps and _has_wv_data:
                        st.divider()
                        st.markdown("#### 📈 波形監視トレンド")
                        st.caption(
                            "各時期の波形ピーク値（検査ウィンドウ内最大値の平均）と NG 率の推移。"
                            "良品範囲が設定されているステップ・変数のみ NG 判定が有効です。"
                        )

                        for _wvs_step in _wv_tr_steps:
                            _wvs_sn  = _wvs_step["name"]
                            _wvs_col = _wvs_step.get("color", "#4472C4")
                            for _wvs_var in _wvs_step.get("waveform_vars", []):
                                # ファイル毎の統計収集
                                _wvt_lbls, _wvt_peaks, _wvt_ng_rates = [], [], []
                                for _rr in _tr_res_list:
                                    _rvs = (_rr.get("wv_stats") or {}).get(_wvs_sn, {}).get(_wvs_var)
                                    if not _rvs or _rvs["total"] == 0:
                                        continue
                                    _wvt_lbls.append(_rr["label"])
                                    _wvt_peaks.append(
                                        float(np.mean(_rvs["peaks"])) if _rvs["peaks"] else 0.0
                                    )
                                    _wvt_ng_rates.append(
                                        _rvs["ng_count"] / _rvs["total"] * 100
                                    )

                                if len(_wvt_lbls) < 2:
                                    continue

                                _fig_wvt = make_subplots(
                                    rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=(
                                        "ピーク値トレンド（検査ウィンドウ内最大値の時期別平均）",
                                        "NG率トレンド [%]",
                                    ),
                                    vertical_spacing=0.18,
                                    row_heights=[0.6, 0.4],
                                )

                                # ピーク値折れ線
                                _hex_w = _wvs_col.lstrip("#")
                                _rgb_w = (
                                    tuple(int(_hex_w[i:i+2], 16) for i in (0, 2, 4))
                                    if len(_hex_w) == 6 else (68, 114, 196)
                                )
                                _fill_w = f"rgba({_rgb_w[0]},{_rgb_w[1]},{_rgb_w[2]},0.15)"
                                _fig_wvt.add_trace(go.Scatter(
                                    x=_wvt_lbls, y=_wvt_peaks,
                                    mode="lines+markers",
                                    name="ピーク平均",
                                    line=dict(color=_wvs_col, width=2),
                                    marker=dict(size=9, color=_wvs_col,
                                                line=dict(width=1.5, color="white")),
                                    hovertemplate="%{x}<br>ピーク平均: %{y:.3f}<extra></extra>",
                                ), row=1, col=1)

                                # NG率棒グラフ
                                _ng_bar_cols = [
                                    "#e74c3c" if r > 0 else "#27ae60" for r in _wvt_ng_rates
                                ]
                                _fig_wvt.add_trace(go.Bar(
                                    x=_wvt_lbls, y=_wvt_ng_rates,
                                    name="NG率",
                                    marker_color=_ng_bar_cols,
                                    hovertemplate="%{x}<br>NG率: %{y:.1f}%<extra></extra>",
                                ), row=2, col=1)

                                _fig_wvt.update_layout(
                                    title=dict(
                                        text=f"<b>{_wvs_sn}　/　{_wvs_var}</b>　波形監視トレンド",
                                        font=dict(size=14),
                                    ),
                                    height=420,
                                    margin=dict(t=60, b=48, l=80, r=40),
                                    showlegend=True,
                                    legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                                    hovermode="x unified",
                                )
                                _fig_wvt.update_yaxes(title_text=_wvs_var, row=1, col=1)
                                _fig_wvt.update_yaxes(title_text="NG率 [%]", row=2, col=1)
                                _fig_wvt.update_xaxes(title_text="時期", row=2, col=1)
                                st.plotly_chart(
                                    _fig_wvt, width="stretch",
                                    key=f"wvtr_{_tr_pname}_{_wvs_sn}_{_wvs_var}",
                                )

                    # ── 波形検査 検出点トレンド ─────────────────────────────
                    st.divider()
                    st.markdown("#### 🔍 波形検査 検出点トレンド")

                    # 解析エラーがあれば表示
                    _wi_errs = [
                        (r["label"], r["wi_err"])
                        for r in _tr_res_list
                        if r.get("wi_err")
                    ]
                    if _wi_errs:
                        with st.expander(
                            f"⚠️ 波形検査 解析エラー ({len(_wi_errs)} 件)",
                            expanded=True,
                        ):
                            for _el, _ee in _wi_errs:
                                st.error(f"**{_el}**: {_ee}")

                    _wi_det_keys: dict = {}
                    for _r in _tr_res_list:
                        for _dk, _dv in _r.get("wi_det_stats", {}).items():
                            if _dk.startswith("__"):  # 内部管理キーをスキップ
                                continue
                            if _dk not in _wi_det_keys:
                                _wi_det_keys[_dk] = {
                                    "label":      _dv["label"],
                                    "color":      _dv["color"],
                                    "is_xy":      _dv.get("is_xy", False),
                                    "is_formula": _dv.get("is_formula", False),
                                    "x_label":    _dv.get("x_label", "t"),
                                    "y_label":    _dv.get("y_label", "v"),
                                    "expr":       _dv.get("expr", ""),
                                }
                    if not _wi_det_keys:
                        # 設定状況を診断して案内
                        _wi_trig_cur = st.session_state.get("wi_trigger", "")
                        _wi_trend_configured = False
                        for _nc in num_cols:
                            _vk = f"wvol___global_{_nc}"
                            for _det in st.session_state.get(f"{_vk}_t_det_list", []):
                                if st.session_state.get(
                                        f"{_vk}_{_det['id']}_trend_on", False):
                                    _wi_trend_configured = True
                                    break
                            if _wi_trend_configured:
                                break
                        if not _wi_trend_configured:
                            st.info(
                                "🔍 波形検査タブで検出点を追加し、"
                                "**📈 傾向解析に出す** をオンにしてから再実行してください。"
                            )
                        else:
                            st.warning(
                                f"⚠️ 波形検査 検出点は設定済みですが、いずれのCSVでも検出点が見つかりませんでした。"
                                f"（波形検査トリガー: **{_wi_trig_cur}**）\n\n"
                                "考えられる原因:\n"
                                "- トリガー変数が比較CSVに存在しない\n"
                                "- 傾き変化点の「閾値」が 0\n"
                                "- 閾値超え検出の「閾値」が波形に合っていない\n"
                                "- 検索範囲が狭すぎてどのサイクルでも検出できない"
                            )
                    if _wi_det_keys:
                        st.caption(
                            "「波形検査」タブで **📈 傾向解析に出す** を有効にした検出点の、"
                            "CSVファイルごとの傾向です。"
                        )
                        for _wdk, _wdinfo in _wi_det_keys.items():
                            _wd_label          = _wdinfo["label"]
                            _wd_color          = _wdinfo["color"]
                            _wdinfo_is_xy      = _wdinfo.get("is_xy", False)
                            _wdinfo_is_formula = _wdinfo.get("is_formula", False)
                            _wd_xlabel         = _wdinfo.get("x_label", "t [ms]")
                            _wd_ylabel         = _wdinfo.get("y_label", "v")
                            # ── per-CSV データ収集 ──────────────────────
                            _wd_lbls: list     = []
                            _wd_t_means: list  = []
                            _wd_t_stds:  list  = []
                            _wd_t_ranges: list = []
                            _wd_v_means: list  = []
                            _wd_v_stds:  list  = []
                            _wd_v_ranges: list = []
                            _wd_ns: list       = []
                            for _r in _tr_res_list:
                                _dstat = _r.get("wi_det_stats", {}).get(_wdk)
                                if _dstat is None:
                                    continue
                                _wd_lbls.append(_r["label"])
                                _wd_t_means.append(_dstat["t_mean"])
                                _wd_t_stds.append(_dstat["t_std"])
                                _wd_t_ranges.append(_dstat.get("t_range", 0.0))
                                _wd_v_means.append(_dstat["v_mean"])
                                _wd_v_stds.append(_dstat["v_std"])
                                _wd_v_ranges.append(_dstat.get("v_range", 0.0))
                                _wd_ns.append(_dstat.get("n", 1))
                            if not _wd_lbls:
                                continue
                            with st.expander(f"**{_wd_label}**", expanded=True):
                                # ─────────────────────────────────────────
                                # A) 傾向チャート（平均 ± σ）
                                # ─────────────────────────────────────────
                                if _chart_mode == "📈 傾向チャート（平均 ± σ）":
                                    if _wdinfo_is_formula:
                                        # 数式: 1行（数式結果のみ）
                                        _wd_fig = make_subplots(rows=1, cols=1)
                                        if len(_wd_lbls) > 1:
                                            _wd_fig.add_trace(go.Scatter(
                                                x=_wd_lbls + _wd_lbls[::-1],
                                                y=([m + s for m, s in zip(_wd_t_means, _wd_t_stds)]
                                                   + [m - s for m, s in zip(_wd_t_means[::-1], _wd_t_stds[::-1])]),
                                                fill="toself",
                                                fillcolor="rgba(68,114,196,0.15)",
                                                line=dict(width=0),
                                                showlegend=True, name="±1σ",
                                            ), row=1, col=1)
                                        _wd_fig.add_trace(go.Scatter(
                                            x=_wd_lbls, y=_wd_t_means,
                                            mode="lines+markers", name="数式結果 平均",
                                            line=dict(color=_wd_color, width=2),
                                            marker=dict(size=9, color=_wd_color),
                                            hovertemplate="%{x}<br>平均: %{y:.4f}<br>σ: %{customdata:.4f}<extra></extra>",
                                            customdata=_wd_t_stds,
                                        ), row=1, col=1)
                                        _wd_fig.update_layout(
                                            title=dict(text=f"<b>{_wd_label}</b>　傾向", font=dict(size=13)),
                                            height=300,
                                            margin=dict(t=60, b=48, l=80, r=100),
                                            showlegend=True,
                                            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                                            hovermode="x unified",
                                        )
                                        _wd_fig.update_yaxes(title_text="数式結果")
                                        _wd_fig.update_xaxes(title_text="時期")
                                    else:
                                        # 時間軸 / XY: t + v の 2行
                                        _t_title = f"{'X: ' + _wd_xlabel if _wdinfo_is_xy else 't 値 [ms]'}"
                                        _v_title = f"{'Y: ' + _wd_ylabel if _wdinfo_is_xy else 'v 値'}"
                                        _wd_fig = make_subplots(
                                            rows=2, cols=1,
                                            subplot_titles=(_t_title, _v_title),
                                            vertical_spacing=0.25,
                                            row_heights=[0.5, 0.5],
                                        )
                                        if len(_wd_lbls) > 1:
                                            _wd_fig.add_trace(go.Scatter(
                                                x=_wd_lbls + _wd_lbls[::-1],
                                                y=([m + s for m, s in zip(_wd_t_means, _wd_t_stds)]
                                                   + [m - s for m, s in zip(_wd_t_means[::-1], _wd_t_stds[::-1])]),
                                                fill="toself", fillcolor="rgba(68,114,196,0.15)",
                                                line=dict(width=0), showlegend=True, name="t ±1σ",
                                            ), row=1, col=1)
                                        _wd_fig.add_trace(go.Scatter(
                                            x=_wd_lbls, y=_wd_t_means,
                                            mode="lines+markers", name="t 平均 [ms]",
                                            line=dict(color=_wd_color, width=2),
                                            marker=dict(size=9, color=_wd_color),
                                            hovertemplate="%{x}<br>t 平均: %{y:.3f} ms<br>σ: %{customdata:.3f} ms<extra></extra>",
                                            customdata=_wd_t_stds,
                                        ), row=1, col=1)
                                        if len(_wd_lbls) > 1:
                                            _wd_fig.add_trace(go.Scatter(
                                                x=_wd_lbls + _wd_lbls[::-1],
                                                y=([m + s for m, s in zip(_wd_v_means, _wd_v_stds)]
                                                   + [m - s for m, s in zip(_wd_v_means[::-1], _wd_v_stds[::-1])]),
                                                fill="toself", fillcolor="rgba(68,114,196,0.15)",
                                                line=dict(width=0), showlegend=True, name="v ±1σ",
                                            ), row=2, col=1)
                                        _wd_fig.add_trace(go.Scatter(
                                            x=_wd_lbls, y=_wd_v_means,
                                            mode="lines+markers", name="v 平均",
                                            line=dict(color=_wd_color, width=2, dash="dash"),
                                            marker=dict(size=9, color=_wd_color, symbol="diamond"),
                                            hovertemplate="%{x}<br>v 平均: %{y:.4f}<br>σ: %{customdata:.4f}<extra></extra>",
                                            customdata=_wd_v_stds,
                                        ), row=2, col=1)
                                        _wd_fig.update_layout(
                                            title=dict(text=f"<b>{_wd_label}</b>　傾向", font=dict(size=13)),
                                            height=560,
                                            margin=dict(t=60, b=48, l=80, r=100),
                                            showlegend=True,
                                            legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                                            hovermode="x unified",
                                        )
                                        _wd_fig.update_yaxes(title_text="t [ms]", row=1, col=1)
                                        _wd_fig.update_yaxes(title_text="v", row=2, col=1)
                                        _wd_fig.update_xaxes(title_text="時期", row=2, col=1)
                                    st.plotly_chart(_wd_fig, width="stretch",
                                                    key=f"wi_det_tr_{_wdk.replace('/', '_')}")

                                # ─────────────────────────────────────────
                                # B) Xbar-R 管理図
                                # ─────────────────────────────────────────
                                else:
                                    # サブグループ: n≥2 のCSVのみ有効
                                    _sgw_lbls  = [l for l, n in zip(_wd_lbls,    _wd_ns) if n >= 2]
                                    _sgw_xbar  = [m for m, n in zip(_wd_t_means, _wd_ns) if n >= 2]
                                    _sgw_r     = [r for r, n in zip(_wd_t_ranges,_wd_ns) if n >= 2]
                                    _sgw_s     = [s for s, n in zip(_wd_t_stds,  _wd_ns) if n >= 2]
                                    _sgw_n     = [n for n in _wd_ns if n >= 2]
                                    if len(_sgw_lbls) < 2:
                                        st.info(
                                            "Xbar-R 管理図には n≥2 のサブグループ（サイクル数≥2 のCSVファイル）"
                                            "が **2時期以上** 必要です"
                                        )
                                    else:
                                        _xbar_bar_w = float(np.mean(_sgw_xbar))
                                        _r_bar_w    = float(np.mean(_sgw_r))
                                        _s_bar_w    = float(np.mean(_sgw_s))
                                        _n_avg_w    = float(np.mean(_sgw_n))
                                        _n_rep_w    = int(round(_n_avg_w))
                                        _A2w, _D3w, _D4w, _c4w, _B3w, _B4w = _spc_consts(_n_rep_w)
                                        _use_s_w = (_n_rep_w > 25)
                                        if _use_s_w:
                                            _sigma_w  = _s_bar_w / _c4w
                                            _UCL_xw   = _xbar_bar_w + 3 * _sigma_w / (_n_avg_w ** 0.5)
                                            _LCL_xw   = _xbar_bar_w - 3 * _sigma_w / (_n_avg_w ** 0.5)
                                            _UCL_rw   = _B4w * _s_bar_w
                                            _LCL_rw   = _B3w * _s_bar_w
                                            _sub_v_w  = _sgw_s
                                            _sub_lbl_w = "S（標準偏差）"
                                            _sub_cl_w  = _s_bar_w
                                        else:
                                            _UCL_xw   = _xbar_bar_w + _A2w * _r_bar_w
                                            _LCL_xw   = _xbar_bar_w - _A2w * _r_bar_w
                                            _UCL_rw   = _D4w * _r_bar_w
                                            _LCL_rw   = _D3w * _r_bar_w
                                            _sub_v_w  = _sgw_r
                                            _sub_lbl_w = "R（範囲）"
                                            _sub_cl_w  = _r_bar_w
                                        _xbar_ng_w = [v > _UCL_xw or v < _LCL_xw for v in _sgw_xbar]
                                        _sub_ng_w  = [v > _UCL_rw or (_LCL_rw > 0 and v < _LCL_rw)
                                                      for v in _sub_v_w]
                                        _xbr_ylbl = ("数式結果" if _wdinfo_is_formula
                                                      else ("X: " + _wd_xlabel if _wdinfo_is_xy else "t [ms]"))
                                        _fig_wi_xbr = make_subplots(
                                            rows=2, cols=1,
                                            shared_xaxes=True,
                                            subplot_titles=(
                                                f"X̄ 管理図  (UCL={_UCL_xw:.3f}  CL={_xbar_bar_w:.3f}  LCL={_LCL_xw:.3f})",
                                                f"{'S' if _use_s_w else 'R'} 管理図"
                                                f"  (UCL={_UCL_rw:.3f}  CL={_sub_cl_w:.3f}"
                                                + (f"  LCL={_LCL_rw:.3f}" if _LCL_rw > 0 else "") + ")",
                                            ),
                                            vertical_spacing=0.14,
                                            row_heights=[0.6, 0.4],
                                        )
                                        # X̄ チャート
                                        _xc_w = ["#e74c3c" if ng else _wd_color for ng in _xbar_ng_w]
                                        _xs_w = ["x-thin"  if ng else "circle"  for ng in _xbar_ng_w]
                                        _fig_wi_xbr.add_trace(go.Scatter(
                                            x=_sgw_lbls, y=_sgw_xbar,
                                            mode="lines+markers", name="X̄",
                                            line=dict(color=_wd_color, width=2),
                                            marker=dict(color=_xc_w, size=11, symbol=_xs_w,
                                                        line=dict(width=2, color="white")),
                                            hovertemplate="%{x}<br>X̄ = %{y:.4f}<extra></extra>",
                                        ), row=1, col=1)
                                        for _yv, _dash, _col, _ann in [
                                            (_UCL_xw,     "dash",  "#e74c3c", f"UCL={_UCL_xw:.3f}"),
                                            (_xbar_bar_w, "solid", "#27ae60", f"X̄̄={_xbar_bar_w:.3f}"),
                                            (_LCL_xw,     "dash",  "#e74c3c", f"LCL={_LCL_xw:.3f}"),
                                        ]:
                                            _fig_wi_xbr.add_hline(
                                                y=_yv, line_dash=_dash, line_color=_col, line_width=1.5,
                                                annotation_text=_ann, annotation_position="right",
                                                row=1, col=1,
                                            )
                                        _sigma_xw = (_UCL_xw - _xbar_bar_w) / 3
                                        _fig_wi_xbr.add_hrect(
                                            y0=_xbar_bar_w - _sigma_xw,
                                            y1=_xbar_bar_w + _sigma_xw,
                                            fillcolor="rgba(68,114,196,0.10)", line_width=0,
                                            annotation_text="±1σ帯", annotation_position="right",
                                            row=1, col=1,
                                        )
                                        # R / S チャート
                                        _rc_w = ["#e74c3c" if ng else "#7f8c8d" for ng in _sub_ng_w]
                                        _fig_wi_xbr.add_trace(go.Scatter(
                                            x=_sgw_lbls, y=_sub_v_w,
                                            mode="lines+markers",
                                            name="S" if _use_s_w else "R",
                                            line=dict(color="#7f8c8d", width=2),
                                            marker=dict(color=_rc_w, size=10,
                                                        symbol=["x-thin" if ng else "circle" for ng in _sub_ng_w],
                                                        line=dict(width=2, color="white")),
                                            hovertemplate=f"%{{x}}<br>{'S' if _use_s_w else 'R'} = %{{y:.4f}}<extra></extra>",
                                        ), row=2, col=1)
                                        _sub_lines_w = [
                                            (_UCL_rw,   "dash",  "#e74c3c", f"UCL={_UCL_rw:.3f}"),
                                            (_sub_cl_w, "solid", "#27ae60",
                                             f"{'S̄' if _use_s_w else 'R̄'}={_sub_cl_w:.3f}"),
                                        ]
                                        if _LCL_rw > 0:
                                            _sub_lines_w.append((_LCL_rw, "dash", "#e74c3c", f"LCL={_LCL_rw:.3f}"))
                                        for _yv, _dash, _col, _ann in _sub_lines_w:
                                            _fig_wi_xbr.add_hline(
                                                y=_yv, line_dash=_dash, line_color=_col, line_width=1.5,
                                                annotation_text=_ann, annotation_position="right",
                                                row=2, col=1,
                                            )
                                        # Western Electric ルール
                                        _we_w: list = []
                                        _xv_w = _sgw_xbar
                                        if sum(_xbar_ng_w) > 0:
                                            _we_w.append(f"🔴 Rule1: X̄管理外 **{sum(_xbar_ng_w)}** 点（UCL/LCL超え）")
                                        if len(_xv_w) >= 9:
                                            for _ri in range(len(_xv_w) - 8):
                                                _seg = _xv_w[_ri:_ri + 9]
                                                if all(v > _xbar_bar_w for v in _seg) or all(v < _xbar_bar_w for v in _seg):
                                                    _we_w.append(f"🟠 Rule2: 連続9点が中心線の同側（{_sgw_lbls[_ri]}〜）")
                                                    break
                                        if len(_xv_w) >= 6:
                                            for _ri in range(len(_xv_w) - 5):
                                                _seg = _xv_w[_ri:_ri + 6]
                                                if all(_seg[j] < _seg[j+1] for j in range(5)):
                                                    _we_w.append(f"🟡 Rule3: 連続6点単調増加（{_sgw_lbls[_ri]}〜）")
                                                    break
                                                if all(_seg[j] > _seg[j+1] for j in range(5)):
                                                    _we_w.append(f"🟡 Rule3: 連続6点単調減少（{_sgw_lbls[_ri]}〜）")
                                                    break
                                        for _ww in _we_w:
                                            st.warning(f"**{_wd_label}**: {_ww}")
                                        _fig_wi_xbr.update_layout(
                                            title=dict(
                                                text=f"<b>{_wd_label}</b>　"
                                                     f"Xbar-{'S' if _use_s_w else 'R'} 管理図"
                                                     + (f"　(n={_n_rep_w}、Xbar-S)" if _use_s_w
                                                        else f"　(n={_n_rep_w})"),
                                                font=dict(size=13),
                                            ),
                                            height=480,
                                            margin=dict(t=60, b=48, l=80, r=120),
                                            showlegend=True,
                                            legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
                                            hovermode="x unified",
                                        )
                                        _fig_wi_xbr.update_yaxes(title_text=_xbr_ylbl, row=1, col=1)
                                        _fig_wi_xbr.update_yaxes(title_text=_sub_lbl_w, row=2, col=1)
                                        _fig_wi_xbr.update_xaxes(title_text="時期", row=2, col=1)
                                        st.plotly_chart(_fig_wi_xbr, width="stretch",
                                                        key=f"wi_xbr_{_wdk.replace('/', '_')}")

                                # ── 統計サマリーテーブル（共通）────────────
                                if _wdinfo_is_formula:
                                    _wd_stat_rows = [
                                        {"時期": _sl, "数式結果 平均": f"{_tm:.4f}",
                                         "σ": f"{_ts_:.4f}", "n": _n}
                                        for _sl, _tm, _ts_, _n in zip(
                                            _wd_lbls, _wd_t_means, _wd_t_stds, _wd_ns)
                                    ]
                                else:
                                    _wd_stat_rows = [
                                        {
                                            "時期":        _sl,
                                            "t 平均 [ms]": f"{_tm:.3f}",
                                            "t σ [ms]":   f"{_ts_:.3f}",
                                            "v 平均":      f"{_vm:.4f}",
                                            "v σ":         f"{_vs:.4f}",
                                            "n":           _n,
                                        }
                                        for _sl, _tm, _ts_, _vm, _vs, _n in zip(
                                            _wd_lbls, _wd_t_means, _wd_t_stds,
                                            _wd_v_means, _wd_v_stds, _wd_ns)
                                    ]
                                st.dataframe(pd.DataFrame(_wd_stat_rows),
                                             hide_index=True, use_container_width=True)

            else:
                st.info("CSVファイルをアップロードして「傾向解析を実行」を押してください")


# ═══════════════════════════════════════════════════════════════
# 🔍 波形検査タブ（ステップ依存なしの独立波形監視）
# ═══════════════════════════════════════════════════════════════

with _page_tabs[2]:
    st.subheader("🔍 波形検査")
    st.caption("工程・ステップ設定に関係なく、アナログ変数の波形を直接確認・検査条件を設定できます")

    if not num_cols:
        st.info("アナログ（数値）変数が見つかりません。CSVを読み込んでください。")
    else:
        # ── 初回セッションのみ: ファイルから設定を復元 ─────────────
        if "wi_config_loaded" not in st.session_state:
            _restored = _wi_restore_config(st.session_state)
            st.session_state["wi_config_loaded"] = True
            if _restored:
                st.rerun()  # 復元した値をウィジェットに反映するため即 rerun
        # ══════════════════════════════════════════════════════════════
        # 📋 登録サマリー & 保存・読込
        # ══════════════════════════════════════════════════════════════
        # 現在設定されている検出点を全変数から収集
        _wi_sum_items: list = []
        for _sv in num_cols:
            _svkey = f"wvol___global_{_sv}"
            for _si, _sdet in enumerate(
                    st.session_state.get(f"{_svkey}_t_det_list", [])):
                _did = _sdet.get("id", "")
                _wi_sum_items.append({
                    "変数":     _sv,
                    "グラフ":   "時間軸",
                    "No":       _si + 1,
                    "タイプ":   _sdet.get("type", ""),
                    "有効":     "✅" if st.session_state.get(
                                    f"{_svkey}_{_did}_on", False) else "−",
                    "傾向解析": "📈" if st.session_state.get(
                                    f"{_svkey}_{_did}_trend_on", False) else "−",
                })
            for _si, _sdet in enumerate(
                    st.session_state.get(f"{_svkey}_xy_det_list", [])):
                _did = _sdet.get("id", "")
                _wi_sum_items.append({
                    "変数":     _sv,
                    "グラフ":   "XY",
                    "No":       _si + 1,
                    "タイプ":   _sdet.get("type", ""),
                    "有効":     "✅" if st.session_state.get(
                                    f"{_svkey}_{_did}_on", False) else "−",
                    "傾向解析": "📈" if st.session_state.get(
                                    f"{_svkey}_{_did}_trend_on", False) else "−",
                })

        _wi_trig_now = st.session_state.get("wi_trigger",
                                            bool_cols[0] if bool_cols else "")
        _wi_edge_now = st.session_state.get("wi_edge", "RISE")
        _wi_n_dets   = len(_wi_sum_items)
        _wi_n_trend  = sum(1 for r in _wi_sum_items if r["傾向解析"] == "📈")

        with st.expander(
            f"📋 登録サマリー　トリガー: **{_wi_trig_now}** / {_wi_edge_now}"
            f"　検出点: **{_wi_n_dets}** 件　傾向解析送出: **{_wi_n_trend}** 件",
            expanded=True,
        ):
            if _wi_sum_items:
                st.dataframe(
                    pd.DataFrame(_wi_sum_items),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info(
                    "検出点はまだ設定されていません。"
                    "下のトリガー設定から変数を選んで検出点を追加してください。"
                )

            st.divider()
            st.markdown("##### 💾 設定の保存・読込")
            if "wi_saved_setups" not in st.session_state:
                st.session_state["wi_saved_setups"] = _wi_load_from_file()

            # ── 保存 ──────────────────────────────────────
            _wsave_c1, _wsave_c2 = st.columns([3, 1])
            with _wsave_c1:
                _wi_save_name = st.text_input(
                    "設定名", placeholder="例: 工程A検査セット",
                    key="wi_save_name_input",
                )
            with _wsave_c2:
                st.markdown(
                    "<div style='padding-top:28px'></div>",
                    unsafe_allow_html=True,
                )
                if st.button("💾 この設定を保存", key="wi_save_btn",
                             type="secondary", use_container_width=True):
                    if not _wi_save_name.strip():
                        st.warning("設定名を入力してください")
                    else:
                        # 現在の全 wave inspection 関連キーをスナップショット
                        _snap: dict = {
                            "wi_trigger": st.session_state.get("wi_trigger", ""),
                            "wi_edge":    st.session_state.get("wi_edge", "RISE"),
                        }
                        for _sv2 in num_cols:
                            _svk2 = f"wvol___global_{_sv2}"
                            for _ss_k, _ss_v in st.session_state.items():
                                if isinstance(_ss_k, str) and (
                                    _ss_k.startswith(_svk2)
                                    or _ss_k == f"wi_sel_{_wi_pname if '_wi_pname' in dir() else '__global'}"
                                ):
                                    _snap[_ss_k] = _ss_v
                        _wi_saved_new = {
                            "name":       _wi_save_name.strip(),
                            "created_at": __import__("datetime").datetime.now()
                                          .strftime("%Y-%m-%d %H:%M"),
                            "trigger":    st.session_state.get("wi_trigger", ""),
                            "edge":       st.session_state.get("wi_edge", "RISE"),
                            "n_dets":     _wi_n_dets,
                            "n_trend":    _wi_n_trend,
                            "snapshot":   _snap,
                        }
                        st.session_state["wi_saved_setups"].append(_wi_saved_new)
                        _wi_save_to_file(st.session_state["wi_saved_setups"])
                        st.toast(
                            f"✅ 「{_wi_save_name.strip()}」を保存しました"
                            f"（検出点 {_wi_n_dets} 件）"
                        )
                        st.rerun()

            # ── 保存済み一覧 ───────────────────────────────
            if st.session_state["wi_saved_setups"]:
                st.markdown("**📁 保存済み設定一覧**")
                _wi_load_idx: int | None = None
                _wi_del_idx:  int | None = None
                for _wsi, _wse in enumerate(
                        st.session_state["wi_saved_setups"]):
                    _wsc1, _wsc2, _wsc3, _wsc4 = st.columns([3, 4, 1, 1])
                    with _wsc1:
                        st.markdown(f"**{_wse['name']}**")
                    with _wsc2:
                        st.caption(
                            f"📅 {_wse.get('created_at', '')}　"
                            f"トリガー: {_wse.get('trigger', '')} / "
                            f"{_wse.get('edge', '')}　"
                            f"検出点: {_wse.get('n_dets', 0)} 件　"
                            f"傾向: {_wse.get('n_trend', 0)} 件"
                        )
                    with _wsc3:
                        if st.button("📂 読込", key=f"wi_load_{_wsi}",
                                     help="この設定を現在の設定として読み込む"):
                            _wi_load_idx = _wsi
                    with _wsc4:
                        if st.button("🗑", key=f"wi_del_{_wsi}",
                                     help="この保存済み設定を削除"):
                            _wi_del_idx = _wsi

                if _wi_load_idx is not None:
                    _snap_ld = st.session_state[
                        "wi_saved_setups"][_wi_load_idx]["snapshot"]
                    for _lk, _lv in _snap_ld.items():
                        st.session_state[_lk] = _lv
                    _ld_name = st.session_state[
                        "wi_saved_setups"][_wi_load_idx]["name"]
                    st.toast(f"✅ 「{_ld_name}」を読み込みました")
                    st.rerun()

                if _wi_del_idx is not None:
                    _del_name = st.session_state[
                        "wi_saved_setups"][_wi_del_idx]["name"]
                    st.session_state["wi_saved_setups"].pop(_wi_del_idx)
                    _wi_save_to_file(st.session_state["wi_saved_setups"])
                    st.toast(f"🗑 「{_del_name}」を削除しました")
                    st.rerun()

        st.divider()
        # ── トリガー設定 ──────────────────────────────────────────────
        _wi_col1, _wi_col2 = st.columns([2, 2])
        with _wi_col1:
            _wi_trigger_default = bool_cols[0] if bool_cols else ""
            _wi_trigger = st.selectbox(
                "トリガー変数", bool_cols,
                index=bool_cols.index(_wi_trigger_default) if _wi_trigger_default in bool_cols else 0,
                key="wi_trigger",
            )
        with _wi_col2:
            _wi_edge = st.radio("エッジ", ["RISE", "FALL"],
                                horizontal=True, key="wi_edge")

        # 常に __global を使用（工程に依存しない独立モード）
        _wi_pname = "__global"

        # ── サイクル検出 ────────────────────────────────────────────
        if not _wi_trigger:
            st.warning("トリガー変数を選択してください")
        else:
            try:
                _wi_cs = cached_detect_cycles(df, _wi_trigger, _wi_edge)
                _wi_n  = len(_wi_cs)
            except Exception:
                _wi_n = 0

            if _wi_n == 0:
                st.error("サイクルが検出できません。トリガー変数・エッジを変更してください")
            else:
                st.caption(f"📊 {_wi_n} サイクル検出  ／  アナログ変数 {len(num_cols)} 件")

                # ── 変数選択 ────────────────────────────────────────
                _wi_sel = st.multiselect(
                    "検査する変数を選択",
                    num_cols,
                    default=num_cols[:3],
                    key=f"wi_sel_{_wi_pname}",
                )

                if not _wi_sel:
                    st.info("変数を1件以上選択してください")
                else:
                    # サイクル表示範囲（重さ対策）
                    _wi_cyc_range = st.slider(
                        "表示サイクル範囲",
                        1, _wi_n,
                        (1, min(_wi_n, 10)),
                        key=f"wi_cyc_range_{_wi_pname}",
                    )

                    # df をサイクル範囲でスライス
                    _wi_cs_list = list(_wi_cs)
                    _wi_s_idx = _wi_cs_list[_wi_cyc_range[0] - 1]
                    _wi_e_idx = (
                        _wi_cs_list[_wi_cyc_range[1]]
                        if _wi_cyc_range[1] < len(_wi_cs_list)
                        else len(df) - 1
                    )
                    _wi_df = df.loc[_wi_s_idx:_wi_e_idx]

                    # ── 基準CSV df を取得（比較データ閲覧時にオーバーレイ）──
                    _wi_ref_overlay = None
                    _wi_store2 = st.session_state.get("csv_store", {})
                    _wi_ref_key2  = st.session_state.get("ref_csv_key", "")
                    _wi_act_key2  = st.session_state.get("active_csv", "")
                    _wi_is_cmp_mode = st.session_state.get("compare_mode", False)
                    _wi_cmp_entries: list = []
                    if _wi_is_cmp_mode:
                        # 比較モード: compare_csv_keys の各CSV を compare_entries に
                        _wi_cmp_keys = st.session_state.get("compare_csv_keys", [])
                        for _wi_ci, _wi_ck in enumerate(_wi_cmp_keys):
                            if _wi_ck not in _wi_store2:
                                continue
                            _wi_ce = _wi_store2[_wi_ck]
                            _wi_cmp_entries.append({
                                "key":       _wi_ck,
                                "label":     _wi_ce.get("label", _wi_ck),
                                "df":        _wi_ce["df"],
                                "result_df": None,   # standalone モード: start_offsets は0
                                "color":     _CMP_PALETTE[_wi_ci % len(_CMP_PALETTE)],
                            })
                        if _wi_cmp_entries:
                            st.caption(
                                f"🔀 比較モード: {len(_wi_cmp_entries)}件のCSV を重ねて表示"
                            )
                    elif (_wi_ref_key2
                            and _wi_ref_key2 in _wi_store2
                            and _wi_act_key2 != _wi_ref_key2):
                        _wi_ref_overlay = _wi_store2[_wi_ref_key2]["df"]
                        _wi_ref_lbl = _wi_store2[_wi_ref_key2].get(
                            "label", _wi_ref_key2)
                        st.caption(
                            f"🟠 基準データ（{_wi_ref_lbl}）を橙色で重ねて表示しています"
                        )

                    # ── スタンドアロンモードで波形オーバーレイを呼び出す ──
                    _render_waveform_overlay(
                        _wi_df,
                        _wi_trigger,
                        _wi_edge,
                        step_stat=None,
                        step=None,
                        pname=_wi_pname,
                        result_df=None,
                        _sa_vars=_wi_sel,
                        _ref_df=_wi_ref_overlay,
                        _compare_entries=_wi_cmp_entries if _wi_cmp_entries else None,
                    )
                    # 現在の設定を自動保存（サーバー再起動対策）
                    _wi_save_config(st.session_state, num_cols)
