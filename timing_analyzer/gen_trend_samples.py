"""
gen_trend_samples.py — 傾向解析デモ用 複数時期CSVを生成

シナリオ: 圧入ライン（ProB）のガイドレール摩耗が徐々に進行
  - 立ち上げ直後     : 正常（基準値に近い）
  - 1ヶ月後         : わずかに遅延・ばらつき増加
  - 3ヶ月後         : 明らかな遅延傾向
  - 6ヶ月後         : 顕著な劣化・一部極端NG
  - 9ヶ月後（警告） : 複数工程で閾値超え → メンテナンス要

出力先: timing_analyzer/trend_samples/
  trend_01_startup.csv   … 立ち上げ（基準値登録用）
  trend_02_1month.csv    … 1ヶ月後
  trend_03_3month.csv    … 3ヶ月後
  trend_04_6month.csv    … 6ヶ月後
  trend_05_9month.csv    … 9ヶ月後（要メンテ）
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUT_DIR = Path(__file__).parent / "trend_samples"
OUT_DIR.mkdir(exist_ok=True)

# ─── 共通パラメータ ───────────────────────────────────────────────
CYCLE_MS = 600
PULSE_MS = 50
BOOL_COLS = [
    'Assembly.ProA_CycleStart',
    'Assembly.ProA_LoadReq',
    'Assembly.ProA_SliderFwd',
    'Assembly.ProA_PartDetect',
    'Assembly.ProA_ClampON',
    'Assembly.ProB_CycleStart',
    'Assembly.ProB_PressDown',
    'Assembly.ProB_PressContact',
    'Assembly.ProB_PressComplete',
    'Assembly.ProB_PressUp',
    'Assembly.ProC_CycleStart',
    'Assembly.ProC_CamTrigger',
    'Assembly.ProC_InspectDone',
    'Assembly.ProC_GateOpen',
    'Assembly.ProC_EjectDone',
]


def generate(filename: str, n_cycles: int, seed: int, start_dt: datetime,
             # 劣化パラメータ（ms 単位のオフセット加算）
             pb_contact_drift: float = 0.0,   # ProB 接触タイミング遅れ
             pb_complete_drift: float = 0.0,  # ProB 圧入完了タイミング遅れ
             pb_noise_scale: float = 1.0,     # ProB ノイズ倍率
             pa_detect_drift: float = 0.0,    # ProA 検出遅れ
             pc_inspect_drift: float = 0.0,   # ProC 検査遅れ
             anomaly_pb_slow: set = None,      # 特定サイクルに大幅遅延
             anomaly_pb_extreme: set = None,
             anomaly_pa_slow: set = None,
             anomaly_pc_slow: set = None,
             ):
    if anomaly_pb_slow    is None: anomaly_pb_slow    = set()
    if anomaly_pb_extreme is None: anomaly_pb_extreme = set()
    if anomaly_pa_slow    is None: anomaly_pa_slow    = set()
    if anomaly_pc_slow    is None: anomaly_pc_slow    = set()

    rng = np.random.default_rng(seed)
    total_rows = n_cycles * CYCLE_MS

    data = {c: np.zeros(total_rows, dtype=np.int8) for c in BOOL_COLS}
    data['Assembly.ProA_Running']        = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProB_Running']        = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProC_Running']        = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProB_Step']           = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.LineStep']            = np.zeros(total_rows, dtype=np.int8)
    data['Assembly.ProA_ClampPressure']  = np.zeros(total_rows)
    data['Assembly.ProB_PressForce']     = np.zeros(total_rows)
    data['Assembly.ProC_InspectScore']   = np.zeros(total_rows)

    def pulse(col, start, dur=PULSE_MS):
        s = int(np.clip(start, 0, total_rows - 1))
        e = min(s + dur, total_rows)
        data[col][s:e] = 1

    def fill_range(col, start, end):
        s = int(np.clip(start, 0, total_rows - 1))
        e = int(np.clip(end,   0, total_rows))
        if e > s:
            data[col][s:e] = 1

    for c in range(n_cycles):
        base = c * CYCLE_MS
        cyc  = c + 1

        # ── ProA ─────────────────────────────────────────────────
        pa_base  = base
        pulse('Assembly.ProA_CycleStart', pa_base)

        pa_extra  = 80 if cyc in anomaly_pa_slow else 0
        pa_load   = pa_base + int(rng.normal(50,  5))
        pa_slider = pa_base + int(rng.normal(180, 15))
        pa_detect = pa_base + int(rng.normal(320 + pa_detect_drift, 18)) + pa_extra
        pa_clamp  = pa_base + int(rng.normal(400 + pa_detect_drift, 18)) + pa_extra

        pa_load   = max(pa_base + 10,   pa_load)
        pa_slider = max(pa_load  + 20,  pa_slider)
        pa_detect = max(pa_slider + 20, pa_detect)
        pa_clamp  = max(pa_detect + 20, pa_clamp)

        pa_end = min(pa_clamp + PULSE_MS, base + CYCLE_MS)
        fill_range('Assembly.ProA_LoadReq',    pa_load,   pa_slider)
        fill_range('Assembly.ProA_SliderFwd',  pa_slider, pa_detect)
        fill_range('Assembly.ProA_PartDetect', pa_detect, pa_clamp)
        fill_range('Assembly.ProA_ClampON',    pa_clamp,  pa_end)

        for t in range(pa_clamp, min(pa_clamp + 150, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pa_clamp) / 150.0
            if   p < 0.25: v = 8.0 * p / 0.25
            elif p < 0.75: v = 8.0
            else:          v = 8.0 * (1.0 - (p - 0.75) / 0.25)
            data['Assembly.ProA_ClampPressure'][t] = max(0.0, v + rng.normal(0, 0.12))

        # ── ProB ─────────────────────────────────────────────────
        pb_base = base
        pulse('Assembly.ProB_CycleStart', pb_base)

        pb_extra   = 60  if cyc in anomaly_pb_slow    else 0
        pb_extreme = 150 if cyc in anomaly_pb_extreme else 0

        pb_noise = 8 * pb_noise_scale
        pb_down     = pb_base + int(rng.normal(60,  pb_noise))
        pb_contact  = pb_base + int(rng.normal(200 + pb_contact_drift,  20 * pb_noise_scale))
        pb_complete = pb_base + int(rng.normal(400 + pb_complete_drift, 25 * pb_noise_scale)) + pb_extra + pb_extreme
        pb_up       = pb_base + int(rng.normal(480 + pb_complete_drift, 15 * pb_noise_scale)) + pb_extra + pb_extreme

        pb_down     = max(pb_base + 10,     pb_down)
        pb_contact  = max(pb_down  + 20,    pb_contact)
        pb_complete = max(pb_contact + 20,  pb_complete)
        pb_up       = max(pb_complete + 20, pb_up)

        pb_end = min(pb_up + PULSE_MS, base + CYCLE_MS)
        fill_range('Assembly.ProB_PressDown',     pb_down,     pb_contact)
        fill_range('Assembly.ProB_PressContact',  pb_contact,  pb_complete)
        fill_range('Assembly.ProB_PressComplete', pb_complete, pb_up)
        fill_range('Assembly.ProB_PressUp',       pb_up,       pb_end)

        peak_f = 25.0 + (1.5 if (cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme) else 0.0)
        span   = max(1, pb_complete - pb_contact)
        for t in range(pb_contact, min(pb_up + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pb_contact) / span
            if   p < 0.20: v = peak_f * p / 0.20
            elif p < 0.85: v = peak_f
            else:          v = peak_f * max(0.0, 1.0 - (p - 0.85) / 0.15) * 0.5
            data['Assembly.ProB_PressForce'][t] = max(0.0, v + rng.normal(0, 0.25))

        # ── ProC ─────────────────────────────────────────────────
        pc_base = base
        pulse('Assembly.ProC_CycleStart', pc_base)

        pc_extra = 50 if cyc in anomaly_pc_slow else 0
        is_ng    = cyc in anomaly_pc_slow

        pc_cam     = pc_base + int(rng.normal(30,  3))
        pc_inspect = pc_base + int(rng.normal(120 + pc_inspect_drift, 12)) + pc_extra
        pc_gate    = pc_base + int(rng.normal(175 + pc_inspect_drift,  8)) + pc_extra
        pc_eject   = pc_base + int(rng.normal(260 + pc_inspect_drift, 12)) + pc_extra

        pc_cam     = max(pc_base + 10,    pc_cam)
        pc_inspect = max(pc_cam   + 20,   pc_inspect)
        pc_gate    = max(pc_inspect + 10, pc_gate)
        pc_eject   = max(pc_gate   + 20,  pc_eject)

        pc_end = min(pc_eject + PULSE_MS, base + CYCLE_MS)
        fill_range('Assembly.ProC_CamTrigger',  pc_cam,     pc_inspect)
        fill_range('Assembly.ProC_InspectDone', pc_inspect, pc_gate)
        fill_range('Assembly.ProC_GateOpen',    pc_gate,    pc_eject)
        fill_range('Assembly.ProC_EjectDone',   pc_eject,   pc_end)

        score_base = 58.0 if is_ng else 94.0
        score_std  =  4.0 if is_ng else  2.0
        for t in range(pc_cam, min(pc_inspect + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            data['Assembly.ProC_InspectScore'][t] = float(
                np.clip(rng.normal(score_base, score_std), 0, 100))

        # ── Running & Step ────────────────────────────────────────
        def _c(v): return int(np.clip(v, 0, total_rows))

        data['Assembly.ProA_Running'][_c(base):_c(pa_end)] = 1
        data['Assembly.ProB_Running'][_c(base):_c(pb_end)] = 1
        data['Assembly.ProC_Running'][_c(base):_c(pc_end)] = 1

        data['Assembly.ProB_Step'][_c(base)       : _c(pb_down)    ] = 1
        data['Assembly.ProB_Step'][_c(pb_down)    : _c(pb_contact) ] = 2
        data['Assembly.ProB_Step'][_c(pb_contact) : _c(pb_complete)] = 3
        data['Assembly.ProB_Step'][_c(pb_complete): _c(pb_end)     ] = 4

        ends = sorted([pa_end, pb_end, pc_end])
        data['Assembly.LineStep'][_c(base)    : _c(ends[0])] = 3
        data['Assembly.LineStep'][_c(ends[0]) : _c(ends[1])] = 2
        data['Assembly.LineStep'][_c(ends[1]) : _c(ends[2])] = 1

    # ── DataFrame 構築 ─────────────────────────────────────────
    ts_list = []
    cur = start_dt
    for _ in range(total_rows):
        ts_list.append(cur.strftime('%Y-%m-%d %H:%M:%S.%f'))
        cur += timedelta(milliseconds=1)

    df = pd.DataFrame({'Date Time': ts_list})
    df['Assembly.ProB_Step'] = data['Assembly.ProB_Step'].astype(int)
    df['Assembly.LineStep']  = data['Assembly.LineStep'].astype(int)
    for c in BOOL_COLS:
        df[c] = data[c].astype(int)
    df['Assembly.ProA_Running']       = data['Assembly.ProA_Running'].astype(int)
    df['Assembly.ProB_Running']       = data['Assembly.ProB_Running'].astype(int)
    df['Assembly.ProC_Running']       = data['Assembly.ProC_Running'].astype(int)
    df['Assembly.ProA_ClampPressure'] = np.round(data['Assembly.ProA_ClampPressure'], 3)
    df['Assembly.ProB_PressForce']    = np.round(data['Assembly.ProB_PressForce'],    3)
    df['Assembly.ProC_InspectScore']  = np.round(data['Assembly.ProC_InspectScore'],  2)

    out = OUT_DIR / filename
    df.to_csv(out, index=False)
    print(f"  OK {filename}  ({len(df):,} rows, {n_cycles} cycles)")
    return df


# ═══════════════════════════════════════════════════════════════════
# 5 時期のCSVを生成
# ═══════════════════════════════════════════════════════════════════

print("傾向解析デモ用CSVを生成中...")

# ① 立ち上げ直後（基準値登録用） — 2025-04-01
generate(
    "trend_01_startup.csv",
    n_cycles=40, seed=42,
    start_dt=datetime(2025, 4, 1, 8, 0, 0),
    pb_contact_drift=0,
    pb_complete_drift=0,
    pb_noise_scale=1.0,
    pa_detect_drift=0,
    pc_inspect_drift=0,
    anomaly_pb_slow={10, 23},       # まれに遅い（許容範囲）
    anomaly_pa_slow={5},
)

# ② 1ヶ月後 — 2025-05-01
generate(
    "trend_02_1month.csv",
    n_cycles=40, seed=101,
    start_dt=datetime(2025, 5, 1, 8, 0, 0),
    pb_contact_drift=5,             # 接触が 5ms 遅れ始める
    pb_complete_drift=8,
    pb_noise_scale=1.1,
    pa_detect_drift=2,
    pc_inspect_drift=0,
    anomaly_pb_slow={7, 18},
)

# ③ 3ヶ月後 — 2025-07-01
generate(
    "trend_03_3month.csv",
    n_cycles=40, seed=202,
    start_dt=datetime(2025, 7, 1, 8, 0, 0),
    pb_contact_drift=15,
    pb_complete_drift=25,
    pb_noise_scale=1.3,
    pa_detect_drift=5,
    pc_inspect_drift=3,
    anomaly_pb_slow={3, 12, 25},
)

# ④ 6ヶ月後 — 2025-10-01
generate(
    "trend_04_6month.csv",
    n_cycles=40, seed=303,
    start_dt=datetime(2025, 10, 1, 8, 0, 0),
    pb_contact_drift=28,
    pb_complete_drift=50,
    pb_noise_scale=1.6,
    pa_detect_drift=10,
    pc_inspect_drift=8,
    anomaly_pb_slow={5, 11, 20, 30},
    anomaly_pb_extreme={35},
)

# ⑤ 9ヶ月後（要メンテナンス） — 2026-01-01
generate(
    "trend_05_9month.csv",
    n_cycles=40, seed=404,
    start_dt=datetime(2026, 1, 1, 8, 0, 0),
    pb_contact_drift=45,
    pb_complete_drift=85,
    pb_noise_scale=2.0,
    pa_detect_drift=18,
    pc_inspect_drift=15,
    anomaly_pb_slow={2, 6, 10, 15, 22, 28, 34},
    anomaly_pb_extreme={18, 38},
    anomaly_pc_slow={8, 20, 33},
)

print(f"\n完了！ → {OUT_DIR}")
print("\n【傾向解析での使い方】")
print("  1. メインアプリで sample_playback.csv を読み込み、工程・ステップ・基準値を設定")
print("  2. 傾向解析タブ → 上記5ファイルをまとめてアップロード")
print("  3. 各ファイルにラベル（立ち上げ / 1ヶ月後 / ...）を設定して解析実行")
