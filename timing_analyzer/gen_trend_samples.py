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
    '供給工程.サイクル開始',
    '供給工程.投入要求',
    '供給工程.スライダ前進',
    '供給工程.部品検出',
    '供給工程.クランプON',
    '圧入工程.サイクル開始',
    '圧入工程.プレス下降',
    '圧入工程.プレス接触',
    '圧入工程.圧入完了',
    '圧入工程.プレス上昇',
    '検査排出工程.サイクル開始',
    '検査排出工程.カメラトリガ',
    '検査排出工程.検査完了',
    '検査排出工程.ゲート開',
    '検査排出工程.排出完了',
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
    data['供給工程.実行中']             = np.zeros(total_rows, dtype=np.int8)
    data['圧入工程.実行中']             = np.zeros(total_rows, dtype=np.int8)
    data['検査排出工程.実行中']         = np.zeros(total_rows, dtype=np.int8)
    data['圧入工程.ステップ番号']       = np.zeros(total_rows, dtype=np.int8)
    data['ライン.稼働工程数']           = np.zeros(total_rows, dtype=np.int8)
    data['供給工程.クランプ圧力[MPa]']   = np.zeros(total_rows)
    data['供給工程.スライダ変位[mm]']    = np.zeros(total_rows)
    data['圧入工程.プレス力[kN]']        = np.zeros(total_rows)
    data['圧入工程.プレス変位[mm]']      = np.zeros(total_rows)
    data['圧入工程.モータトルク[N·m]']  = np.zeros(total_rows)
    data['検査排出工程.検査スコア']      = np.zeros(total_rows)
    data['検査排出工程.照度センサ[lx]']  = np.zeros(total_rows)

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

        # ── 供給工程 ─────────────────────────────────────────────
        pa_base  = base
        pulse('供給工程.サイクル開始', pa_base)

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
        fill_range('供給工程.投入要求',   pa_load,   pa_slider)
        fill_range('供給工程.スライダ前進', pa_slider, pa_detect)
        fill_range('供給工程.部品検出',   pa_detect, pa_clamp)
        fill_range('供給工程.クランプON', pa_clamp,  pa_end)

        for t in range(pa_clamp, min(pa_clamp + 150, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pa_clamp) / 150.0
            if   p < 0.25: v = 8.0 * p / 0.25
            elif p < 0.75: v = 8.0
            else:          v = 8.0 * (1.0 - (p - 0.75) / 0.25)
            data['供給工程.クランプ圧力[MPa]'][t] = max(0.0, v + rng.normal(0, 0.12))

        slider_span = max(1, pa_detect - pa_slider)
        for t in range(pa_slider, min(pa_end, base + CYCLE_MS)):
            if t >= total_rows: break
            progress = min(1.0, (t - pa_slider) / slider_span)
            data['供給工程.スライダ変位[mm]'][t] = 120.0 * progress + rng.normal(0, 0.3)

        # ── 圧入工程 ─────────────────────────────────────────────
        pb_base = base
        pulse('圧入工程.サイクル開始', pb_base)

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
        fill_range('圧入工程.プレス下降', pb_down,     pb_contact)
        fill_range('圧入工程.プレス接触', pb_contact,  pb_complete)
        fill_range('圧入工程.圧入完了',  pb_complete, pb_up)
        fill_range('圧入工程.プレス上昇', pb_up,       pb_end)

        peak_f = 25.0 + (1.5 if (cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme) else 0.0)
        # 劣化でピーク力が上昇（摩擦増加）
        peak_f += pb_contact_drift * 0.05
        span   = max(1, pb_complete - pb_contact)
        for t in range(pb_contact, min(pb_up + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pb_contact) / span
            if   p < 0.20: v = peak_f * p / 0.20
            elif p < 0.85: v = peak_f
            else:          v = peak_f * max(0.0, 1.0 - (p - 0.85) / 0.15) * 0.5
            data['圧入工程.プレス力[kN]'][t] = max(0.0, v + rng.normal(0, 0.25 * pb_noise_scale))

        disp_peak = 40.0 + (3.0 if cyc in anomaly_pb_extreme else 0.0)
        press_span = max(1, pb_up - pb_down)
        for t in range(pb_down, min(pb_up + 50, base + CYCLE_MS)):
            if t >= total_rows: break
            p = min(1.0, (t - pb_down) / press_span)
            if p < 0.6:
                disp = disp_peak * np.sin(p / 0.6 * np.pi / 2)
            else:
                disp = disp_peak * (1.0 - (p - 0.6) / 0.4)
            data['圧入工程.プレス変位[mm]'][t] = max(0.0, disp + rng.normal(0, 0.15 * pb_noise_scale))

        torque_peak = 12.0 + pb_contact_drift * 0.08 + (1.0 if (cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme) else 0.0)
        for t in range(pb_contact, min(pb_complete + 20, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pb_contact) / max(1, pb_complete - pb_contact)
            if   p < 0.15: v = torque_peak * p / 0.15
            elif p < 0.90: v = torque_peak * (1.0 + 0.1 * np.sin(p * np.pi))
            else:          v = torque_peak * (1.0 - (p - 0.90) / 0.10)
            data['圧入工程.モータトルク[N·m]'][t] = max(0.0, v + rng.normal(0, 0.2 * pb_noise_scale))

        # ── 検査排出工程 ─────────────────────────────────────────
        pc_base = base
        pulse('検査排出工程.サイクル開始', pc_base)

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
        fill_range('検査排出工程.カメラトリガ', pc_cam,     pc_inspect)
        fill_range('検査排出工程.検査完了',     pc_inspect, pc_gate)
        fill_range('検査排出工程.ゲート開',     pc_gate,    pc_eject)
        fill_range('検査排出工程.排出完了',     pc_eject,   pc_end)

        score_base = 58.0 if is_ng else 94.0 - pc_inspect_drift * 0.3
        score_std  =  4.0 if is_ng else  2.0 + pc_inspect_drift * 0.05
        for t in range(pc_cam, min(pc_inspect + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            data['検査排出工程.検査スコア'][t] = float(
                np.clip(rng.normal(score_base, score_std), 0, 100))

        illum_peak = 850.0 - pc_inspect_drift * 2.0 if not is_ng else 820.0
        illum_span = max(1, pc_inspect - pc_cam)
        for t in range(pc_cam, min(pc_inspect + 10, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pc_cam) / illum_span
            if   p < 0.05: v = illum_peak * p / 0.05
            elif p < 0.95: v = illum_peak
            else:          v = illum_peak * (1.0 - (p - 0.95) / 0.05)
            data['検査排出工程.照度センサ[lx]'][t] = max(0.0, v + rng.normal(0, 5.0))

        # ── 実行中信号 & ステップカウンタ ────────────────────────
        def _c(v): return int(np.clip(v, 0, total_rows))

        data['供給工程.実行中'][_c(base):_c(pa_end)] = 1
        data['圧入工程.実行中'][_c(base):_c(pb_end)] = 1
        data['検査排出工程.実行中'][_c(base):_c(pc_end)] = 1

        data['圧入工程.ステップ番号'][_c(base)       : _c(pb_down)    ] = 1
        data['圧入工程.ステップ番号'][_c(pb_down)    : _c(pb_contact) ] = 2
        data['圧入工程.ステップ番号'][_c(pb_contact) : _c(pb_complete)] = 3
        data['圧入工程.ステップ番号'][_c(pb_complete): _c(pb_end)     ] = 4

        ends = sorted([pa_end, pb_end, pc_end])
        data['ライン.稼働工程数'][_c(base)    : _c(ends[0])] = 3
        data['ライン.稼働工程数'][_c(ends[0]) : _c(ends[1])] = 2
        data['ライン.稼働工程数'][_c(ends[1]) : _c(ends[2])] = 1

    # ── DataFrame 構築 ─────────────────────────────────────────
    ts_list = []
    cur = start_dt
    for _ in range(total_rows):
        ts_list.append(cur.strftime('%Y-%m-%d %H:%M:%S.%f'))
        cur += timedelta(milliseconds=1)

    df = pd.DataFrame({'Date Time': ts_list})
    df['圧入工程.ステップ番号'] = data['圧入工程.ステップ番号'].astype(int)
    df['ライン.稼働工程数']     = data['ライン.稼働工程数'].astype(int)
    for c in BOOL_COLS:
        df[c] = data[c].astype(int)
    df['供給工程.実行中']             = data['供給工程.実行中'].astype(int)
    df['圧入工程.実行中']             = data['圧入工程.実行中'].astype(int)
    df['検査排出工程.実行中']         = data['検査排出工程.実行中'].astype(int)
    df['供給工程.クランプ圧力[MPa]']   = np.round(data['供給工程.クランプ圧力[MPa]'],   3)
    df['供給工程.スライダ変位[mm]']    = np.round(data['供給工程.スライダ変位[mm]'],    2)
    df['圧入工程.プレス力[kN]']        = np.round(data['圧入工程.プレス力[kN]'],        3)
    df['圧入工程.プレス変位[mm]']      = np.round(data['圧入工程.プレス変位[mm]'],      2)
    df['圧入工程.モータトルク[N·m]']  = np.round(data['圧入工程.モータトルク[N·m]'],   3)
    df['検査排出工程.検査スコア']      = np.round(data['検査排出工程.検査スコア'],       2)
    df['検査排出工程.照度センサ[lx]']  = np.round(data['検査排出工程.照度センサ[lx]'],  1)

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
