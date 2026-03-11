"""
gen_sample.py  — サンプルCSV生成スクリプト
自動車部品組み立てライン（3工程）のPLCログデータを生成する

ProA: 供給工程  (部品ピッキング・クランプ)
ProB: 圧入工程  (プレスフィッティング) ← ボトルネック
ProC: 検査排出工程 (カメラ検査・排出)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate(filename, n_cycles, seed,
             anomaly_pa_slow, anomaly_pb_slow, anomaly_pc_slow,
             anomaly_pb_extreme=None):

    rng = np.random.default_rng(seed)
    if anomaly_pb_extreme is None:
        anomaly_pb_extreme = set()

    CYCLE_MS   = 1600   # 1サイクル長
    PULSE_MS   = 50     # 信号パルス幅
    total_rows = n_cycles * CYCLE_MS

    t0 = datetime(2024, 3, 1, 8, 0, 0)

    bool_cols = [
        'ProA_CycleStart',
        'ProA_LoadReq', 'ProA_SliderFwd', 'ProA_PartDetect', 'ProA_ClampON',
        'ProB_CycleStart',
        'ProB_PressDown', 'ProB_PressContact', 'ProB_PressComplete', 'ProB_PressUp',
        'ProC_CycleStart',
        'ProC_CamTrigger', 'ProC_InspectDone', 'ProC_GateOpen', 'ProC_EjectDone',
    ]
    data = {c: np.zeros(total_rows, dtype=np.int8) for c in bool_cols}
    data['ProA_ClampPressure'] = np.zeros(total_rows)
    data['ProB_PressForce']    = np.zeros(total_rows)
    data['ProC_InspectScore']  = np.zeros(total_rows)

    def pulse(col, start, dur=PULSE_MS):
        s = int(np.clip(start, 0, total_rows - 1))
        e = min(s + dur, total_rows)
        data[col][s:e] = 1

    for c in range(n_cycles):
        base = c * CYCLE_MS
        cyc  = c + 1

        # ── ProA (供給工程) ──────────────────────────────────
        pa_base = base
        pulse('ProA_CycleStart', pa_base)

        pa_extra  = 150 if cyc in anomaly_pa_slow else 0
        pa_load   = pa_base + int(rng.normal(50,  5))
        pa_slider = pa_base + int(rng.normal(180, 15))
        pa_detect = pa_base + int(rng.normal(320, 18)) + pa_extra
        pa_clamp  = pa_base + int(rng.normal(400, 18)) + pa_extra

        pa_load   = max(pa_base + 10,   pa_load)
        pa_slider = max(pa_load  + 20,  pa_slider)
        pa_detect = max(pa_slider + 20, pa_detect)
        pa_clamp  = max(pa_detect + 20, pa_clamp)

        pulse('ProA_LoadReq',    pa_load)
        pulse('ProA_SliderFwd',  pa_slider)
        pulse('ProA_PartDetect', pa_detect)
        pulse('ProA_ClampON',    pa_clamp)

        # ClampPressure: 0 → 8MPa → 0 (180ms台形)
        for t in range(pa_clamp, min(pa_clamp + 180, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pa_clamp) / 180.0
            if   p < 0.25: v = 8.0 * p / 0.25
            elif p < 0.75: v = 8.0
            else:          v = 8.0 * (1.0 - (p - 0.75) / 0.25)
            data['ProA_ClampPressure'][t] = max(0.0, v + rng.normal(0, 0.12))

        # ── ProB (圧入工程) ──────────────────────────────────
        pb_base = base + 450
        pulse('ProB_CycleStart', pb_base)

        pb_extra   = 200 if cyc in anomaly_pb_slow    else 0
        pb_extreme = 700 if cyc in anomaly_pb_extreme else 0

        pb_down     = pb_base + int(rng.normal(60,  8))
        pb_contact  = pb_base + int(rng.normal(200, 20))
        pb_complete = pb_base + int(rng.normal(400, 25)) + pb_extra + pb_extreme
        pb_up       = pb_base + int(rng.normal(480, 15)) + pb_extra + pb_extreme

        pb_down     = max(pb_base + 10,     pb_down)
        pb_contact  = max(pb_down  + 20,    pb_contact)
        pb_complete = max(pb_contact + 20,  pb_complete)
        pb_up       = max(pb_complete + 20, pb_up)

        pulse('ProB_PressDown',     pb_down)
        pulse('ProB_PressContact',  pb_contact)
        pulse('ProB_PressComplete', pb_complete)
        pulse('ProB_PressUp',       pb_up)

        peak_f = 25.0 + (1.5 if (cyc in anomaly_pb_slow or cyc in anomaly_pb_extreme) else 0.0)
        span   = max(1, pb_complete - pb_contact)
        for t in range(pb_contact, min(pb_up + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            p = (t - pb_contact) / span
            if   p < 0.20: v = peak_f * p / 0.20
            elif p < 0.85: v = peak_f
            else:          v = peak_f * max(0.0, 1.0 - (p - 0.85) / 0.15) * 0.5
            data['ProB_PressForce'][t] = max(0.0, v + rng.normal(0, 0.25))

        # ── ProC (検査排出工程) ──────────────────────────────
        pc_base = base + 1050
        pulse('ProC_CycleStart', pc_base)

        pc_extra   = 80 if cyc in anomaly_pc_slow else 0
        is_ng      = cyc in anomaly_pc_slow

        pc_cam     = pc_base + int(rng.normal(30,  3))
        pc_inspect = pc_base + int(rng.normal(120, 12)) + pc_extra
        pc_gate    = pc_base + int(rng.normal(175,  8)) + pc_extra
        pc_eject   = pc_base + int(rng.normal(260, 12)) + pc_extra

        pc_cam     = max(pc_base + 10,    pc_cam)
        pc_inspect = max(pc_cam   + 20,   pc_inspect)
        pc_gate    = max(pc_inspect + 10, pc_gate)
        pc_eject   = max(pc_gate   + 20,  pc_eject)

        pulse('ProC_CamTrigger',  pc_cam)
        pulse('ProC_InspectDone', pc_inspect)
        pulse('ProC_GateOpen',    pc_gate)
        pulse('ProC_EjectDone',   pc_eject)

        score_base = 58.0 if is_ng else 94.0
        score_std  =  4.0 if is_ng else  2.0
        for t in range(pc_cam, min(pc_inspect + 30, base + CYCLE_MS)):
            if t >= total_rows: break
            data['ProC_InspectScore'][t] = float(
                np.clip(rng.normal(score_base, score_std), 0, 100))

    # ─── DataFrame構築 ─────────────────────────────────────────
    cur = t0
    ts_list = []
    for _ in range(total_rows):
        ms = cur.microsecond // 1000
        ts_list.append(cur.strftime('%Y/%m/%d %H:%M:%S.') + f'{ms:03d}')
        cur += timedelta(milliseconds=1)

    df = pd.DataFrame({'Timestamp': ts_list})
    for c in bool_cols:
        df[c] = data[c]
    df['ProA_ClampPressure'] = np.round(data['ProA_ClampPressure'], 3)
    df['ProB_PressForce']    = np.round(data['ProB_PressForce'],    3)
    df['ProC_InspectScore']  = np.round(data['ProC_InspectScore'],  2)

    df.to_csv(filename, index=False)
    print(f"OK {filename}  ({len(df):,} rows, {n_cycles} cycles)")


# ─── 通常ファイル ──────────────────────────────────────────────
generate(
    'sample_playback.csv',
    n_cycles=30, seed=42,
    anomaly_pa_slow={10, 23},
    anomaly_pb_slow={7, 14, 21},
    anomaly_pc_slow={18},
)

# ─── 異常ファイル（機械劣化シナリオ） ─────────────────────────
generate(
    'sample_playback_ng.csv',
    n_cycles=30, seed=123,
    anomaly_pa_slow={5, 12, 17, 20, 25},
    anomaly_pb_slow={3, 7, 10, 14, 17, 20, 23, 26, 29},
    anomaly_pb_extreme={25},
    anomaly_pc_slow={6, 11, 15, 19, 22, 28},
)

# ─── 検証 ──────────────────────────────────────────────────────
for fname in ['sample_playback.csv', 'sample_playback_ng.csv']:
    df = pd.read_csv(fname)
    print(f"\n--- {fname} ---")
    print("columns:", df.columns.tolist())
    print("shape  :", df.shape)
    CYCLE_MS = 1600
    print("cyc | ProA_PartDetect | ProB_PressComplete | ProC_InspectDone")
    for c in range(min(5, 30)):
        base = c * CYCLE_MS
        def rise(col, b=base):
            s = df[col].iloc[b:b+CYCLE_MS]
            rs = s[s.diff().fillna(0) > 0].index
            return (int(rs[0]) - b) if len(rs) else -1
        print(f"  {c+1:2d} | {rise('ProA_PartDetect'):5d} ms"
              f"         | {rise('ProB_PressComplete'):5d} ms"
              f"           | {rise('ProC_InspectDone'):5d} ms")
