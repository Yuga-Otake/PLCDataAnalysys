"""
views/process.py — 工程とステップ

旧画面のウィザード・自動検出・ステップ追加・ステップ編集・サイクル設定を 1 画面にまとめる。
上から順に「1. サイクル開始信号 → 2. ステップ → 3. タイミングの確認」と進む一本道。
CSV の事前診断（core/profile.py）から、工程・動作シーケンスの仮説を提案し、ワンクリックで採用できる。
"""
import pandas as pd
import streamlit as st

from core.analysis import (MODE_LABELS, analyze, cycle_starts, default_color, describe_step,
                           gantt_figure, mean_cycle_ms, step_candidates, step_stats,
                           trigger_candidates)
from core.profile import active_bool_vars, cycle_health, group_hypotheses, sequence_hypothesis
from core.state import add_process, delete_process, pk, ref_entry, rename_process
from core.ui import chip, page_header

ss = st.session_state
page_header(
    "工程とステップ",
    "1 サイクルの始まりを示す信号で工程を区切り、その中で時間を測りたい動作をステップとして選びます。",
    eyebrow="準備",
)
for msg in ss.pop("_proc_msgs", []):
    st.toast(msg)


def _rerun_with(msg: str):
    ss["_proc_msgs"] = [msg]
    st.rerun()


ref_key, ref = ref_entry()
if not ref:
    st.info("先に基準データを読み込んでください。")
    if st.button("データ画面へ", type="primary"):
        st.switch_page("views/data.py")
    st.stop()

df = ref["df"]
bool_cols = [c for c, t in ref["col_types"].items() if t == "bool"]
num_cols = [c for c, t in ref["col_types"].items() if t == "numeric"]
processes = ss.setdefault("processes", {})
trig_counts = dict(trigger_candidates(df, tuple(bool_cols)))


def _signal_label(v: str) -> str:
    return f"{v}（立ち上がり {trig_counts.get(v, 0)} 回）"


# ── 工程の選択・追加 ─────────────────────────────────────────
with st.popover("工程を追加", icon=":material/add:"):
    new_name = st.text_input("工程名", placeholder="例: 圧入工程", key="_proc_new_name")
    by_count = [v for v, _ in trigger_candidates(df, tuple(bool_cols))]
    new_sig = st.selectbox("サイクル開始信号", by_count, format_func=_signal_label, key="_proc_new_sig",
                           help="1 サイクルに 1 回だけ立ち上がる Bool 変数を選びます。回数の多い順に並んでいます")
    if st.button("登録", type="primary", disabled=not new_name.strip(), key="_proc_new_add"):
        name = new_name.strip()
        if name in processes:
            st.error(f"「{name}」は登録済みです")
        else:
            add_process(name, new_sig)
            ss["sel_process"] = name
            _rerun_with(f"工程「{name}」を登録しました")

# ── 工程の提案（変数名のまとまりと、周期的に立ち上がる信号から推定）──────────
active = active_bool_vars(ref)
groups = group_hypotheses(df, tuple(active))
taken = {ss.get(pk(p, "trigger")) for p in processes}
proposals = [g for g in groups if g["trigger"] and g["trigger"] not in taken and g["name"] not in processes]


def _adopt(gs: list):
    for g in gs:
        add_process(g["name"], g["trigger"])
    ss["sel_process"] = gs[0]["name"]
    ss["_proc_msgs"] = [f"工程を {len(gs)} 件作成しました。次にステップの仮説を確認してください"]


if proposals:
    with st.container(border=True):
        h1, h2 = st.columns([4, 1.3], vertical_alignment="center")
        h1.markdown("**工程の提案**　変数名のまとまりと、周期的に立ち上がる信号から推定しました。")
        if len(proposals) > 1:
            h2.button(f"{len(proposals)} 件すべて作成", type="primary", key="_proc_prop_all",
                      on_click=_adopt, args=(proposals,), width="stretch")
        for g in proposals:
            c1, c2, c3, c4 = st.columns([1.6, 2.6, 2.2, 1], vertical_alignment="center")
            c1.markdown(f"**{g['name']}**")
            c2.markdown(f"<span class='apb-crumb'>サイクル開始信号</span><br>{g['trigger']}", unsafe_allow_html=True)
            c3.markdown(f"<span class='apb-crumb'>{g['reason']}</span><br>"
                        f"周期 {g['period_ms']:.0f} ms・{g['n_cycles']} サイクル・変数 {len(g['vars'])}",
                        unsafe_allow_html=True)
            c4.button("作成", key=f"_proc_prop_{g['name']}", on_click=_adopt, args=([g],), width="stretch")

if not processes:
    st.info("工程がまだありません。上の提案から作成するか、「工程を追加」から登録してください。")
    st.stop()

names = list(processes)
if ss.get("sel_process") not in names:
    ss["sel_process"] = names[0]
sel_col, info_col = st.columns([2, 3], vertical_alignment="bottom")
with sel_col:
    pname = st.selectbox("工程", names, key="sel_process")
trigger = ss.get(pk(pname, "trigger"))
if trigger not in bool_cols:
    ss[pk(pname, "trigger")] = trigger = bool_cols[0]
edge = ss.get(pk(pname, "edge"), "RISE")
starts = cycle_starts(df, trigger, edge)
with info_col:
    cyc = mean_cycle_ms(df, starts)
    st.markdown(
        f"<div class='apb-kv'><span>基準データのサイクル数 <b class='apb-num'>{len(starts)}</b></span>"
        f"<span>平均周期 <b class='apb-num'>{cyc:.0f} ms</b></span>"
        f"<span>ステップ <b class='apb-num'>{len(ss.get(pk(pname, 'steps_list'), []))}</b></span></div>",
        unsafe_allow_html=True,
    )

# ── 1. サイクル開始信号 ──────────────────────────────────────
st.subheader("1. サイクル開始信号", anchor=False)
with st.container(border=True):
    c1, c2, c3 = st.columns([3, 2, 1.4])
    with c1:
        st.selectbox("信号", bool_cols, key=pk(pname, "trigger"), format_func=_signal_label)
    with c2:
        if ss.get(pk(pname, "edge")) not in ("RISE", "FALL"):
            ss[pk(pname, "edge")] = "RISE"
        st.radio("サイクル開始のタイミング", ["RISE", "FALL"], key=pk(pname, "edge"), horizontal=True,
                 format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
    with c3:
        st.number_input("タクト目標 [ms]", min_value=0, step=10, key=pk(pname, "takt"),
                        help="0 のときは表示しません")
    if len(starts) == 0:
        st.warning("この信号ではサイクルが見つかりません。別の信号か、立ち下がりを試してください。")
    elif len(starts) == 1:
        st.caption("サイクルが 1 つしか見つかっていません。ばらつきを見るには 2 サイクル以上のデータが必要です。")
    health = cycle_health(df, trigger, edge)
    if health["n"] >= 3:
        outl = health["outliers"]
        line = (f"<span class='apb-kv'><span>周期 <b class='apb-num'>{health['median_ms']:.0f} ms</b></span>"
                f"<span>ばらつき <b class='apb-num'>{health['cv_pct']:.1f}%</b></span></span>")
        if outl:
            listed = "、".join(f"#{c} {r}" for c, r in list(outl.items())[:3]) + ("…" if len(outl) > 3 else "")
            clean = health["clean"]
            line += (f"<br>{chip(f'外れサイクル {len(outl)}', 'warn')}　<span class='apb-crumb'>{listed}。"
                     "基準値の算出では外して計算します"
                     + (f"（きれいな範囲は #{clean[0]}〜#{clean[1]}）" if clean else "") + "</span>")
        st.markdown(line, unsafe_allow_html=True)

    with st.expander("工程名の変更・削除"):
        # 選択中の工程（sel_process）を書き換えるので、ウィジェット描画前に走る on_click で処理する
        def _do_rename(old: str, key: str):
            new = ss.get(key, "").strip()
            if new in ss["processes"]:
                ss["_proc_err"] = f"「{new}」は登録済みです"
                return
            rename_process(old, new)
            ss["sel_process"] = new
            ss["_proc_msgs"] = ["工程名を変更しました"]

        def _do_delete(name: str):
            delete_process(name)
            ss.pop("sel_process", None)
            ss["_proc_msgs"] = [f"工程「{name}」を削除しました"]

        if ss.get("_proc_err"):
            st.error(ss.pop("_proc_err"))
        r1, r2 = st.columns([3, 1], vertical_alignment="bottom")
        rkey = f"_proc_rename_{pname}"
        renamed = r1.text_input("工程名", value=pname, key=rkey)
        r2.button("名前を変更", disabled=(not renamed.strip() or renamed.strip() == pname),
                  key=f"_proc_rename_btn_{pname}", on_click=_do_rename, args=(pname, rkey))
        confirm = st.checkbox(f"「{pname}」とそのステップ・基準値を削除する", key=f"_proc_del_ok_{pname}")
        st.button("工程を削除", disabled=not confirm, key=f"_proc_del_{pname}",
                  on_click=_do_delete, args=(pname,))

# ── 2. ステップ ─────────────────────────────────────────────
st.subheader("2. ステップ", anchor=False)
steps = list(ss.get(pk(pname, "steps_list"), []))


def _save_steps(new_steps: list):
    ss[pk(pname, "steps_list")] = new_steps


# 2-a. ステップの追加
used_vars = {s.get("variable") for s in steps} | {s.get("start_var") for s in steps}
cands = [c for c in step_candidates(df, tuple(active), trigger, edge) if c["var"] not in used_vars]

# 2-a-1. 動作シーケンスの仮説（知っている人が最小限の修正で直せるように、表で直してから登録する）
grp = next((g for g in groups if trigger in g["vars"]), None)
with st.expander("動作シーケンスの仮説から登録", expanded=not steps):
    if not starts:
        st.caption("サイクル開始信号を決めると、動作の順番の仮説が表示されます。")
    else:
        t1, t2 = st.columns([4, 2], vertical_alignment="center")
        t1.caption("1 サイクルの中で変数が変化する順番を、全サイクルの中央値で並べた仮説です。"
                   "違うものはチェックを外し、順番・名前・測り方を直してから登録してください。"
                   "確からしさが「低」のものは、見つからないサイクルや、順番が入れ替わるサイクルがあります。")
        widen = t2.toggle("他の工程の変数も含める", key=f"_proc_seq_all_{pname}", value=grp is None,
                          help=f"オフのときは「{grp['name']}」の変数だけで仮説を作ります" if grp else None)
        pool = active if (widen or grp is None) else grp["vars"]
        seq = sequence_hypothesis(df, trigger, edge, tuple(v for v in pool if v not in used_vars),
                                  "" if widen or grp is None else grp["name"])
        if not seq:
            st.caption("仮説にできる変化が見つかりませんでした。")
        else:
            how = {"on_period": "ON期間", "RISE": "立ち上がり時刻", "FALL": "立ち下がり時刻"}
            kind_txt = {"RISE": "立ち上がる", "ON": "開始時点で ON→切れる", "FALL": "立ち下がる"}

            def _why(r):
                parts = []
                if r["det_rate"] < 0.95:
                    parts.append(f"{(1 - r['det_rate']) * 100:.0f}% のサイクルで見つからない")
                if r["swap_rate"] > 0:
                    parts.append(f"{r['swap_rate'] * 100:.0f}% のサイクルで次と順番が入れ替わる")
                return "、".join(parts)

            seq_df = pd.DataFrame([{
                "採用": r["include"], "順番": i + 1, "ステップ名": r["label"],
                "測り方": how["on_period"] if r["mode"] == "on_period" else how[r["edge"]],
                "変数": r["var"], "変化": kind_txt[r["kind"]], "時刻 [ms]": round(r["t_ms"], 1),
                "ばらつき [ms]": round(r["spread_ms"], 1), "確からしさ": r["confidence"], "理由": _why(r),
            } for i, r in enumerate(seq)])
            edited_seq = st.data_editor(
                seq_df, hide_index=True, width="stretch", key=f"_proc_seq_{pname}_{trigger}_{edge}_{len(steps)}_{widen}",
                disabled=["変数", "変化", "時刻 [ms]", "ばらつき [ms]", "確からしさ", "理由"],
                column_config={
                    "採用": st.column_config.CheckboxColumn(width="small"),
                    "順番": st.column_config.NumberColumn(min_value=1, step=1, width="small"),
                    "測り方": st.column_config.SelectboxColumn(options=list(how.values()), required=True),
                    "時刻 [ms]": st.column_config.NumberColumn(help="サイクル開始からの時刻（中央値）"),
                    "ばらつき [ms]": st.column_config.NumberColumn(help="時刻の四分位範囲（IQR）"),
                },
            )
            adopt = edited_seq[edited_seq["採用"]].sort_values("順番", kind="stable")
            if st.button(f"採用した {len(adopt)} 件をステップとして登録", type="primary", disabled=adopt.empty,
                         key=f"_proc_seq_add_{pname}"):
                inv = {v: k for k, v in how.items()}
                for _, r in adopt.iterrows():
                    m = inv[r["測り方"]]
                    step = {"name": (r["ステップ名"] or r["変数"]).strip(), "color": default_color(len(steps)),
                            "variable": r["変数"]}
                    step.update({"mode": "on_period", "edge": "RISE"} if m == "on_period" else {"mode": "single", "edge": m})
                    steps.append(step)
                _save_steps(steps)
                _rerun_with(f"仮説から {len(adopt)} 件のステップを登録しました")

with st.expander(f"検索して追加（候補 {len(cands)} 件）", expanded=False):
    if not starts:
        st.caption("サイクル開始信号を決めると候補が表示されます。")
    elif not cands:
        st.caption("追加できる候補はありません。")
    else:
        st.caption("サイクルの中で状態が変わる Bool 変数が候補です。名前で検索し、下のプルダウンでまとめて選んでから追加します。"
                   "検索を変えても、選んだ変数はプルダウンに残ります。")
        kind_label = {"RISE": "立ち上がる", "ON": "開始時点で ON", "FALL": "立ち下がる"}
        by_var = {c["var"]: c for c in cands}
        order = {c["var"]: i for i, c in enumerate(cands)}
        q1, q2 = st.columns([3, 2], vertical_alignment="bottom")
        query = q1.text_input("変数名で検索", key=f"_proc_q_{pname}", placeholder="例: 圧入 プレス（スペース区切りで、すべて含むものに絞り込み）")
        kinds = q2.pills("変化の種類", list(kind_label), selection_mode="multi", default=list(kind_label),
                         format_func=kind_label.get, key=f"_proc_kinds_{pname}")
        words = query.lower().split()
        hits = [c["var"] for c in cands
                if c["kind"] in (kinds or []) and all(w in c["var"].lower() for w in words)]

        # 選択はプルダウン（multiselect）に持つ。検索で候補が絞られても、選んだものは選択肢に残す
        pick_key = f"_proc_pick_{pname}_{trigger}_{edge}"
        chosen = [v for v in ss.get(pick_key, []) if v in by_var]
        ss[pick_key] = chosen

        def _select_hits(chosen=chosen, hits=hits):
            ss[pick_key] = sorted(set(chosen) | set(hits), key=order.get)

        b1, b2, b3 = st.columns([2, 1.2, 3], vertical_alignment="center")
        b1.button(f"検索結果 {len(hits)} 件をすべて選択", key=f"_proc_pick_all_{pname}", disabled=not hits,
                  on_click=_select_hits, icon=":material/done_all:")
        b2.button("選択をクリア", key=f"_proc_pick_clear_{pname}", disabled=not chosen,
                  on_click=lambda: ss.update({pick_key: []}))
        b3.caption(f"候補 {len(cands)} 件のうち {len(hits)} 件が一致")
        options = chosen + [v for v in hits if v not in chosen]
        picked = st.multiselect(
            "追加する変数", options, key=pick_key,
            format_func=lambda v: f"{v}｜{kind_label[by_var[v]['kind']]}  {by_var[v]['t_ms']:.0f} ms",
            placeholder="クリックすると検索結果が一覧されます（ここに入力しても絞り込めます）",
            help="時刻はサイクル開始からの中央値。開始時点で ON の変数は ON の継続時間です",
        )
        a1, a2 = st.columns([3, 1], vertical_alignment="bottom")
        add_as = a1.radio("測り方", ["on_period", "single"], horizontal=True, key="_proc_cand_mode",
                          format_func=lambda m: {"on_period": "ON の間の長さ（開始時刻と長さ）",
                                                 "single": "立ち上がる時刻だけ"}[m])
        if a2.button(f"{len(picked)} 件を追加", type="primary", disabled=not picked,
                     key=f"_proc_cand_add_{pname}"):
            for v in sorted(picked, key=order.get):
                steps.append({"name": v, "color": default_color(len(steps)), "mode": add_as,
                              "variable": v, "edge": "RISE"})
            _save_steps(steps)
            _rerun_with(f"ステップを {len(picked)} 件追加しました")

# 2-b. 登録済みステップ
if steps:
    st.caption("行を選ぶと、名前・測り方・色の変更や並べ替え、削除ができます。")
    sel = st.dataframe(
        pd.DataFrame([{"#": i + 1, "ステップ": s.get("name", ""), "測り方": MODE_LABELS.get(s.get("mode"), ""),
                       "計測内容": describe_step(s)} for i, s in enumerate(steps)]),
        hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"_proc_steps_{pname}_{len(steps)}",
        column_config={"#": st.column_config.NumberColumn(width="small")},
    )
    rows = sel.selection.rows if sel else []
    if rows:
        idx = rows[0]
        step = steps[idx]
        kid = f"{pname}_{idx}_{step.get('name', '')}"
        with st.container(border=True):
            st.markdown(f"**{step.get('name', '')}** を編集")
            e1, e2 = st.columns([4, 1])
            new_label = e1.text_input("ステップ名", value=step.get("name", ""), key=f"_pe_name_{kid}")
            new_color = e2.color_picker("色", value=step.get("color", default_color(idx)), key=f"_pe_color_{kid}")
            modes = ["single", "on_period", "range", "numeric"]
            mode = st.radio("測り方", modes, index=modes.index(step.get("mode", "single")), horizontal=True,
                            format_func=lambda m: MODE_LABELS[m], key=f"_pe_mode_{kid}")
            var_opts = [v for v in bool_cols if v != trigger]
            upd = {"mode": mode}
            if mode in ("single", "on_period"):
                cur = step.get("variable") or step.get("start_var") or var_opts[0]
                upd["variable"] = st.selectbox("変数", var_opts, key=f"_pe_var_{kid}",
                                               index=var_opts.index(cur) if cur in var_opts else 0)
                if mode == "single":
                    upd["edge"] = st.radio("タイミング", ["RISE", "FALL"], horizontal=True, key=f"_pe_edge_{kid}",
                                           index=0 if step.get("edge", "RISE") == "RISE" else 1,
                                           format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
            elif mode == "range":
                r1, r2, r3, r4 = st.columns([3, 1.4, 3, 1.4])
                sv = step.get("start_var") or step.get("variable") or var_opts[0]
                ev = step.get("end_var") or step.get("variable") or var_opts[0]
                upd["start_var"] = r1.selectbox("開始の変数", var_opts, key=f"_pe_sv_{kid}",
                                                index=var_opts.index(sv) if sv in var_opts else 0)
                upd["start_edge"] = r2.radio("開始", ["RISE", "FALL"], key=f"_pe_se_{kid}",
                                             index=0 if step.get("start_edge", "RISE") == "RISE" else 1,
                                             format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
                upd["end_var"] = r3.selectbox("終了の変数", var_opts, key=f"_pe_ev_{kid}",
                                              index=var_opts.index(ev) if ev in var_opts else 0)
                upd["end_edge"] = r4.radio("終了", ["RISE", "FALL"], key=f"_pe_ee_{kid}",
                                           index=0 if step.get("end_edge", "FALL") == "RISE" else 1,
                                           format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
            else:
                if not num_cols:
                    st.caption("アナログ変数がないため、数値条件は使えません。")
                else:
                    n1, n2, n3 = st.columns([3, 1, 2])
                    nv = step.get("variable") if step.get("variable") in num_cols else num_cols[0]
                    ops = ["==", ">=", "<=", ">", "<"]
                    upd["variable"] = n1.selectbox("アナログ変数", num_cols, index=num_cols.index(nv),
                                                   key=f"_pe_nvar_{kid}")
                    upd["op"] = n2.selectbox("条件", ops, key=f"_pe_op_{kid}",
                                             index=ops.index(step.get("op", "==")) if step.get("op") in ops else 0)
                    upd["value"] = n3.number_input("値", value=float(step.get("value", 0)), key=f"_pe_val_{kid}")
            if num_cols:
                upd["waveform_vars"] = st.multiselect(
                    "一緒に波形を見るアナログ変数（任意）", num_cols, key=f"_pe_wv_{kid}", placeholder="なし",
                    default=[v for v in step.get("waveform_vars", []) if v in num_cols])
            b1, b2, b3, _, b5 = st.columns([1.3, 1, 1, 2, 1])
            if b1.button("変更を反映", type="primary", key=f"_pe_apply_{kid}"):
                clean = {k: v for k, v in step.items()
                         if k not in ("variable", "edge", "start_var", "start_edge", "end_var", "end_edge",
                                      "op", "value")}
                steps[idx] = {**clean, **upd, "name": new_label.strip() or step.get("name", ""),
                              "color": new_color}
                _save_steps(steps)
                _rerun_with("ステップを更新しました")
            if b2.button("上へ", disabled=idx == 0, key=f"_pe_up_{kid}", icon=":material/arrow_upward:"):
                steps[idx - 1], steps[idx] = steps[idx], steps[idx - 1]
                _save_steps(steps)
                st.rerun()
            if b3.button("下へ", disabled=idx == len(steps) - 1, key=f"_pe_dn_{kid}",
                         icon=":material/arrow_downward:"):
                steps[idx + 1], steps[idx] = steps[idx], steps[idx + 1]
                _save_steps(steps)
                st.rerun()
            if b5.button("削除", key=f"_pe_del_{kid}", icon=":material/delete:"):
                removed = steps.pop(idx)
                _save_steps(steps)
                _rerun_with(f"「{removed.get('name', '')}」を削除しました")

# 2-c. 手動で追加
with st.expander("手動で追加（開始〜終了・数値条件など）"):
    m_mode = st.radio("測り方", ["range", "numeric", "single", "on_period"], horizontal=True, key="_pm_mode",
                      format_func=lambda m: MODE_LABELS[m])
    var_opts = [v for v in bool_cols if v != trigger]
    new_step = {"mode": m_mode, "color": default_color(len(steps))}
    if m_mode == "range":
        r1, r2, r3, r4 = st.columns([3, 1.4, 3, 1.4])
        new_step["start_var"] = r1.selectbox("開始の変数", var_opts, key="_pm_sv")
        new_step["start_edge"] = r2.radio("開始", ["RISE", "FALL"], key="_pm_se",
                                          format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
        new_step["end_var"] = r3.selectbox("終了の変数", var_opts, key="_pm_ev")
        new_step["end_edge"] = r4.radio("終了", ["RISE", "FALL"], index=1, key="_pm_ee",
                                        format_func=lambda x: "立ち上がり" if x == "RISE" else "立ち下がり")
        default_name = f"{new_step['start_var']}〜{new_step['end_var']}"
    elif m_mode == "numeric":
        if not num_cols:
            st.caption("アナログ変数がないため、数値条件は使えません。")
            default_name = ""
        else:
            n1, n2, n3 = st.columns([3, 1, 2])
            new_step["variable"] = n1.selectbox("アナログ変数", num_cols, key="_pm_nvar")
            new_step["op"] = n2.selectbox("条件", ["==", ">=", "<=", ">", "<"], key="_pm_op")
            new_step["value"] = n3.number_input("値", value=0.0, key="_pm_val")
            default_name = f"{new_step['variable']}{new_step['op']}{new_step['value']:g}"
    else:
        new_step["variable"] = st.selectbox("変数", var_opts, key="_pm_var")
        new_step["edge"] = "RISE"
        default_name = new_step["variable"]
    m_name = st.text_input("ステップ名", placeholder=default_name or "ステップ名", key="_pm_name")
    if st.button("追加", type="primary", disabled=not default_name, key="_pm_add"):
        new_step["name"] = m_name.strip() or default_name
        if any(s.get("name") == new_step["name"] for s in steps):
            st.error(f"「{new_step['name']}」は同じ工程にすでにあります。名前を変えてください")
        else:
            steps.append(new_step)
            _save_steps(steps)
            _rerun_with(f"「{new_step['name']}」を追加しました")

# ── 3. タイミングの確認 ──────────────────────────────────────
st.subheader("3. タイミングの確認", anchor=False)
if not steps:
    st.caption("ステップを追加すると、基準データでのタイミングがここに表示されます。")
    st.stop()
result = analyze(df, trigger, edge, steps)
stats = step_stats(result, steps)
missing = [s.get("name") for s in steps if s.get("name") not in {x["name"] for x in stats}]
if not stats:
    st.warning("基準データではどのステップも検出できませんでした。変数やサイクル開始信号を確認してください。")
    st.stop()
st.plotly_chart(gantt_figure(stats, int(ss.get(pk(pname, "takt"), 0))), width="stretch",
                key=f"_proc_gantt_{pname}", config={"displayModeBar": False})
if missing:
    st.markdown(chip("未検出", "warn") + "　基準データで見つからなかったステップ: " + "、".join(missing),
                unsafe_allow_html=True)
st.dataframe(
    pd.DataFrame([{
        "ステップ": s["name"], "測り方": MODE_LABELS.get(s["mode"], ""),
        "開始 平均 [ms]": round(s["start_mean"], 1),
        "長さ 平均 [ms]": round(s["dur_mean"], 1) if s["mode"] != "single" else None,
        "ばらつき σ [ms]": round(s["start_std"] if s["mode"] == "single" else s["dur_std"], 2),
        "サイクル数": s["n"],
    } for s in stats]),
    hide_index=True, width="stretch",
)
takt = int(ss.get(pk(pname, "takt"), 0))
end = max(s["end_mean"] for s in stats)
if takt > 0:
    kind = "ok" if end <= takt else "ng"
    st.markdown(chip(f"最後のステップ終了 {end:.0f} ms ／ タクト目標 {takt} ms", kind), unsafe_allow_html=True)

st.divider()
if st.button("基準値の登録へ", type="primary", icon=":material/arrow_forward:", key="_proc_next"):
    st.switch_page("views/baseline.py")
