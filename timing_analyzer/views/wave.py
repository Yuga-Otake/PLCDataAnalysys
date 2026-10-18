"""
views/wave.py — 波形検査条件（準備の詳細設定）

アナログ変数の波形に検出点・良品範囲・検査ウィンドウを設定する。
旧メイン画面の「波形検査」タブを移したもの。検査ロジックは core/wave_inspect.py（移設のみで内容は同一）。
設定した検出点のうち「傾向解析に出す」をオンにしたものは、推移画面に表示される。
"""
import pandas as pd
import streamlit as st

from core.profile import WAVE_POINT_TYPES, wave_candidates
from core.state import pk, ref_entry
from core.ui import page_header
from core.wave_inspect import (_render_waveform_overlay, _wi_add_dialog, _wi_edit_dialog,
                               _wi_load_from_file, _wi_restore_config, _wi_save_config,
                               _wi_save_to_file, adopt_point, cached_detect_cycles,
                               organized_points, registered_points, set_columns)

# 重ね表示の色（CSV ごと）
_CMP_PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]

page_header(
    "波形検査条件",
    "アナログ値の波形から検出点（変化点・しきい値の通過など）や良品範囲を設定します。"
    "「傾向解析に出す」をオンにした検出点は、推移画面で時期ごとの変化を確認できます。",
    eyebrow="準備（詳細設定）",
)

ss = st.session_state
store = ss.get("csv_store", {})
ref_key, ref = ref_entry()
if not store:
    st.info("先にデータを読み込んでください。")
    if st.button("データ画面へ", type="primary"):
        st.switch_page("views/data.py")
    st.stop()

keys = list(store)
if ss.get("wave_data") not in keys:
    ss["wave_data"] = ref_key or keys[0]
c1, c2 = st.columns(2)
data_key = c1.selectbox("表示するデータ", keys, key="wave_data",
                        format_func=lambda k: ("基準｜" if store[k].get("is_ref") else "比較｜") + store[k]["label"])
ss["wave_overlay"] = [k for k in ss.get("wave_overlay", []) if k in keys and k != data_key]
overlay_keys = c2.multiselect("重ねて比べるデータ（任意）", [k for k in keys if k != data_key],
                              key="wave_overlay", format_func=lambda k: store[k]["label"], placeholder="なし",
                              help="選ばない場合、比較データを表示しているときは基準データを重ねます")
df = store[data_key]["df"]
col_types = store[data_key]["col_types"]
bool_cols = [c for c, t in col_types.items() if t == "bool"]
num_cols = [c for c, t in col_types.items() if t == "numeric"]
set_columns(bool_cols, num_cols)
for msg in ss.pop("_wave_msgs", []):
    st.success(msg)


# ── 検出点の候補（基準データの事前解析）────────────────────────────
def _candidate_panel():
    processes = {n: {"trigger_col": ss.get(pk(n, "trigger"), p.get("trigger_col", "")),
                     "edge": ss.get(pk(n, "edge"), p.get("edge", "RISE"))}
                 for n, p in ss.get("processes", {}).items()}
    if not ref:
        st.caption("基準データを読み込むと、波形の形から検出点の候補を出します。")
        return
    if not processes:
        st.caption("工程を登録すると、そのサイクル開始信号を基準に、波形の形から検出点の候補を出します。")
        if st.button("工程とステップへ", key="_wave_to_process"):
            st.switch_page("views/process.py")
        return
    with st.spinner("基準データの波形を調べています…"):
        hyp = wave_candidates(ref, processes)
    rows, n_done = [], 0
    for var, h in hyp.items():
        done = registered_points(var)
        for i, c in enumerate(h["cands"]):
            if (c["dtype"], c["name"]) in done:
                n_done += 1
                continue
            hint = "—" if pd.isna(c["t_ms"]) else f"{c['t_ms']:.0f} ms／{c['v']:.4g}"
            rows.append({"採用": c["include"], "波形": h["spark"], "変数": var, "工程": h["process"],
                         "形": h["shape"], "検出点タイプ": c["dtype"], "名前": c["name"], "目安": hint,
                         "確からしさ": c["confidence"], "傾向解析に出す": True, "説明": c["note"],
                         "_id": f"{var}|{i}"})
    if not rows:
        st.caption("候補はすべて登録済みです。" if n_done else
                   "検出点の候補になるアナログ波形は見つかりませんでした（変化しない変数・状態番号は除いています）。")
        return
    st.caption("基準データのサイクルごとの波形の形から、検出点のタイプと設定値（しきい値など）を仮に決めたものです。"
               "違うものはチェックを外し、タイプや名前を直してから登録してください。"
               "登録後は下の一覧の ⚙️ で、波形を見ながら調整できます。"
               + (f"（登録済みの {n_done} 件は除いています）" if n_done else ""))
    df_rows = pd.DataFrame(rows)
    edited = st.data_editor(
        df_rows.drop(columns="_id"), hide_index=True, width="stretch",
        key="_wave_cands_" + str(abs(hash(tuple(df_rows["_id"]))) % 10**8),
        disabled=["波形", "変数", "工程", "形", "目安", "確からしさ", "説明"],
        column_config={
            "採用": st.column_config.CheckboxColumn(width="small"),
            "波形": st.column_config.LineChartColumn("波形（平均）", width="small"),
            "変数": st.column_config.TextColumn(width="medium"),
            "工程": st.column_config.TextColumn(width="small"),
            "形": st.column_config.TextColumn(width="small", help="山形・台形・谷形・くぼみ（サイクルごとの形の多数決）"),
            "検出点タイプ": st.column_config.SelectboxColumn(options=WAVE_POINT_TYPES, required=True),
            "目安": st.column_config.TextColumn("目安（時刻／値）", width="small",
                                              help="基準データで検出した時刻と値の中央値"),
            "確からしさ": st.column_config.TextColumn(
                width="small", help="全サイクルで検出できた割合と、サイクル間の時刻のばらつきから決めています"),
            "傾向解析に出す": st.column_config.CheckboxColumn("推移に出す", width="small"),
        })
    picked = edited[edited["採用"]]
    if st.button(f"採用した {len(picked)} 件を検出点として登録", type="primary",
                 disabled=picked.empty, key="_wave_adopt"):
        for i, r in picked.iterrows():
            var = df_rows.loc[i, "変数"]
            h = hyp[var]
            c = h["cands"][int(df_rows.loc[i, "_id"].split("|")[1])]
            dtype = r["検出点タイプ"]
            params = c["params"] if dtype == c["dtype"] else h["defaults"].get(dtype, {})
            adopt_point(var, dtype, str(r["名前"] or "").strip(), params, h["trigger"], h["edge"],
                        h["wpost"], bool(r["傾向解析に出す"]), process=h["process"])
        ss["_wave_msgs"] = [f"検出点を {len(picked)} 件登録しました。下の一覧で確認・調整できます。"]
        st.rerun()


_has_points = any(ss.get(f"wvol___global_{v}_t_det_list") or ss.get(f"wvol___global_{v}_xy_det_list")
                  for v in num_cols)
with st.expander("検出点の候補（基準データの事前解析）", expanded=not _has_points):
    _candidate_panel()

# ── ここから旧メイン画面の波形検査タブを移設（ロジックは変更なし）────────────

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
    # ダイアログ遷移フラグ処理（追加→編集）
    if st.session_state.get("wi_transition_to_edit") \
            and "wi_edit_target" in st.session_state:
        st.session_state.pop("wi_transition_to_edit", None)
        _wi_edit_dialog()

    # 登録済みの検出点を工程ごとに整理（変数のサイクル開始信号で工程に割り当て、基準データでの位置を付ける）
    _wi_points = organized_points(ref["df"] if ref else None)
    _wi_sum_items: list = [{
        "変数": it["var"], "トリガー": it["trigger"], "エッジ": it["edge"], "グラフ": it["graph"],
        "No": it["no"], "タイプ": it["type"], "名前": it["name"],
        "有効": "✅" if it["on"] else "−", "傾向解析": "📈" if it["trend"] else "−",
        "_svkey": it["vkey"], "_did": it["did"], "_sv": it["var"], "_graph": it["graph"], "_dtype": it["type"],
        "_pt": it,
    } for it in _wi_points]
    _wi_n_dets  = len(_wi_sum_items)
    _wi_n_trend = sum(1 for r in _wi_sum_items if r["傾向解析"] == "📈")
    _wi_procs = list(dict.fromkeys(r["_pt"]["process"] for r in _wi_sum_items))

    with st.expander(
        f"📋 登録サマリー　検出点: **{_wi_n_dets}** 件（{len([p for p in _wi_procs if p])} 工程）"
        f"　傾向解析送出: **{_wi_n_trend}** 件",
        expanded=True,
    ):
        st.caption("工程ごとに、基準データで検出した時刻の順に並べています。"
                   "位置は工程のステップに対して、その検出点がどこで起きているかです。")
        for _wi_proc in _wi_procs:
            _rows = [r for r in _wi_sum_items if r["_pt"]["process"] == _wi_proc]
            if _wi_proc:
                st.markdown(f"**{_wi_proc}**　<span class='apb-lead'>サイクル開始 {_rows[0]['トリガー']}"
                            f"（{'立ち上がり' if _rows[0]['エッジ'] == 'RISE' else '立ち下がり'}）・{len(_rows)} 点</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("**工程に紐づいていない検出点**", unsafe_allow_html=True)
                st.caption("サイクル開始信号がどの工程とも一致しません。⚙️ で工程のサイクル開始信号を選ぶと、その工程に入ります。")
            # スマホでは列が縦に積まれて読めなくなるので、横並びのまま横スクロールさせる（core/ui.py の CSS）
            with st.container(key=f"wisumgrid_{abs(hash(_wi_proc)) % 10**6}"):
                _w = [2.2, 1.4, 1.4, 2.6, 0.5, 0.5, 0.6]
                for _lbl, _col in zip(["変数", "タイプ", "名前", "位置（基準データ）", "有効", "📈", ""],
                                      st.columns(_w)):
                    _col.caption(_lbl)
                for _wi_row in _rows:
                    _pt = _wi_row["_pt"]
                    _wr = st.columns(_w)
                    _wr[0].markdown(f"**{_wi_row['変数']}**" + ("　<small>XY</small>" if _pt["graph"] == "XY" else ""),
                                    unsafe_allow_html=True)
                    _wr[1].markdown(_wi_row["タイプ"])
                    _wr[2].markdown(_wi_row["名前"] or "−")
                    if _pt["t"] is not None:
                        _pos = f"{_pt['t']:.0f} ms" + (f"　{_pt['where']}" if _pt["where"] else "")
                        if _pt["rate"] < 0.9:
                            _pos += f"　<small>（検出 {_pt['rate'] * 100:.0f}% のサイクル）</small>"
                    else:
                        _pos = "<small>判定・XY・未検出のため位置なし</small>"
                    _wr[3].markdown(_pos, unsafe_allow_html=True)
                    _wr[4].markdown(_wi_row["有効"])
                    _wr[5].markdown(_wi_row["傾向解析"])
                    with _wr[6]:
                        if st.button("⚙️", key=f"_wisum_edit_{_pt['dkey']}", help="波形を見ながら編集"):
                            st.session_state["wi_edit_target"] = {k: v for k, v in _wi_row.items() if k != "_pt"}
                            _wi_edit_dialog()

        # ── 行追加ボタン ────────────────────────────────────────
        _wadd_c1, _wadd_c2 = st.columns([9.6, 0.6])
        with _wadd_c2:
            if st.button("＋", key="_wisum_add_btn", help="検出点を追加"):
                _wi_add_dialog()

        if not _wi_sum_items:
            st.info(
                "検出点はまだ設定されていません。"
                "右の ＋ ボタンまたは下のトリガー設定から変数を選んで検出点を追加してください。"
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
                    _snap: dict = {}
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
    # 常に __global を使用（工程に依存しない独立モード）
    _wi_pname = "__global"
    # サイクル表示用デフォルトトリガー（変数ごとのトリガーは編集ダイアログで設定）
    _wi_trigger = bool_cols[0] if bool_cols else ""
    _wi_edge    = "RISE"

    # ── サイクル検出 ────────────────────────────────────────────
    if not _wi_trigger:
        st.warning("Bool変数がありません。サイクルを検出できません")
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

                # ── 重ねて比べるデータ（ページ上部の選択）──────────────
                _wi_ref_overlay = None
                _wi_cmp_entries: list = [
                    {"key": k, "label": store[k]["label"], "df": store[k]["df"], "result_df": None,
                     "color": _CMP_PALETTE[i % len(_CMP_PALETTE)]}
                    for i, k in enumerate(overlay_keys)
                ]
                if not _wi_cmp_entries and ref_key and data_key != ref_key:
                    _wi_ref_overlay = store[ref_key]["df"]
                    st.caption(f"基準データ（{store[ref_key]['label']}）を橙色で重ねて表示しています")

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
