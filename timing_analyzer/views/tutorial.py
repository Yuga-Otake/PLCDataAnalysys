"""
views/tutorial.py — チュートリアル（練習データのダウンロードと、案内の開始・再開・終了）

案内そのもの（暗転・スポットライト・カーソル・吹き出し）は core/tutorial.py が全画面で出す。
"""
import streamlit as st

from core import tutorial as tut
from core.tutorial_data import CMP_FILES, REF_FILE, tutorial_zip
from core.ui import page_header

ss = st.session_state
page_header(
    "チュートリアル",
    "練習用のデータを使って、準備（基準を作る）から点検（変化を見つける）までを、実際の画面で操作しながら覚えます。"
    "画面が暗くなり、操作する場所が照らされます。所要時間は 15 分ほどです。",
    eyebrow="ツール",
)

with st.container(border=True):
    st.markdown("**練習データ**")
    st.markdown(f"2 工程（供給工程・圧入工程）の設備を 1 ms 周期で記録した CSV です。"
                f"基準データ（{REF_FILE}）と、時期の違う比較データ {len(CMP_FILES)} 本"
                f"（{'、'.join(f[3:-4] for f in CMP_FILES)}）が入っています。"
                "時期が進むにつれて、ある変化が少しずつ起きています。チュートリアルの中で見つけてください。")
    st.download_button("練習データをダウンロード", data=tutorial_zip(), file_name="APB_チュートリアル.zip",
                       mime="application/zip", icon=":material/download:", key="tut_dl",
                       on_click=lambda: ss.update(_tut_downloaded=True))

st.markdown("**進める内容**")
st.markdown("\n".join(f"{n}. {ch}" for n, ch in enumerate(tut.CHAPTERS, start=1)))

i, n = tut.progress()
if tut.active():
    st.info(f"チュートリアルを進めています（ステップ {i + 1}/{n}）。画面の吹き出しの案内に従ってください。")
    st.button("チュートリアルを終了する", key="tut_page_stop", on_click=tut.stop)
else:
    has_work = bool(ss.get("processes")) or any(e.get("source") != "sample" and k not in
                                                 ("sample_baseline_3cyc.csv", "sample_playback.csv")
                                                 for k, e in ss.get("csv_store", {}).items())
    if 0 < i < n - 1:
        c1, c2 = st.columns([1, 1])
        if c1.button("続きから再開", type="primary", key="tut_resume", width="stretch"):
            ss["tut_on"] = True
            st.rerun()
        restart = c2.button("最初からやり直す", key="tut_restart", width="stretch")
    else:
        restart = st.button("チュートリアルを始める", type="primary", key="tut_start", icon=":material/school:")
    if has_work:
        st.caption("始めると、今読み込んでいるデータと工程・基準値などの設定は消えます。"
                   "必要なら先に「データ」画面で保存してください。")
    if restart:
        tut.start()
        st.rerun()
