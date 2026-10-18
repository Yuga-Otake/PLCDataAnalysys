"""
core/bundle.py — プロジェクトファイル（基準データ＋設定を 1 つにまとめた ZIP）

中身:
  manifest.json   形式・作成日時・入っている CSV の役割（基準／比較）とラベル
  settings.json   設定ファイルと同じもの（工程・ステップ・基準値・波形検査条件）
  data/*.csv      APB の CSV 形式（Date Time 列）で書き出したデータ
  README.txt      中身の説明（ZIP を開いた人向け）
読み込むと、今のデータを置き換えて、データと設定をまとめて復元する。
"""
import io
import json
import os
import zipfile
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import streamlit as st

from core.data import (add_csv, apply_settings_json, detect_bool_columns, set_use_sample, settings_summary,
                       store)
from core.project import build_settings_json

FORMAT = "apb-project"
VERSION = 1
_README = """APB タイミング解析 プロジェクトファイル

このファイルを APB タイミング解析の「データ」画面で読み込むと、基準データ・比較データと、
工程・ステップ・基準値・波形検査条件の設定がまとめて元に戻ります。

  settings.json  設定（「設定を保存」で保存される JSON と同じもの）
  data/          データ（APB の CSV 形式）
  manifest.json  どの CSV が基準データか・比較データか、表示名など

作成日時: {created}
基準データ: {ref}
比較データ: {n_cmp} 件
"""


def bundle_file_name() -> str:
    return f"apb_project_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"


def _csv_bytes(df: pd.DataFrame) -> bytes:
    """APB の CSV 形式（先頭列 Date Time）で書き出す。読み込み側の高速な経路でそのまま読める。

    列が数千ある APB のデータでも速いように、DataFrame はコピーせず pyarrow で書く（値は失われない）。
    """
    ts = pd.to_datetime(df["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    tb = pa.Table.from_pandas(df.drop(columns="Timestamp"), preserve_index=False)
    tb = tb.add_column(0, "Date Time", pa.array(ts.to_numpy(dtype=object), type=pa.string()))
    sink = io.BytesIO()
    pacsv.write_csv(tb, sink)
    return sink.getvalue()


def _safe_name(key: str, used: set) -> str:
    name = os.path.basename(str(key)) or "data"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    base, n = name[:-4], 1
    while name in used:
        n += 1
        name = f"{base}_{n}.csv"
    used.add(name)
    return name


def bundle_maker(include_compare: bool):
    """ダウンロードボタンに渡す関数を返す（押されたときに別スレッドで ZIP を作る）。

    別スレッドでは session_state を読めないので、必要なもの（設定と DataFrame）はここで取り出しておく。
    """
    settings = build_settings_json()
    entries = [(k, dict(e)) for k, e in store().items() if e.get("is_ref") or include_compare]
    entries.sort(key=lambda kv: not kv[1].get("is_ref"))      # 基準データを先頭に

    def make() -> bytes:
        buf = io.BytesIO()
        used, files = set(), []
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for key, e in entries:
                name = _safe_name(key, used)
                z.writestr(f"data/{name}", _csv_bytes(e["df"]))
                files.append({"path": f"data/{name}", "key": key, "label": e.get("label", name),
                              "role": "reference" if e.get("is_ref") else "comparison", "rows": len(e["df"])})
            created = datetime.now().strftime("%Y-%m-%d %H:%M")
            z.writestr("settings.json", settings)
            z.writestr("manifest.json", json.dumps(
                {"format": FORMAT, "version": VERSION, "created_at": created, "files": files,
                 "settings": "settings.json"}, ensure_ascii=False, indent=2))
            ref = next((f["label"] for f in files if f["role"] == "reference"), "なし")
            z.writestr("README.txt", _README.format(
                created=created, ref=ref, n_cmp=sum(f["role"] == "comparison" for f in files)))
        return buf.getvalue()

    return make


def is_bundle(name: str) -> bool:
    return str(name).lower().endswith(".zip")


def load_bundle(file) -> str:
    """プロジェクトファイルを読み込み、今のデータと設定を置き換える。戻り値は結果のメッセージ"""
    from analyzer import load_csv

    with zipfile.ZipFile(file) as z:
        names = set(z.namelist())
        if "manifest.json" not in names:
            raise ValueError("プロジェクトファイルではありません（manifest.json がありません）")
        manifest = json.loads(z.read("manifest.json").decode("utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError("プロジェクトファイルの形式が違います")
        if int(manifest.get("version", 0)) > VERSION:
            raise ValueError("新しい版のツールで作られたファイルです。ツールを更新してください")
        loaded = []
        for f in manifest.get("files", []):
            if f.get("path") not in names:
                raise ValueError(f"{f.get('path')} が見つかりません")
            df = load_csv(io.BytesIO(z.read(f["path"])))
            loaded.append((f, df))
        settings = json.loads(z.read(manifest.get("settings", "settings.json")).decode("utf-8")) \
            if manifest.get("settings", "settings.json") in names else None

    # ここまでで読めることを確かめてから、今のデータを置き換える
    ss = st.session_state
    set_use_sample(False)
    store().clear()
    ss["ref_csv_key"] = None
    ss.pop("active_csv", None)
    for f, df in loaded:
        add_csv(f.get("key") or f["path"], df, detect_bool_columns(df), f.get("label", f["path"]),
                f.get("role") == "reference", "project")
    n_ref = sum(f.get("role") == "reference" for f, _ in loaded)
    msg = f"プロジェクトファイルを読み込みました（基準データ {n_ref}・比較データ {len(loaded) - n_ref}"
    if settings:
        apply_settings_json(settings)
        msg += "・" + settings_summary(settings)
    msg += "）"
    ss["_project_saved"] = True
    return msg
