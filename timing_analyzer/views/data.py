"""
views/data.py — データ（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "データ",
    "基準データ・比較データの読み込みと、設定ファイルの保存・読み込みをまとめて行います。",
    will_have=['基準データ（正常時）と比較データの一覧、ラベル・取得日の編集', 'フォルダからの一括読み込み', '設定ファイル（JSON）の保存と読み込み'],
    legacy_where="サイドバー（① 基準CSV／② 比較CSV／設定ファイル）",
    legacy_kwargs=dict(view_mode='⚙️ 設定', tab='🏭 工程タイミング'),
    eyebrow="準備",
)
