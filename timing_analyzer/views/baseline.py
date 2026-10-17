"""
views/baseline.py — 基準値（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "基準値",
    "正常時の各ステップのタイミングを基準値として登録します。点検ではこの値との差を見ます。",
    will_have=['基準にするデータの選択と、登録前のプレビュー', 'ステップごとの基準値の確認・修正'],
    legacy_where="工程タイミング ＞ 設定 ＞ 基準値",
    legacy_kwargs=dict(view_mode='⚙️ 設定', tab='🏭 工程タイミング'),
    eyebrow="準備",
)
