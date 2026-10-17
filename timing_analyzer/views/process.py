"""
views/process.py — 工程とステップ（段階 2・3 で実装予定の仮ページ）
"""
from core.ui import planned_page

planned_page(
    "工程とステップ",
    "サイクル開始信号を選んで工程を登録し、時間を測りたい動作をステップとして定義します。",
    will_have=['工程の登録と、サイクル開始信号・タクト目標の設定', 'ステップ候補の自動検出とチェックによる追加', 'ガントチャートでの確認、手動追加・順序・色の変更（詳細設定）'],
    legacy_where="工程タイミング ＞ 設定",
    legacy_kwargs=dict(view_mode='⚙️ 設定', tab='🏭 工程タイミング'),
    eyebrow="準備",
)
