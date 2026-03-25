# Windows EXE ビルド手順

## 前提条件

- Python 3.10 以上がインストールされていること
- インターネット接続（初回のみ依存パッケージのダウンロードが必要）

## ビルド方法

### 方法 A: バッチファイル（推奨）

```
windows_app\build.bat をダブルクリック
```

自動で依存パッケージのインストールとビルドが実行されます。

### 方法 B: 手動

```bat
:: APBPoC ディレクトリで実行
pip install pyinstaller pywebview

pyinstaller windows_app\APBAnalyzer.spec --noconfirm
```

## 出力先

```
dist\
  APBAnalyzer\
    APBAnalyzer.exe   ← これをダブルクリックで起動
    (依存 DLL 群)
```

> **注意**: `APBAnalyzer.exe` 単体では動きません。
> `APBAnalyzer` フォルダごと配布してください。

## 起動の流れ

1. EXE をダブルクリック
2. スプラッシュ画面が表示される（内部でローカルサーバーを起動）
3. 約 5〜15 秒でアプリ画面が開く
4. ウィンドウを閉じると自動終了

## 配布方法

`dist\APBAnalyzer` フォルダを ZIP にまとめて配布するだけです。
配布先の PC に Python は不要です。

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| ビルドが失敗する | `pip install pyinstaller --upgrade` を試す |
| 起動タイムアウトが出る | PC スペックによっては遅いので再度試す |
| ウィルス対策ソフトが反応する | PyInstaller 製 EXE は誤検知が多い。除外設定を追加 |
| 画面が真っ白のまま | pywebview のバックエンドに Edge/WebView2 が必要（Windows 10/11 なら標準搭載） |
