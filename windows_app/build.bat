@echo off
chcp 65001 >nul
setlocal

echo =========================================
echo  APB タイミング解析 - EXE ビルド
echo =========================================
echo.

:: カレントを APBPoC ルートに移動
cd /d "%~dp0.."

:: 依存パッケージのインストール確認
echo [1/3] 依存パッケージを確認しています...
pip install -q pyinstaller pywebview streamlit pandas numpy plotly
if errorlevel 1 (
    echo ERROR: pip install に失敗しました
    pause & exit /b 1
)

:: 前回のビルド成果物を削除
echo [2/3] 前回のビルドをクリーンアップ...
if exist dist\APBAnalyzer  rmdir /s /q dist\APBAnalyzer
if exist build\APBAnalyzer rmdir /s /q build\APBAnalyzer

:: PyInstaller 実行
echo [3/3] EXE をビルドしています（数分かかります）...
pyinstaller windows_app\APBAnalyzer.spec --noconfirm
if errorlevel 1 (
    echo.
    echo ERROR: ビルドに失敗しました。上のエラーメッセージを確認してください。
    pause & exit /b 1
)

echo.
echo =========================================
echo  ビルド完了！
echo  dist\APBAnalyzer\APBAnalyzer.exe
echo =========================================
echo.
echo フォルダを開きますか？ (Y/N)
set /p ans=
if /i "%ans%"=="Y" explorer dist\APBAnalyzer

pause
