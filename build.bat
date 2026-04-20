@echo off
chcp 65001 > nul
echo ============================================
echo  APB Timing Analyzer - EXE ビルドスクリプト
echo ============================================
echo.

:: PyInstaller 確認・インストール
where pyinstaller >nul 2>&1
if %errorlevel% neq 0 (
    echo [情報] PyInstaller をインストール中...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo [エラー] PyInstaller のインストールに失敗しました。
        pause
        exit /b 1
    )
)

:: 古いビルド成果物を削除
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist APBTimingAnalyzer.spec del APBTimingAnalyzer.spec

echo [情報] ビルド開始... （初回は 5〜15 分かかります）
echo.

pyinstaller ^
  --onedir ^
  --name APBTimingAnalyzer ^
  --collect-all streamlit ^
  --collect-all plotly ^
  --collect-all altair ^
  --collect-all pandas ^
  --hidden-import numpy ^
  --hidden-import scipy ^
  --hidden-import pyarrow ^
  --hidden-import openpyxl ^
  --hidden-import streamlit.runtime.scriptrunner.magic_funcs ^
  --hidden-import jsonschema.validators ^
  --add-data "timing_analyzer;timing_analyzer" ^
  launcher.py

if %errorlevel% neq 0 (
    echo.
    echo [エラー] ビルドに失敗しました。上のエラーメッセージを確認してください。
    pause
    exit /b 1
)

echo.
echo ============================================
echo  ビルド完了！
echo  出力先: dist\APBTimingAnalyzer\
echo  実行ファイル: dist\APBTimingAnalyzer\APBTimingAnalyzer.exe
echo ============================================
echo.
echo 配布するときは dist\APBTimingAnalyzer\ フォルダごと ZIP して渡してください。
pause
