# zeroDL_cpp

## 概要
O'REILLYから出版されている「ゼロから作るDeep Learning」を、勉強用にC++で実装しなおしたものです。

ライセンスはMITライセンスです。
なお、ソースコードの利用については自己責任でお願いします。

## 使用ライブラリ

- Eigen: 行列演算ライブラリ
- matplotlib-cpp: 描画用ライブラリ(Pythonのmatplotlibの一部機能を使用可能)
- picojson: jsonを読み込むライブラリ

【参考】
https://eigen.tuxfamily.org/index.php?title=Main_Page
https://github.com/lava/matplotlib-cpp
https://github.com/kazuho/picojson

## 動作環境
Windows10 WSL(Ubuntu 18.04LTS)

## コンパイラ
g++ version 7.5.0

## 使い方

VSCode拡張機能"CMake Tools"をインストールしていることを前提とします。
以下の手順で設定します。

1. このリポジトリをクローン
2. ディレクトリ最上位に存在するCMakeLists.txtを別のフォルダに退避し、このディレクトリから削除(後ほど使用)
3. VSCode拡張機能"Remote WSL"で、WSL環境のUbuntuに接続
4. 「shift+ctrl+P」でコマンドパレットを開き、「CMake: Quick Start」を検索し、選択、Enter
5. 適当なプロジェクト名を入力し、Enter
6. "Executable"を選択肢、Enter
7. CMakeLists.txtが生成され、"build"ディレクトリが作成される。
8. 生成されたCMakeLists.txtの内容を2.の手順で退避したCMakeLists.txtの内容に書き換える
9. 問題なければ、「shift + F7」でビルドが開始される。build targetを尋ねられた場合は、allを入力すればOK


