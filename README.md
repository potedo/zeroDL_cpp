# zeroDL_cpp

## 概要
O'REILLYから出版されている「ゼロから作るDeep Learning」を、勉強用にC++で実装しなおしたものです。

ライセンスはMITライセンスです。
なお、ソースコードの利用については自己責任でお願いします。

## 使用ライブラリ

- Eigen: 行列演算ライブラリ
- matplotlib-cpp: 描画用ライブラリ(Pythonのmatplotlibの一部機能を使用可能)
- picojson: jsonを読み込むライブラリ

【参考】  </br>
https://eigen.tuxfamily.org/index.php?title=Main_Page  </br>
https://github.com/lava/matplotlib-cpp  </br>
https://github.com/kazuho/picojson  </br>

## 動作環境
Windows10 WSL(Ubuntu 18.04LTS)

上記以外の環境では動作未確認のため、自己責任でお願いします。

## コンパイラ
g++ version 7.5.0

## 使い方

### 初期設定
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

また、以下「MNIST Datasetの配置」に記載した作業を実施すること。

### MNIST Datasetの配置

http://yann.lecun.com/exdb/mnist/

上記サイトから
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz

をダウンロードし、"datasets/data/" 以下に解凍して配置する。


## プロジェクト構成

- ch〇〇：ゼロから作るDeep Learningの各章に対応した内容を実装したフォルダです。実装の際にライブラリの使い方を確認したコードも含まれます。
- datasets: MNISTのデータおよびそのローダを配置するディレクトリ
- include, src: 6章以降で使用するモジュールをライブラリ化したもの
- simple_lib: 5章までで使用するモジュールをライブラリ化したもの
- test: 動作確認用のコードを格納するディレクトリ。テスト自体は単なる動作確認をしたもの(境界値分析などは未実施)

## ライブラリ "zeroDLcpp" 実装内容

大きく分けて、以下の内容を実装しています。

- Layer
- Model
- Dataset
- Optimizer
- Trainer

Layerを組み合わせてModelを作成し、TrainerにModel, Optimizer, Datasetを設定することで学習できるようにしています。
必要に応じて、ベースとなる抽象基底クラスを継承することで機能を拡張できる設計にしてあります。

### Layer

BaseLayerという抽象基底クラスを継承する形で実装。
順伝播はforward, 逆伝播はbackwardというメソッドで実装。
また、引数も返り値も `vector<MatrixXd>` で実装している(複数の入力値を持つLayerに対するI/Fの共通化)

#### 実装クラス

- AddLayer: 2つの入力(Matrix)を加算するクラス
- MulLayer: 2つの入力(Matrix)の要素積(アダマール積)を計算するクラス
- ReLU: ReLUの計算を行うクラス
- Sigmoid: シグモイド関数の計算を行うクラス
- Affine: 全結合層の計算を行うクラス(KerasでいうところのDenseに相当)
- SoftmaxWithLoss: ソフトマックス関数の計算およびクロスエントロピーの計算結果を出力するクラス
- BatchNormalization: バッチ正則化を行うクラス
- Dropout: ドロップアウトの計算を行うクラス
- Conv2D: 畳み込み演算を行うクラス
- Pooling: max poolingを行うクラス

### Activation

以下の計算を実装

- step_function
- sigmoid
- relu
- identity_function

いずれもテンプレートを用いて実装。

### Dataset

Datasetを抽象基底クラスとして実装(Trainerで具体的な実装を差し替えできるようにするための処置)
"datasets/"以下に実装した"mnist"は、このDatasetを継承して作成。

### Loss

cross_entropy_errorのみ実装

### Optimizer

Optimizerという抽象基底クラスを継承する形で実装。
updateというメソッドで1step分の更新をする機能を持つ。
引数は、「更新対象のパラメータを格納したunordered_map」「Layerのbackwardで計算した勾配の情報を格納したunordered_map」として実装している。

#### 実装クラス

- SGD
- Momentum
- AdaGrad
- RMSprop
- Adam

### Model

BaseModelという抽象基底クラスを継承する形で実装。
predict, loss, accuracy, gradient, get_paramsというメソッドが最低限必要となる。
- predict: 推論を実行する関数
- loss: 入力とラベルから損失関数の値を計算する関数
- accuracy: 入力とラベルから正答率を計算する関数
- gradient: 入力とラベルから、勾配を計算する関数
- get_params: Model内に格納している更新可能パラメータを格納したunordered_mapを返す関数

#### 実装クラス

- TwoLayerMLP: 2層の多層パーセプトロン
- MultiLayerModel: 層の数・隠れユニットの数を自由に変更できる多層パーセプトロン。バッチ正則化、ドロップアウト、L2正則化にも対応している。
- SimpleConvModel: 簡単なCNNのモデル。ゼロから作るDeep Learning p.230のモデルを参照

### Trainer

訓練用のコードを容易に実装するためのクラス。
コンストラクタで、以下のクラスのポインタ(shared_ptr)を引数に与える必要がある。

- Model
- Optimizer
- Dataset

上から順に、訓練対象のモデル、最適化アルゴリズムの指定、使用するデータセットである。

### utils

その他実装上便利な関数を実装。今回は次の1つだけを実装

- load_json: jsonの内容を読み込む関数
