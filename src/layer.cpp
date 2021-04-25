#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "../include/layer.h"
#include "../include/activation.h"
#include "../include/loss.h"

namespace MyDL
{

    using namespace Eigen;
    using std::cout;
    using std::endl;
    using std::vector;

    // -------------------------------------------------
    //          AddLayer
    // -------------------------------------------------
    vector<MatrixXd> AddLayer::forward(vector<MatrixXd> inputs)
    {

        vector<MatrixXd> outs;

        // 本当はここに入力のバリデーションを入れる(要素数が2)

        outs.push_back(inputs[0] + inputs[1]);
        return outs;
    }

    vector<MatrixXd> AddLayer::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dx, dy;
        dx = douts[0];
        dy = douts[0];

        grads.push_back(dx);
        grads.push_back(dy);
        return grads;
    }

    // -------------------------------------------------
    //          MulLayer
    // -------------------------------------------------
    vector<MatrixXd> MulLayer::forward(vector<MatrixXd> inputs)
    {
        _x = inputs[0];
        _y = inputs[1];

        vector<MatrixXd> outs;
        MatrixXd out;

        out = _x.array() * _y.array();

        outs.push_back(out);
        return outs;
    }

    vector<MatrixXd> MulLayer::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dx, dy;

        dx = douts[0].array() * _y.array();
        dy = douts[0].array() * _x.array();

        grads.push_back(dx);
        grads.push_back(dy);
        return grads;
    }

    // -------------------------------------------------
    //          ReLU
    // -------------------------------------------------
    vector<MatrixXd> ReLU::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X;
        vector<MatrixXd> outs;
        X = inputs[0];

        _mask = X.unaryExpr([](double p) { return p > 0; }).cast<double>();
        outs.push_back(X.array() * _mask.array()); // 負の数に対して、maskした部分は-0になるのが気がかり
        return outs;
    }

    vector<MatrixXd> ReLU::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dx;

        dx = douts[0].array() * _mask.array();

        grads.push_back(dx);
        return grads;
    }

    // -------------------------------------------------
    //          Sigmoid
    // -------------------------------------------------
    vector<MatrixXd> Sigmoid::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X;
        vector<MatrixXd> outs;
        X = inputs[0];

        _y = X.unaryExpr([](double p) { return 1 / (1 + exp(-p)); });

        outs.push_back(_y);
        return outs;
    }

    vector<MatrixXd> Sigmoid::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dx;

        dx = _y.array() * (MatrixXd::Ones(_y.rows(), _y.cols()) - _y).array();

        grads.push_back(dx);
        return grads;
    }

    // -------------------------------------------------
    //          Affine
    // -------------------------------------------------
    Affine::Affine(MatrixXd& W, MatrixXd& b)
    {
        _W = W;
        _b = b;
    }

    Affine::Affine(const shared_ptr<MatrixXd> W, const shared_ptr<MatrixXd> b)
    {
        pW = W;
        pb = b;
    }

    Affine::Affine(const int input_size, const int output_size, const double weight_init_std)
    {
        auto W = std::make_shared<MatrixXd>(input_size, output_size);
        auto b = std::make_shared<MatrixXd>(1, output_size);
        *W = weight_init_std * MatrixXd::Random(input_size, output_size);
        *b = MatrixXd::Zero(1, output_size);
        pW = W;
        pb = b;
    }

    vector<MatrixXd> Affine::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X, Y, W;
        VectorXd b;
        vector<MatrixXd> outs;
        X = inputs[0];
        _X = X;
        W = *pW;
        b = pb->row(0);
        // W = _W;
        // b = _b.row(0); // 非推奨 → コピーが走るだけ

        Y = (X * W).rowwise() + b.transpose();

        outs.push_back(Y);
        return outs;
    }

    vector<MatrixXd> Affine::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dX, dout;
        MatrixXd W;

        dout = douts[0];
        W = *pW;
        // W = _W; // 非推奨 → コピーが走るだけ

        dX = dout * W.transpose();
        dW = _X.transpose() * dout;
        db = dout.colwise().sum();

        grads.push_back(dX);
        return grads;
    }

    // -------------------------------------------------
    //          SoftmaxWithLoss
    // -------------------------------------------------
    vector<MatrixXd> SoftmaxWithLoss::forward(vector<MatrixXd> inputs)
    {
        vector<MatrixXd> outs;
        MatrixXd out = MatrixXd::Zero(1,1);
        MatrixXd X = inputs[0];
        _Y = softmax(X);
        _t = inputs[1];

        _loss = cross_entropy_error(_Y, _t);

        out << _loss;
        outs.push_back(out);
        return outs;
    }

    vector<MatrixXd> SoftmaxWithLoss::backward(vector<MatrixXd> dout)
    {
        vector<MatrixXd> grads;

        // 出力のバリデーション
        if (!(dout[0].rows() == 1 && dout[0].cols() == 1))
        {
            cout << "dout shape is not valid @ SoftmaxWithLoss Layer!!" << endl;
            cout << "Please Check Your DNN Architecture." << endl;
            return grads;
        }

        double batch_size = _t.rows();
        MatrixXd dx;

        dx = (_Y - _t) / batch_size;

        grads.push_back(dx);
        return grads;
    }

    // -------------------------------------------------
    //          BatchNormalization
    // -------------------------------------------------
    BatchNorm::BatchNorm(const shared_ptr<MatrixXd> gamma, const shared_ptr<MatrixXd> beta, double momentum)
    {
        int cols;

        pgamma = gamma;
        pbeta = beta;
        _momentum = momentum;

        cols = pgamma->cols();

        // pgamma, pbetaが横ベクトルであることをここでバリデーションしておく？(rows()==1を確認とか。)

        // ここで初期化しておく。ミニバッチにおける平均なので、横ベクトルになる。
        _avg_mean = VectorXd::Zero(cols);
        _avg_var  = VectorXd::Zero(cols);
    }
    // もしかすると、I/Fとしてデータの次元だけ与えるようにして、コンストラクタでパラメータの実体を作成し、shared_ptrを格納する方がきれいに作れる？
    // affineも同様。同じようにすれば、ネットワークの構築作業がかなり楽になる気がする。

    BatchNorm::BatchNorm(const int input_size, const double weight_init_std, const double momentum)
    {
        auto gamma = std::make_shared<MatrixXd>(1, input_size);
        auto beta  = std::make_shared<MatrixXd>(1, input_size);
        *gamma = weight_init_std * MatrixXd::Random(1, input_size); // Onesで初期化する方が良い？
        *beta  = weight_init_std * MatrixXd::Random(1, input_size); // Zerosで初期化する方が良い？

        pgamma = gamma;
        pbeta  = beta;
        _momentum = momentum;

        _avg_mean = VectorXd::Zero(input_size);
        _avg_var  = VectorXd::Zero(input_size);
    }


    vector<MatrixXd> BatchNorm::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X = inputs[0];
        MatrixXd Xn, Xc, out;
        VectorXd gamma, beta;
        VectorXd mu, var, std;
        vector<MatrixXd> outs;
        double momentum = _momentum;

        gamma = pgamma->row(0);
        beta = pbeta->row(0);

        bool train_flg = Config::getInstance().get_flag(); // SingletonのConfigクラスからモード取得

        if (train_flg){
            mu = X.colwise().mean(); // ミニバッチにおけるデータの平均値(各次元ごと)
            Xc = X.rowwise() - mu.transpose();   // 入力データの各次元から平均値を引き、中心化
            var = Xc.array().pow(2).colwise().mean();                  // ミニバッチにおける標本分散(各次元ごと)
            std = var.unaryExpr([](double p){return sqrt(p + 1e-7);}); // ミニバッチにおける標準偏差(各次元ごと)

            Xn = Xc.array().rowwise() / std.transpose().array();

            _batch_size = X.rows();
            _Xc = Xc;
            _Xn = Xn;
            _std = std;
            _avg_mean = (momentum * _avg_mean.array() + (1 - momentum) * mu.array()).matrix();
            _avg_var  = (momentum * _avg_var.array()  + (1 - momentum) * var.array()).matrix();
        }
        else
        {
            Xc = X.rowwise() - _avg_mean.transpose(); // broadcast演算
            std = _avg_var.unaryExpr([](double p) { return sqrt(p + 1e-7); });
            Xn = Xc.array().rowwise() / std.transpose().array();
        }

        // gammaとbetaはVectorXdに変換するため、row(0)を使用している。
        out = (Xn * gamma.asDiagonal()).rowwise() + beta.transpose();
        outs.push_back(out);

        return outs;
    }



    vector<MatrixXd> BatchNorm::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dout  = douts[0];
        MatrixXd dXn, dXc;
        MatrixXd Xn = _Xn;
        MatrixXd Xc = _Xc;
        VectorXd std = _std;
        VectorXd var;
        MatrixXd dX;
        VectorXd gamma, beta, dmu, dstd, dvar;

        gamma = pgamma->row(0); // ブロードキャスト演算用にVectorXd化
        beta  = pbeta->row(0);  // 同上

        // parameter's gradient
        dbeta = dout.colwise().sum();
        dgamma = (dout.array() * Xn.array()).colwise().sum();

        // For inputs' gradient
        dXn  = dout * gamma.asDiagonal();                        // ブロードキャスト演算
        dXc  = dXn.array().rowwise() / _std.transpose().array(); // ブロードキャスト演算

        // colwise()を忘れないように -> shapeが合わない
        dstd = ((-1.0 * dXn).array() * Xc.array()).colwise().sum().array().rowwise()
                / (std.transpose().array().pow(2)); // ブロードキャスト演算


        // // デバッグ用(上記dstdのshapeが合わない原因調査。colwise().sum()が抜けていたせいでshapeが合わなかった)
        // MatrixXd tmp_dstd;
        // tmp_dstd = ((-1.0) * dXn).array() * Xc.array(); // ブロードキャスト演算
        // var  = std.array().pow(2);
        // dstd = tmp_dstd.colwise().sum().array().rowwise() / var.transpose().array());


        dvar = (0.5 * dstd).array() * std.array().inverse(); // Vector同士の要素積はarray()を使用。要素商はarray().inverse()とする。
        dXc += (((2.0 / _batch_size)*Xc).array().rowwise() * dvar.transpose().array()).matrix(); // 左辺と右辺でmatrix型かarray型かを合わせる
        dmu  = dXc.colwise().sum();
        dX   = dXc.rowwise() - ((1/_batch_size) * dmu.transpose());

        grads.push_back(dX);

        return grads;
    }

    // -------------------------------------------------
    //          Dropout
    // -------------------------------------------------
    Dropout::Dropout(const int row, const int col, const double dropout_ratio)
    {
        _dropout_ratio = dropout_ratio;
        _mask = MatrixXd::Zero(row, col).cast<bool>();
    }

    vector<MatrixXd> Dropout::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X = inputs[0];
        MatrixXd out;
        vector<MatrixXd> outs;
        bool train_flg = Config::getInstance().get_flag();
        int col, row;
        col = _mask.cols(); // サイズの取得
        row = _mask.rows(); // サイズの取得

        // Eigen MatrixXd::Random は -1 ~ 1 の範囲で乱数生成するので、これを0~1の範囲に変更する
        double HI = 1.0;
        double LO = 0;
        double range = HI - LO;

        if (train_flg)
        {
            // 乱数の範囲調整 → 自作関数にし、utilとして使用？
            MatrixXd rand = MatrixXd::Random(row, col);
            rand = (rand + MatrixXd::Constant(row, col, 1.)*range/2.);
            rand = (rand + MatrixXd::Constant(row, col, LO));
            _mask = rand.array() < _dropout_ratio;
            MatrixXd mask = _mask.cast<double>();
            out = X.array() * mask.array();
        } else {
            out = X * (1.0 - _dropout_ratio);
        }

        outs.push_back(out);
        return outs;
    }

    vector<MatrixXd> Dropout::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        MatrixXd dout = douts[0];
        MatrixXd grad;

        grad = dout.array() * _mask.cast<double>().array();

        grads.push_back(grad);
        return grads;
    }


}