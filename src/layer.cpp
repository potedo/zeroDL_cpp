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

        dx = douts[0].array() * (MatrixXd::Ones(_y.rows(), _y.cols()) - _y).array();

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


    BatchNorm::BatchNorm(const int input_size, const double momentum)
    {
        auto gamma = std::make_shared<MatrixXd>(1, input_size);
        auto beta = std::make_shared<MatrixXd>(1, input_size);
        *gamma = MatrixXd::Ones(1, input_size); // Onesで初期化する方が良い？
        *beta = MatrixXd::Zero(1, input_size);  // Zerosで初期化する方が良い？

        pgamma = gamma;
        pbeta = beta;
        _momentum = momentum;

        _avg_mean = VectorXd::Zero(input_size);
        _avg_var = VectorXd::Zero(input_size);
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
    Dropout::Dropout(const double dropout_ratio)
    {
        _dropout_ratio = dropout_ratio;
    }

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
        col = X.cols(); // サイズの取得
        row = X.rows(); // サイズの取得
        _mask = MatrixXd::Zero(row, col).cast<bool>();

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

    // -------------------------------------------------
    //          Convolution
    // -------------------------------------------------

    Conv2D::Conv2D(int C, int H, int W, int Fh, int Fw, int Fn, int stride, int pad, double weight_init_std)
    :_C(C), _H(H), _W(W), _Fh(Fh), _Fw(Fw), _Fn(Fn), _stride(stride), _pad(pad)
    {
        pW = std::make_shared<MatrixXd>(C*Fh*Fw, Fn);
        pb = std::make_shared<MatrixXd>(1, Fn);
        *pW = weight_init_std * MatrixXd::Random(C*Fh*Fw, Fn);
        *pb = MatrixXd::Zero(1, Fn);
    }

    vector<MatrixXd> Conv2D::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X, Y, W;
        VectorXd b;
        vector<MatrixXd> outs;
        X = inputs[0];
        _N = X.rows(); // batch size
        
        W = *pW;
        b = pb->row(0);

        int Oh = 1 + (_H + 2 * _pad - _Fh) / _stride;
        int Ow = 1 + (_W + 2 * _pad - _Fw) / _stride;

        im2col(X, _col);

        Y = (_col * W).rowwise() + b.transpose();
        Map<MatrixXd> Y_reshaped(Y.data(), _N, _Fn*Oh*Ow);

        outs.push_back(Y_reshaped);
        return outs;
    }

    vector<MatrixXd> Conv2D::backward(vector<MatrixXd> douts)
    {
        vector<MatrixXd> grads;
        // doutとして入ってくるのは、(N, Fn*Oh*Ow) という形状が前提

        MatrixXd dout = douts[0];
        MatrixXd W, dcol, dX;
        VectorXd b;
        W = *pW;


        int Oh = (2*_pad+_H-_Fh) / _stride + 1;
        int Ow = (2*_pad+_W-_Fw) / _stride + 1;

        Map<MatrixXd> reshaped_dout(dout.data(), _N*Oh*Ow, _Fn);

        db = reshaped_dout.colwise().sum();
        dW = _col.transpose() * reshaped_dout;

        dcol = reshaped_dout * W.transpose(); // (N×Oh×Ow) × (C×Fh×Fw)

        col2im(dcol, dX);

        grads.push_back(dX);

        return grads;
    }

    void Conv2D::padding(MatrixXd& img, MatrixXd& pad_img)
    {
        pad_img = MatrixXd::Zero(_N, _C * (_H + 2 * _pad) * (_W + 2 * _pad));

        int pad_H_elems = _H + 2 * _pad;
        int pad_W_elems = _W + 2 * _pad;
        int pad_C_elems = pad_H_elems * pad_W_elems;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < _H; h++)
                {
                    for (int w = 0; w < _W; w++)
                    {
                        // 1pixelずつ置き換え
                        pad_img(n, c * pad_C_elems + (h + _pad) * pad_W_elems + (w + _pad)) = img(n, c * (_H * _W) + _W * h + w);
                    }
                }
            }
        }
    }

    void Conv2D::im2col(MatrixXd& img, MatrixXd& col)
    {
        MatrixXd pad_img;

        padding(img, pad_img);

        int Oh = (2 * _pad + _H - _Fh) / _stride + 1;
        int Ow = (2 * _pad + _W - _Fw) / _stride + 1;

        col = MatrixXd::Zero(_N * Oh * Ow, _Fh * _Fw * _C);

        int h_start = 0;
        int w_start = 0;

        int pad_H = _H + 2 * _pad;
        int pad_W = _W + 2 * _pad;

        int tmp_col_row, tmp_col_width;
        int tmp_img_start; 

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h_start = 0; h_start < Oh; h_start++)
                {
                    for (int w_start = 0; w_start < Ow; w_start++)
                    {
                        for (int h_offset = 0; h_offset < _Fh; h_offset++)
                        {
                            tmp_col_row = n * Oh * Ow + h_start * Ow + w_start;
                            tmp_col_width = c * _Fh * _Fw + h_offset * _Fw;
                            tmp_img_start = c * pad_H * pad_W + (h_start * _stride + h_offset) * pad_W + w_start * _stride;
                            col.block(tmp_col_row, tmp_col_width, 1, _Fw) = pad_img.block(n, tmp_img_start, 1, _Fw);
                        }
                    }
                }
            }
        }
    }

    void Conv2D::col2im(MatrixXd& col, MatrixXd& img)
    {
        int pad_H = _H + 2 * _pad;
        int pad_W = _W + 2 * _pad;

        MatrixXd pad_img = MatrixXd::Zero(_N, _C*pad_H*pad_W);

        int Oh = (2 * _pad + _H - _Fh) / _stride + 1;
        int Ow = (2 * _pad + _W - _Fw) / _stride + 1;

        int tmp_img_start;
        int tmp_col_row, tmp_col_width;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < Oh; h++)
                {
                    for (int w = 0; w < Ow; w++)
                    {
                        for (int h_offset = 0; h_offset < _Fh; h_offset++)
                        {
                            tmp_img_start = pad_H*pad_W*c+(h*_stride+h_offset)*pad_W+w*_stride;
                            tmp_col_row = Oh*Ow*n+Ow*h+w;
                            tmp_col_width = c*_Fh*_Fw+h_offset*_Fw;
                            pad_img.block(n, tmp_img_start, 1, _Fw) += col.block(tmp_col_row, tmp_col_width, 1, _Fw);
                        }
                    }
                }
            }
        }

        suppress(pad_img, img);

    }

    void Conv2D::suppress(MatrixXd& pad_img, MatrixXd& img)
    {
        img = MatrixXd::Zero(_N, _C*_H*_W);
        int pad_H = 2*_pad + _H;
        int pad_W = 2*_pad + _W;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < _H; h++)
                {
                    for (int w = 0; w < _W; w++)
                    {
                        // 1pixelずつ置き換え
                        img(n, c*_H*_W+h*_W+w) = pad_img(n, c*pad_H*pad_W+(h+_pad)*pad_W+(_pad+w));
                    }
                }
            }
        }
    }

    // -------------------------------------------------
    //          Pooling
    // -------------------------------------------------

    Pooling::Pooling(int c, int h, int w, int Ph, int Pw, int stride, int pad)
    :_C(c), _H(h), _W(w), _Ph(Ph), _Pw(Pw), _stride(stride), _pad(pad)
    {
    }

    vector<MatrixXd> Pooling::forward(vector<MatrixXd> inputs)
    {
        MatrixXd X, col, vec_out;
        vector<MatrixXd> outs;

        X = inputs[0];
        _X = X;
        _N = X.rows();

        int Oh = (2*_pad + _H - _Ph) / _stride + 1;
        int Ow = (2*_pad + _W - _Pw) / _stride + 1;

        im2col(X, col); // colは N*Oh*Ow × C*Ph*Pw のshapeをもつ

        MatrixXd reshaped_col = MatrixXd::Zero(_C*_N*Oh*Ow, _Ph*_Pw);
        MatrixXd tmp_c_col = MatrixXd::Zero(_N*Oh*Ow, _Ph*_Pw);

        // column majorなので、直接reshapeすると意図した挙動にならない
        // 各チャネルごとにreshapeをかけなおす
        for (int c = 0; c < _C; c++)
        {
            tmp_c_col = col.block(0, c*_Ph*_Pw, _N*Oh*Ow, _Ph*_Pw);
            reshaped_col.block(c*_Ph*_Pw, 0, _N*Oh*Ow, _Ph*_Pw) = tmp_c_col;
        }

        int num_rows = reshaped_col.rows();
        _argmax = MatrixXi::Zero(num_rows, 1);
        vec_out = MatrixXd::Zero(num_rows, 1);
        MatrixXi::Index dummy_row = 0;
        MatrixXi::Index max_col = 0;

        for (int r = 0; r < num_rows; r++)
        {
            vec_out(r, 0) = reshaped_col.row(r).maxCoeff(&dummy_row, &max_col);
            _argmax(r, 0) = max_col;
        }
    
        Map<MatrixXd> out(vec_out.data(), _N, Oh*Ow*_C);

        outs.push_back(out);

        return outs;
    }

    vector<MatrixXd> Pooling::backward(vector<MatrixXd> douts)
    {
        MatrixXd dout; // size: N × (C×Oh×Ow)
        MatrixXd dX;
        vector<MatrixXd> grads;

        int Oh = (2 * _pad + _H - _Ph) / _stride + 1;
        int Ow = (2 * _pad + _W - _Pw) / _stride + 1;

        dout = douts[0];
        Map<MatrixXd> vec_dout(dout.data(), _N*Oh*Ow*_C, 1);

        MatrixXd dmax = MatrixXd::Zero(_N*_C*Oh*Ow, _Ph*_Pw);
        int argmax_index = 0;

        for (int r = 0; r < dmax.rows(); r++)
        {
            argmax_index = _argmax(r, 0);
            dmax(r, argmax_index) = vec_dout(r, 0);
        }

        MatrixXd tmp_c_dmax = MatrixXd::Zero(_N*Oh*Ow, _Ph*_Pw);
        MatrixXd dcol = MatrixXd::Zero(_N*Oh*Ow, _C*_Ph*_Pw);

        for (int c = 0; c < _C; c++)
        {
            tmp_c_dmax = dmax.block(c*_N*Oh*Ow, 0, _N*Oh*Ow, _Ph*_Pw);
            dcol.block(0, c*_Ph*_Pw, _N*Oh*Ow, _Ph*_Pw) = tmp_c_dmax;
        }

        col2im(dcol, dX);

        grads.push_back(dX);

        return grads;
    }

    void Pooling::im2col(MatrixXd& img, MatrixXd& col)
    {
        MatrixXd pad_img;

        padding(img, pad_img);

        int Oh = (2 * _pad + _H - _Ph) / _stride + 1;
        int Ow = (2 * _pad + _W - _Pw) / _stride + 1;

        col = MatrixXd::Zero(_N * Oh * Ow, _Ph * _Pw * _C);

        int h_start = 0;
        int w_start = 0;

        int pad_H = _H + 2 * _pad;
        int pad_W = _W + 2 * _pad;

        int tmp_col_row, tmp_col_width;
        int tmp_img_start;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h_start = 0; h_start < Oh; h_start++)
                {
                    for (int w_start = 0; w_start < Ow; w_start++)
                    {
                        for (int h_offset = 0; h_offset < _Ph; h_offset++)
                        {
                            tmp_col_row = n * Oh * Ow + h_start * Ow + w_start;
                            tmp_col_width = c * _Ph * _Pw + h_offset * _Pw;
                            tmp_img_start = c * pad_H * pad_W + (h_start * _stride + h_offset) * pad_W + w_start * _stride;
                            col.block(tmp_col_row, tmp_col_width, 1, _Pw) = pad_img.block(n, tmp_img_start, 1, _Pw);
                        }
                    }
                }
            }
        }
    }

    void Pooling::col2im(MatrixXd& col, MatrixXd& img)
    {
        int pad_H = _H + 2 * _pad;
        int pad_W = _W + 2 * _pad;

        MatrixXd pad_img = MatrixXd::Zero(_N, _C*pad_H*pad_W);

        int Oh = (2 * _pad + _H - _Ph) / _stride + 1;
        int Ow = (2 * _pad + _W - _Pw) / _stride + 1;

        int tmp_img_start;
        int tmp_col_row, tmp_col_width;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < Oh; h++)
                {
                    for (int w = 0; w < Ow; w++)
                    {
                        for (int h_offset = 0; h_offset < _Ph; h_offset++)
                        {
                            tmp_img_start = pad_H*pad_W*c+(h*_stride+h_offset)*pad_W+w*_stride;
                            tmp_col_row = Oh*Ow*n+Ow*h+w;
                            tmp_col_width = c*_Ph*_Pw+h_offset*_Pw;
                            pad_img.block(n, tmp_img_start, 1, _Pw) += col.block(tmp_col_row, tmp_col_width, 1, _Pw);
                        }
                    }
                }
            }
        }

        suppress(pad_img, img);
    }

    void Pooling::padding(MatrixXd &img, MatrixXd &pad_img)
    {
        pad_img = MatrixXd::Zero(_N, _C * (_H + 2 * _pad) * (_W + 2 * _pad));

        int pad_H_elems = _H + 2 * _pad;
        int pad_W_elems = _W + 2 * _pad;
        int pad_C_elems = pad_H_elems * pad_W_elems;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < _H; h++)
                {
                    for (int w = 0; w < _W; w++)
                    {
                        // 1pixelずつ置き換え
                        pad_img(n, c * pad_C_elems + (h + _pad) * pad_W_elems + (w + _pad)) = img(n, c * (_H * _W) + _W * h + w);
                    }
                }
            }
        }
    }

    void Pooling::suppress(MatrixXd &pad_img, MatrixXd &img)
    {
        img = MatrixXd::Zero(_N, _C * _H * _W);
        int pad_H = 2 * _pad + _H;
        int pad_W = 2 * _pad + _W;

        for (int n = 0; n < _N; n++)
        {
            for (int c = 0; c < _C; c++)
            {
                for (int h = 0; h < _H; h++)
                {
                    for (int w = 0; w < _W; w++)
                    {
                        // 1pixelずつ置き換え
                        img(n, c * _H * _W + h * _W + w) = pad_img(n, c * pad_H * pad_W + (h + _pad) * pad_W + (_pad + w));
                    }
                }
            }
        }
    }
}