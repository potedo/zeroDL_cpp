#include <string>
#include <map>
#include <functional>
#include <Eigen/Dense>
#include "two_layer_net.h"
#include "simple_activation.h"
#include "simple_loss.h"
#include "numerical_gradient.h"

#include <iostream>

namespace MyDL
{

    using namespace Eigen;
    using std::cout;
    using std::endl;

    // デフォルトコンストラクタ(初期値は適当)
    TwoLayerNet::TwoLayerNet() : _input_size(3), _hidden_size(3), _output_size(2), _weight_init_std(0.01)
    {
        // paramsに格納する変数の初期化
        MatrixXd W1 = _weight_init_std * MatrixXd::Random(_input_size, _hidden_size);
        VectorXd b1 = VectorXd::Zero(_hidden_size);
        MatrixXd W2 = _weight_init_std * MatrixXd::Random(_hidden_size, _output_size);
        VectorXd b2 = VectorXd::Zero(_output_size);

        params["W1"] = W1;
        params["W2"] = W2;
        params["b1"] = b1;
        params["b2"] = b2;
    }

    // 初期値ありのコンストラクタ
    TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std) : _input_size(input_size), _hidden_size(hidden_size), _output_size(output_size), _weight_init_std(weight_init_std)
    {
        // paramsに格納する変数の初期化
        MatrixXd W1 = _weight_init_std * MatrixXd::Random(_input_size, _hidden_size);
        VectorXd b1 = VectorXd::Zero(_hidden_size);
        MatrixXd W2 = _weight_init_std * MatrixXd::Random(_hidden_size, _output_size);
        VectorXd b2 = VectorXd::Zero(_output_size);

        params["W1"] = W1;
        params["W2"] = W2;
        params["b1"] = b1;
        params["b2"] = b2;
    }

    MatrixXd TwoLayerNet::predict(MatrixXd &X)
    {
        MatrixXd a1, a2, z1, y, W1, W2;
        VectorXd b1, b2;
        W1 = params["W1"];
        W2 = params["W2"];
        b1 = params["b1"];
        b2 = params["b2"];

        // ブロードキャスト演算をするように実装(numpyとは仕様が違うことに注意)
        a1 = (X * W1).rowwise() + b1.transpose();
        z1 = sigmoid(a1);
        a2 = (z1 * W2).rowwise() + b2.transpose();
        y = softmax(a2);

        _cache["a1"] = a1;
        _cache["z1"] = z1;
        _cache["a2"] = a2;
        _cache["y"] = y;

        return y;
    }

    double TwoLayerNet::loss(MatrixXd &x, MatrixXd &t)
    {
        MatrixXd y;
        y = this->predict(x);

        double loss;
        loss = MyDL::cross_entropy_error(y, t);
        return loss;
    }

    double TwoLayerNet::accuracy(MatrixXd& x, MatrixXd& t){
        MatrixXd y;
        MatrixXd::Index y_row, y_col, t_row, t_col;

        double accuracy = 0;
        int batch_size = t.rows();
        // double max_y, max_t;

        y = this->predict(x);

        // 各行ごとに、最大要素のインデックスを取得 → インデックスが等しければ、accuracyに加算
        for (int i=0; i < batch_size; i++){
            y.row(i).maxCoeff(&y_row, &y_col);
            t.row(i).maxCoeff(&t_row, &t_col);

            accuracy += (double)(y_col == t_col); // カラムのインデックスだけ見ればOK
        }

        return accuracy / batch_size;
    }

    // 数値微分 → この微分でMNISTをやるのは計算時間がかかりすぎるので非推奨
    std::map<std::string, MatrixXd> TwoLayerNet::numerical_gradient(MatrixXd& x, MatrixXd& t){
        using std::map;
        using std::string;

        // [&]は、スコープ外の変数を参照するというキャプチャー(ここではthisポインタを使うために指定)
        std::function<double(MatrixXd)> loss_W = [this, &x, &t](MatrixXd W){return this->loss(x, t);};

        map<string, MatrixXd> grads;

        MatrixXd dW1, dW2, db1, db2;

        // キャプチャで参照を使用している場合、STLのfunctionalモジュールにおける"function"を用いてコールバックすると良い
        // また、ここではDNNのパラメータが直接関数に入力されるわけではないので、引数はparams[...]の参照を渡している
        dW1 = MyDL::numerical_gradient(loss_W, params["W1"]);
        dW2 = MyDL::numerical_gradient(loss_W, params["W2"]);
        db1 = MyDL::numerical_gradient(loss_W, params["b1"]);
        db2 = MyDL::numerical_gradient(loss_W, params["b2"]);

        grads["dW1"] = dW1;
        grads["dW2"] = dW2;
        grads["db1"] = db1;
        grads["db2"] = db2;

        return grads;
    }

    // 数値微分では遅すぎるので、誤差逆伝播法を実装
    std::map<std::string, MatrixXd> TwoLayerNet::gradient(MatrixXd& X, MatrixXd& t){
        using std::map;
        using std::string;

        map<string, MatrixXd> grads;
        MatrixXd dW1, dW2;
        VectorXd db1, db2;

        MatrixXd da2, da1, dz1;
        MatrixXd y, z1, a1, a2, W1, W2, b1, b2;
        int batch_size = t.rows();

        y  = this->predict(X); // これをコールしておかないと、_cacheの各種変数が保存されない
        a1 = _cache["a1"];
        a2 = _cache["a2"];
        z1 = _cache["z1"];
        W1 = params["W1"];
        W2 = params["W2"];
        b1 = params["b1"];
        b2 = params["b2"];

        // 逆伝播計算
        // softmax with loss layer
        da2 = (y - t) / batch_size;
        // affine layer 2
        dz1 = da2 * W2.transpose();
        dW2 = z1.transpose() * da2; // 縦ベクトル × 横ベクトル の構図(バッチ方向に縮約)
        db2 = da2.colwise().sum();
        // sigmoid layer: da1 = z1(1-z1) * dz1
        da1 = z1.array() * (MatrixXd::Ones(z1.rows(), z1.cols()) - z1).array() * dz1.array();
        // affine layer 1
        dW1 = X.transpose() * da1;
        db1 = da1.colwise().sum();
        // dx = da1 * W2.transpose() // → 今回は必要ないのでスルー

        grads["W1"] = dW1;
        grads["W2"] = dW2;
        grads["b1"] = db1;
        grads["b2"] = db2;

        return grads;
    }


}