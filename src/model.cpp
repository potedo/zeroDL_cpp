#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include "../include/model.h"
#include "../include/numerical_gradient.h"

namespace MyDL
{

    using namespace Eigen;
    using std::cout;
    using std::endl;
    using std::make_shared;
    using std::shared_ptr;
    using std::string;
    using std::unordered_map;
    using std::vector;

    TwoLayerMLP::TwoLayerMLP(int input_size, int hidden_size, int output_size, double weight_init_std)
    {
        // 内部保持パラメータ
        _input_size = input_size;
        _hidden_size = hidden_size;
        _output_size = output_size;
        _weight_init_std = weight_init_std;

        // Affine Layer用のパラメータ -> スマートポインタで保持し、それをパラメータのリストに格納
        auto W1 = make_shared<MatrixXd>(input_size, hidden_size);
        auto W2 = make_shared<MatrixXd>(hidden_size, output_size);
        auto b1 = make_shared<MatrixXd>(1, hidden_size);
        auto b2 = make_shared<MatrixXd>(1, output_size);

        *W1 = weight_init_std * MatrixXd::Random(input_size, hidden_size);
        *W2 = weight_init_std * MatrixXd::Random(hidden_size, output_size);
        *b1 = MatrixXd::Zero(1, hidden_size);
        *b2 = MatrixXd::Zero(1, output_size);

        // Layer作成 -> スマートポインタで実装(コンストラクタを抜けた時に、実体が消されないようにするため)
        // 左辺の型はautoにしてはいけない(BaseLayerで統一し、コンテナに格納する ※ポリモーフィズムの実現)
        shared_ptr<BaseLayer> affine1 = make_shared<MyDL::Affine>(W1, b1);
        shared_ptr<BaseLayer> affine2 = make_shared<MyDL::Affine>(W2, b2);
        shared_ptr<BaseLayer> relu1 = make_shared<ReLU>();
        shared_ptr<BaseLayer> last_layer = make_shared<SoftmaxWithLoss>(); // predictではaffine2の出力、lossではlastlayerの出力を使うので分ける

        // 生のポインタを格納すると実体がスコープ外となり解放されてしまうので、shared_ptrで対処
        _layers["Affine1"] = affine1;
        _layers["ReLU1"] = relu1;
        _layers["Affine2"] = affine2;
        _last_layer = last_layer;

        // 各パラメータへのポインタを格納
        if (auto cast_affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            params["W1"] = cast_affine1->pW;
            params["b1"] = cast_affine1->pb;
        }
        if (auto cast_affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            params["W2"] = cast_affine2->pW;
            params["b2"] = cast_affine2->pb;
        }

        // unordered_mapでは追加順が保存されないので、別途順番通り名称を格納したコンテナを用意
        _layer_list.push_back("Affine1");
        // _layer_list.push_back("BatchNorm"); // for batchnorm debug 21/03/21追加
        _layer_list.push_back("ReLU1");
        _layer_list.push_back("Affine2");

        // ------------------------------------------------
        // for batchnorm debug 21/03/21追加
        // ------------------------------------------------
        // auto gamma = make_shared<MatrixXd>(1, hidden_size);
        // auto beta = make_shared<MatrixXd>(1, hidden_size);
        // *gamma = weight_init_std * MatrixXd::Random(1, hidden_size);
        // *beta = MatrixXd::Zero(1, hidden_size);
        // shared_ptr<BaseLayer> batch_norm = make_shared<BatchNorm>(gamma, beta);
        // _layers["BatchNorm"] = batch_norm;
        // if (auto cast_batchnorm = std::dynamic_pointer_cast<BatchNorm>(_layers["BatchNorm"]))
        // {
        //     params["gamma"] = cast_batchnorm->pgamma;
        //     params["beta"] = cast_batchnorm->pbeta;
        // }
    }

    vector<MatrixXd> TwoLayerMLP::predict(vector<MatrixXd> inputs)
    {
        // inputのバリデーションをしておくか？
        vector<MatrixXd> X = inputs; // 入力もvectorなので、そのまま受ければOK
        vector<MatrixXd> tmp_X;

        // mapのrange-forは内部的にstd::pairが返される
        for (auto layer : _layer_list)
        {
            // cout << layer << endl;
            tmp_X = _layers[layer]->forward(X);
            X.swap(tmp_X); // 中身入れ替え
        }
        return X;
    }

    vector<MatrixXd> TwoLayerMLP::loss(vector<MatrixXd> inputs, MatrixXd& t)
    {
        vector<MatrixXd> pred_input, pred_out, loss_inputs, loss_output;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        loss_inputs.push_back(pred_out[0]);
        loss_inputs.push_back(t);

        loss_output = _last_layer->forward(loss_inputs);
        return loss_output;
    }

    double TwoLayerMLP::accuracy(vector<MatrixXd> inputs, MatrixXd& t)
    {
        vector<MatrixXd> pred_out;
        pred_out = predict(inputs);

        MatrixXd y;
        y = pred_out[0];
        double batch_size = t.rows();
        double accuracy = 0;

        MatrixXd::Index y_row, y_col, t_row, t_col;
        for (int i = 0; i < batch_size; i++)
        {
            y.row(i).maxCoeff(&y_row, &y_col);
            t.row(i).maxCoeff(&t_row, &t_col);

            accuracy += (double)(y_col == t_col);
        }

        return accuracy / batch_size;
    }

    unordered_map<string, MatrixXd> TwoLayerMLP::gradient(vector<MatrixXd> inputs, MatrixXd& t)
    {
        // Forward
        vector<MatrixXd> output;
        output = loss(inputs, t); // forward -> 逆伝播計算に必要な情報を各レイヤにキャッシュ

        // Backward
        vector<MatrixXd> dout, tmp_dout;
        dout.push_back(MatrixXd::Ones(1, 1));

        dout = _last_layer->backward(dout);

        // 逆順ループ → Boostライブラリのboost::adaptors::reverse()を使う方がEasyではある
        for (auto it = _layer_list.rbegin(); it != _layer_list.rend(); it++)
        {
            string layer = *it;
            tmp_dout = _layers[layer]->backward(dout);
            dout.swap(tmp_dout);
        }

        unordered_map<string, MatrixXd> grads;
        // _layersに格納している変数はBaseLayerにアップキャストしているので、ダウンキャストが必要 -> nullptrのときは実行しないようにする
        if (shared_ptr<MyDL::Affine> affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            grads["W1"] = affine1->dW;
            grads["b1"] = affine1->db;
        }
        if (shared_ptr<MyDL::Affine> affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            grads["W2"] = affine2->dW;
            grads["b2"] = affine2->db;
        }

        // for batchnorm debug 21/03/21追加
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(_layers["BatchNorm"]))
        {
            grads["gamma"] = batchnorm->dgamma;
            grads["beta"] = batchnorm->dbeta;
        }

        return grads;
    }

    // unordered_map<string, MatrixXd> TwoLayerMLP::numerical_gradient(vector<MatrixXd> inputs)
    // {
    //     // [&]は、スコープ外の変数を参照するというキャプチャー(ここではthisポインタを使うために指定)
    //     std::function<vector<MatrixXd>(MatrixXd)> loss_W = [this, &inputs](MatrixXd W) -> vector<MatrixXd> { return this->loss(inputs); };
    //     std::function<vector<MatrixXd>(VectorXd)> loss_W2 = [this, &inputs](VectorXd W) -> vector<MatrixXd> { return this->loss(inputs); };
    //     unordered_map<string, MatrixXd> grads;

    //     MatrixXd dW1, dW2, db1, db2;

    //     // 直接内部のレイヤのパラメータにアクセスするので、ダウンキャストが必要
    //     if (shared_ptr<MyDL::Affine> affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
    //     {
    //         dW1 = MyDL::numerical_gradient(loss_W, affine1->_W);
    //         db1 = MyDL::numerical_gradient(loss_W2, affine1->_b);
    //     }
    //     if (shared_ptr<MyDL::Affine> affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
    //     {
    //         dW2 = MyDL::numerical_gradient(loss_W, affine2->_W);
    //         db2 = MyDL::numerical_gradient(loss_W2, affine2->_b);
    //     }

    //     grads["dW1"] = dW1;
    //     grads["dW2"] = dW2;
    //     grads["db1"] = db1;
    //     grads["db2"] = db2;

    //     return grads;
    // }

    // -----------------------------------------------------------
    // MultiLayerModel: Weight Decay の検証用
    // 【21/04/06】
    // とりあえず動作することを目指すので、最初はreluで構成
    // 後ほどsigmoid含めて動くように変更。初期値も XavierとHeの両方を選択できるようにする。
    // -----------------------------------------------------------

    MultiLayerModel::MultiLayerModel(const int input_size,
                                     const vector<int> hidden_size,
                                     const int output_size,
                                     const double weight_decay_lambda,
                                     string activation,
                                     const string weight_initializer,
                                     const bool use_dropout,
                                     const double dropout_ratio,
                                     const bool use_batchnorm)
    {
        _input_size = input_size;
        _hidden_size_list = hidden_size;
        _output_size = output_size;
        _weight_decay_lambda = weight_decay_lambda;

        _all_size_list.insert(_all_size_list.end(), {_input_size});
        _all_size_list.insert(_all_size_list.end(), _hidden_size_list.begin(), _hidden_size_list.end());
        _all_size_list.insert(_all_size_list.end(), {_output_size});

        std::transform(activation.begin(), activation.end(), activation.begin(), ::tolower);

        // Create Layers
        for (int i = 1; i < _hidden_size_list.size()+1; i++)
        {
            string tmp_num_str = std::to_string(i);
            _layers["Affine" + tmp_num_str] = make_shared<MyDL::Affine>(_all_size_list[i-1], _all_size_list[i]);
            _layer_list.push_back("Affine" + tmp_num_str);

            // BatchNormalization
            if (use_batchnorm)
            {
                _layers["BatchNorm" + tmp_num_str] = make_shared<BatchNorm>(_all_size_list[i], 0.9); // パラメータも指定できるようにする？
                _layer_list.push_back("BatchNorm" + tmp_num_str);
            }

            // Activation
            if (activation == "relu")
            {
                _layers["ReLU" + tmp_num_str] = make_shared<ReLU>();
                _layer_list.push_back("ReLU" + tmp_num_str);
            }
            else if(activation == "sigmoid") 
            {
                _layers["Sigmoid" + tmp_num_str] = make_shared<Sigmoid>();
                _layer_list.push_back("Sigmoid" + tmp_num_str);
            }

            // Dropout
            if (use_dropout)
            {
                _layers["Dropout" + tmp_num_str] = make_shared<Dropout>(dropout_ratio);
                _layer_list.push_back("Dropout" + tmp_num_str);
            }

        }
        int last_num = _hidden_size_list.size()+1;
        string last_num_str = std::to_string(last_num);
        _layers["Affine" + last_num_str] = make_shared<MyDL::Affine>(_all_size_list[last_num-1], _all_size_list[last_num]);
        _layer_list.push_back("Affine" + last_num_str);

        _last_layer = make_shared<SoftmaxWithLoss>(); // Loss Layer


        // Get pointer to Layer Parameters
        for (int layer_num = 1; layer_num < _all_size_list.size(); layer_num++)
        {
            if (auto cast_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine" + std::to_string(layer_num)]))
            {
                params["W" + std::to_string(layer_num)] = cast_affine->pW;
                params["b" + std::to_string(layer_num)] = cast_affine->pb;
            }
        }

        // Affine Layerの変数初期化
        this->_init_weight(weight_initializer);

    }


    void MultiLayerModel::_init_weight(string weight_initializer)
    {
        // weight_initializerの文字列を小文字に変換
        std::transform(weight_initializer.begin(),
                       weight_initializer.end(),
                       weight_initializer.begin(),
                       ::tolower);
        
        double scale = 1;

        if (weight_initializer == "relu" or weight_initializer == "he")
        {
            for (int layer_num = 1; layer_num <= _hidden_size_list.size()+1; layer_num++)
            {
                scale = sqrt(2.0 / _all_size_list[layer_num-1]);
                *(params["W" + std::to_string(layer_num)]) = scale * MatrixXd::Random(_all_size_list[layer_num-1], _all_size_list[layer_num]);        
            }
        }
        else if (weight_initializer == "sigmoid" or weight_initializer == "xavier")
        {
            for (int layer_num = 1; layer_num <= _hidden_size_list.size() + 1; layer_num++)
            {
                scale = sqrt(1.0 / _all_size_list[layer_num - 1]);
                *(params["W" + std::to_string(layer_num)]) = scale * MatrixXd::Random(_all_size_list[layer_num - 1], _all_size_list[layer_num]);
            }
        }
    }


    vector<MatrixXd> MultiLayerModel::predict(vector<MatrixXd> inputs)
    {
        vector<MatrixXd> X = inputs;
        vector<MatrixXd> tmp_X;

        for (auto layer : _layer_list)
        {
            tmp_X = _layers[layer]->forward(X);
            X.swap(tmp_X);
        }
        return X;
    }

    vector<MatrixXd> MultiLayerModel::loss(vector<MatrixXd> inputs, MatrixXd &t)
    {
        vector<MatrixXd> pred_input, pred_out, loss_inputs, loss_output;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        loss_inputs.push_back(pred_out[0]);
        loss_inputs.push_back(t);

        loss_output = _last_layer->forward(loss_inputs);

        // あとは Weight Decay の項も計算してLossに加える
        double weight_decay = 0;
        for (auto param : params)
        {
            weight_decay += 0.5 * _weight_decay_lambda * (*(param.second)).sum();
        }

        loss_output[0](0) = loss_output[0](0) + weight_decay;

        return loss_output;
    }

    double MultiLayerModel::accuracy(vector<MatrixXd> inputs, MatrixXd &t)
    {
        vector<MatrixXd> pred_out;
        pred_out = predict(inputs);

        MatrixXd y;
        y = pred_out[0];
        double batch_size = t.rows();
        double accuracy = 0;

        MatrixXd::Index y_row, y_col, t_row, t_col;
        for (int i = 0; i < batch_size; i++)
        {
            y.row(i).maxCoeff(&y_row, &y_col);
            t.row(i).maxCoeff(&t_row, &t_col);

            accuracy += (double)(y_col == t_col);
        }

        return accuracy / batch_size;
    }

    unordered_map<string, MatrixXd> MultiLayerModel::gradient(vector<MatrixXd> inputs, MatrixXd &t)
    {
        // Forward
        vector<MatrixXd> output;
        output = loss(inputs, t);

        // Backward
        vector<MatrixXd> dout, tmp_dout;
        dout.push_back(MatrixXd::Ones(1, 1));

        dout = _last_layer->backward(dout);

        for (auto it = _layer_list.rbegin(); it != _layer_list.rend(); it++)
        {
            string layer = *it;
            tmp_dout = _layers[layer]->backward(dout);
            dout.swap(tmp_dout);
        }

        unordered_map<string, MatrixXd> grads;

        for (int i = 1; i <= _hidden_size_list.size() + 1; i++)
        {
            if (auto tmp_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine" + std::to_string(i)]))
            {
                grads["W" + std::to_string(i)] = tmp_affine->dW + _weight_decay_lambda * (*(tmp_affine->pW));
                grads["b" + std::to_string(i)] = tmp_affine->db;
            }
        }

        return grads;
    }

    // -----------------------------------------------------------
    // SimpleConvModel: Convolutionの検証用
    // -----------------------------------------------------------

    SimpleConvModel::SimpleConvModel(const int input_channels,
                                     const int input_height,
                                     const int input_width,
                                     const int filter_num,
                                     const int filter_size,
                                     const int pad,
                                     const int stride,
                                     const int hidden_size,
                                     const int output_size,
                                     const double weight_init_std)
    : _C(input_channels), _H(input_height), _W(input_width), _filter_num(filter_num), _filter_size(filter_size), _pad(pad), _stride(stride), _hidden_size(hidden_size), _output_size(output_size)
    {
        int Oh = (_pad*2 + _H - _filter_size) / _stride + 1;
        int Ow = (_pad*2 + _W - _filter_size) / _stride + 1;
        int Ph = 2;
        int Pw = 2; // Poolingのサイズは固定
        int p_stride = 2;
        int p_pad = 0;

        int pool_output_size = _filter_num * (Oh / 2) * (Ow / 2);

        _layers["Conv1"] = make_shared<Conv2D>(_C, _H, _W, _filter_size, _filter_size, _filter_num, _stride, _pad, weight_init_std);
        _layers["Relu1"] = make_shared<ReLU>();
        _layers["Pool1"] = make_shared<Pooling>(_filter_num, Oh, Ow, Ph, Pw, p_stride, p_pad);
        _layers["Affine1"] = make_shared<MyDL::Affine>(pool_output_size, _hidden_size, weight_init_std);
        _layers["Affine2"] = make_shared<MyDL::Affine>(hidden_size, output_size, weight_init_std);

        _last_layer = make_shared<SoftmaxWithLoss>();

        _layer_list.push_back("Conv1");
        _layer_list.push_back("Relu1");
        _layer_list.push_back("Pool1");
        _layer_list.push_back("Affine1");
        _layer_list.push_back("Affine2");

        if (auto cast_conv = std::dynamic_pointer_cast<Conv2D>(_layers["Conv1"]))
        {
            params["W1"] = cast_conv->pW;
            params["b1"] = cast_conv->pb;
        }

        if (auto cast_affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            params["W2"] = cast_affine1->pW;
            params["b2"] = cast_affine1->pb;
        }

        if (auto cast_affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            params["W3"] = cast_affine2->pW;
            params["b3"] = cast_affine2->pb;
        }

    }

    vector<MatrixXd> SimpleConvModel::predict(vector<MatrixXd> inputs)
    {
        vector<MatrixXd> X = inputs;
        vector<MatrixXd> tmp_X;

        for (auto layer : _layer_list)
        {
            tmp_X = _layers[layer]->forward(X);
            X.swap(tmp_X);
        }
        return X;
    }

    vector<MatrixXd> SimpleConvModel::loss(vector<MatrixXd> inputs, MatrixXd &t)
    {
        vector<MatrixXd> pred_input, pred_out, loss_inputs, loss_output;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        loss_inputs.push_back(pred_out[0]);
        loss_inputs.push_back(t);

        loss_output = _last_layer->forward(loss_inputs);

        return loss_output;
    }

    double SimpleConvModel::accuracy(vector<MatrixXd> inputs, MatrixXd &t)
    {
        vector<MatrixXd> pred_out;
        pred_out = predict(inputs);

        MatrixXd y;
        y = pred_out[0];
        double batch_size = t.rows();
        double accuracy = 0;

        MatrixXd::Index y_row, y_col, t_row, t_col;
        for (int i = 0; i < batch_size; i++)
        {
            y.row(i).maxCoeff(&y_row, &y_col);
            t.row(i).maxCoeff(&t_row, &t_col);

            accuracy += (double)(y_col == t_col);
        }

        return accuracy / batch_size;
    }

    unordered_map<string, MatrixXd> SimpleConvModel::gradient(vector<MatrixXd> inputs, MatrixXd &t)
    {
        // Forward
        vector<MatrixXd> output;
        output = loss(inputs, t);

        // Backward
        vector<MatrixXd> dout, tmp_dout;
        dout.push_back(MatrixXd::Ones(1, 1));

        dout = _last_layer->backward(dout);

        for (auto it = _layer_list.rbegin(); it != _layer_list.rend(); it++)
        {
            string layer = *it;
            tmp_dout = _layers[layer]->backward(dout);
            dout.swap(tmp_dout);
        }

        unordered_map<string, MatrixXd> grads;

        if (auto cast_conv = std::dynamic_pointer_cast<Conv2D>(_layers["Conv1"]))
        {
            grads["W1"] = cast_conv->dW;
            grads["b1"] = cast_conv->db;
        }

        if (auto cast_affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            grads["W2"] = cast_affine1->dW;
            grads["b2"] = cast_affine1->db;
        }

        if (auto cast_affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            grads["W3"] = cast_affine2->dW;
            grads["b3"] = cast_affine2->db;
        }

        return grads;
    }
}