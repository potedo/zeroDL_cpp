#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include "../include/sample_network.h"
#include "../include/numerical_gradient.h"

namespace MyDL{

    using namespace Eigen;
    using std::shared_ptr;
    using std::make_shared;
    using std::string;
    using std::vector;
    using std::unordered_map;
    using std::cout;
    using std::endl;

    TwoLayerNet::TwoLayerNet(int input_size, int hidden_size, int output_size, double weight_init_std)
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
        shared_ptr<BaseLayer> affine1    = make_shared<MyDL::Affine>(W1, b1);
        shared_ptr<BaseLayer> affine2    = make_shared<MyDL::Affine>(W2, b2);
        shared_ptr<BaseLayer> relu1      = make_shared<ReLU>();
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
        _layer_list.push_back("BatchNorm"); // for batchnorm debug 21/03/21追加
        _layer_list.push_back("ReLU1");
        _layer_list.push_back("Affine2");

        // ------------------------------------------------
        // for batchnorm debug 21/03/21追加
        // ------------------------------------------------
        auto gamma = make_shared<MatrixXd>(1, hidden_size);
        auto beta = make_shared<MatrixXd>(1, hidden_size);
        *gamma = weight_init_std * MatrixXd::Random(1, hidden_size);
        *beta = MatrixXd::Zero(1, hidden_size);
        shared_ptr<BaseLayer> batch_norm = make_shared<BatchNorm>(gamma, beta);
        _layers["BatchNorm"] = batch_norm;
        if (auto cast_batchnorm = std::dynamic_pointer_cast<BatchNorm>(_layers["BatchNorm"]))
        {
            params["gamma"] = cast_batchnorm->pgamma;
            params["beta"]  = cast_batchnorm->pbeta;
        }
    }

    vector<MatrixXd> TwoLayerNet::predict(vector<MatrixXd> inputs)
    {
        // inputのバリデーションをしておくか？
        vector<MatrixXd> X = inputs;// 入力もvectorなので、そのまま受ければOK
        vector<MatrixXd> tmp_X;

        // mapのrange-forは内部的にstd::pairが返される
        for(auto layer : _layer_list)
        {
            // cout << layer << endl;
            tmp_X = _layers[layer]->forward(X);
            X.swap(tmp_X); // 中身入れ替え
        }
        return X;
    }

    vector<MatrixXd> TwoLayerNet::loss(vector<MatrixXd> inputs)
    {
        vector<MatrixXd> pred_input, pred_out, loss_inputs, loss_output;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        loss_inputs = inputs;
        loss_inputs[0] = pred_out[0];

        loss_output = _last_layer->forward(loss_inputs);
        return loss_output;
    }

    double TwoLayerNet::accuracy(vector<MatrixXd> inputs)
    {
        vector<MatrixXd> pred_input, pred_out;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        MatrixXd y, t;
        y = pred_out[0];
        t = inputs[1];
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

    unordered_map<string, MatrixXd> TwoLayerNet::gradient(vector<MatrixXd> inputs)
    {
        // Forward
        vector<MatrixXd> output;
        output = loss(inputs); // forward -> 逆伝播計算に必要な情報を各レイヤにキャッシュ

        // Backward
        vector<MatrixXd> dout, tmp_dout;
        dout.push_back(MatrixXd::Ones(1,1));

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
        if(shared_ptr<MyDL::Affine> affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            grads["W1"] = affine1->dW;
            grads["b1"] = affine1->db;
        }
        if(shared_ptr<MyDL::Affine> affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            grads["W2"] = affine2->dW;
            grads["b2"] = affine2->db;
        }

        // for batchnorm debug 21/03/21追加
        if(auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(_layers["BatchNorm"]))
        {
            grads["gamma"] = batchnorm->dgamma;
            grads["beta"]  = batchnorm->dbeta;
        }

        return grads;
    }

    unordered_map<string, MatrixXd> TwoLayerNet::numerical_gradient(vector<MatrixXd> inputs)
    {
        // [&]は、スコープ外の変数を参照するというキャプチャー(ここではthisポインタを使うために指定)
        std::function<vector<MatrixXd>(MatrixXd)> loss_W = [this, &inputs](MatrixXd W) -> vector<MatrixXd> { return this->loss(inputs); };
        std::function<vector<MatrixXd>(VectorXd)> loss_W2 = [this, &inputs](VectorXd W) -> vector<MatrixXd> { return this->loss(inputs); };
        unordered_map<string, MatrixXd> grads;

        MatrixXd dW1, dW2, db1, db2;

        // 直接内部のレイヤのパラメータにアクセスするので、ダウンキャストが必要
        if (shared_ptr<MyDL::Affine> affine1 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine1"]))
        {
            dW1 = MyDL::numerical_gradient(loss_W, affine1->_W);
            db1 = MyDL::numerical_gradient(loss_W2, affine1->_b);
        }
        if (shared_ptr<MyDL::Affine> affine2 = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine2"]))
        {
            dW2 = MyDL::numerical_gradient(loss_W, affine2->_W);
            db2 = MyDL::numerical_gradient(loss_W2, affine2->_b);
        }

        grads["dW1"] = dW1;
        grads["dW2"] = dW2;
        grads["db1"] = db1;
        grads["db2"] = db2;

        return grads;
    }

    // -----------------------------------------------------------
    // MultiLayerNet: Weight Decay の検証用
    // 【21/04/06】
    // とりあえず動作することを目指すので、最初はreluで構成
    // 後ほどsigmoid含めて動くように変更。初期値も XavierとHeの両方を選択できるようにする。
    // -----------------------------------------------------------

    MultiLayerNet::MultiLayerNet(const int input_size,
                                 const vector<int> hidden_size, 
                                 const int output_size,
                                 const double weight_decay_lambda)
    {
        _input_size = input_size;
        _hidden_size_list = hidden_size;
        _output_size = output_size;
        _weight_decay_lambda = weight_decay_lambda;

        // 各パラメータの初期化

        _layers["Affine1"] = make_shared<MyDL::Affine>(_input_size, _hidden_size_list[0]);
        _layers["ReLU1"] = make_shared<ReLU>();
        _layer_list.push_back("Affine1");
        _layer_list.push_back("ReLU1");
        for (int i=0; i<_hidden_size_list.size()-1; i++)
        {
            string tmp_num_str = std::to_string(i+2);
            _layers["Affine" + tmp_num_str] = make_shared<MyDL::Affine>(_hidden_size_list[i], _hidden_size_list[i + 1]);
            _layers["ReLU" + tmp_num_str] = make_shared<ReLU>();
            _layer_list.push_back("Affine" + tmp_num_str);
            _layer_list.push_back("ReLU" + tmp_num_str);
        }
        string tmp_num_str = std::to_string(_hidden_size_list.size()+1);
        _layers["Affine" + tmp_num_str] = make_shared<MyDL::Affine>(_hidden_size_list[_hidden_size_list.size() - 1], _output_size);
        _layers["ReLU" + tmp_num_str] = make_shared<ReLU>();
        _layer_list.push_back("Affine" + tmp_num_str);
        _layer_list.push_back("ReLU" + tmp_num_str);

        _last_layer = make_shared<SoftmaxWithLoss>(); // Loss Layer

        for (int layer_num = 1; layer_num <= _hidden_size_list.size()+1; layer_num++)
        {
            if (auto cast_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine" + std::to_string(layer_num)]))
            {
                params["W" + std::to_string(layer_num)] = cast_affine->pW;
                params["b" + std::to_string(layer_num)] = cast_affine->pb;
            }
        }        

    }


    vector<MatrixXd> MultiLayerNet::predict(vector<MatrixXd> inputs)
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

    vector<MatrixXd> MultiLayerNet::loss(vector<MatrixXd> inputs, MatrixXd& t)
    {
        vector<MatrixXd> pred_input, pred_out, loss_inputs, loss_output;
        pred_input.push_back(inputs[0]);
        pred_out = predict(pred_input);

        loss_inputs.push_back(pred_out[0]);
        loss_inputs.push_back(t);

        loss_output = _last_layer->forward(loss_inputs);

        // あとは Weight Decay の項も計算してLossに加える
        double weight_decay = 0;
        for (auto param: params)
        {
            weight_decay += 0.5 * _weight_decay_lambda * (*(param.second)).sum();
        }

        loss_output[0](0) = loss_output[0](0) + weight_decay;

        return loss_output;
    }

    double MultiLayerNet::accuracy(vector<MatrixXd> inputs, MatrixXd& t)
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

    unordered_map<string, MatrixXd> MultiLayerNet::gradient(vector<MatrixXd> inputs, MatrixXd& t)
    {
        // Forward
        vector<MatrixXd> output;
        output = loss(inputs, t);

        // Backward
        vector<MatrixXd> dout, tmp_dout;
        dout.push_back(MatrixXd::Ones(1,1));

        dout = _last_layer->backward(dout);

        for (auto it = _layer_list.rbegin(); it != _layer_list.rend(); it++)
        {
            string layer = *it;
            tmp_dout = _layers[layer]->backward(dout);
            dout.swap(tmp_dout);
        }

        unordered_map<string, MatrixXd> grads;

        for (int i = 1; i <= _hidden_size_list.size()+1; i++)
        {
            if(auto tmp_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine" + std::to_string(i)]))
            {
                grads["W" + std::to_string(i)] = tmp_affine->dW + _weight_decay_lambda * (*(tmp_affine->pW));
                grads["b" + std::to_string(i)] = tmp_affine->db;
            }
        }

        return grads;
    }



    // -----------------------------------------------------------
    // AffineLayer デバッグ用クラス → 参照先の更新確認
    // -----------------------------------------------------------
    DebugAffine::DebugAffine(int input_size, int output_size, double weight_init_std)
    {
        _input_size = input_size;
        _output_size = output_size;

        MatrixXd W = weight_init_std * MatrixXd::Random(input_size, output_size);
        MatrixXd b = MatrixXd::Zero(1, output_size);

        shared_ptr<BaseLayer> affine = std::make_shared<MyDL::Affine>(W, b);
        _layers["Affine"] = affine;

        if(shared_ptr<MyDL::Affine> tmp_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine"]))
        {
            params["W"] = tmp_affine->_W; // ここではコピーコンストラクタが走る
            params["b"] = tmp_affine->_b;
        }
    }

    void DebugAffine::PrintLayerParams(void)
    {
        if (shared_ptr<MyDL::Affine> tmp_affine = std::dynamic_pointer_cast<MyDL::Affine>(_layers["Affine"]))
        {
            cout << "--- parameter W ---" << endl;
            cout << tmp_affine->_W << endl;
            cout << "--- parameter b ---" << endl;
            cout << tmp_affine->_b << endl;
        }
    }

}