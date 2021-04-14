#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "../include/layer.h"
#include <map>
#include <memory>

using namespace Eigen;

int main()
{
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::map;
    using std::shared_ptr;
    using std::string;
    using std::unique_ptr;
    using std::vector;

    // Params
    // auto W = std::make_shared<MatrixXd>(2, 2); // 左辺はautoで問題なし
    // auto b = std::make_shared<MatrixXd>(1, 2);
    // *W = MatrixXd::Random(2, 2);
    // *b = MatrixXd::Zero(1, 2);
    map<string, shared_ptr<MatrixXd>> params;

    // Layers
    // vector<unique_ptr<BaseLayer>> layers; // 基底クラスのポインタのコンテナを用意する
    vector<shared_ptr<BaseLayer>> layers; // 基底クラスのポインタのコンテナを用意する
    map<string, unique_ptr<BaseLayer>> layers_map;
    // 抽象クラスのポインタに格納するので、unique_ptrとしてはBaseLayerに格納
    // unique_ptr<BaseLayer> p_layer1(new AddLayer()); // C++11ではmake_uniqueが存在しない
    // unique_ptr<BaseLayer> p_layer2(new MulLayer());
    // unique_ptr<BaseLayer> p_layer3(new ReLU());
    // unique_ptr<BaseLayer> p_layer4(new Sigmoid());
    // unique_ptr<BaseLayer> p_layer5(new MyDL::Affine(W, b));
    // unique_ptr<BaseLayer> p_layer6(new SoftmaxWithLoss());

    shared_ptr<BaseLayer> p_layer1 = std::make_shared<AddLayer>();
    shared_ptr<BaseLayer> p_layer2 = std::make_shared<MulLayer>();
    shared_ptr<BaseLayer> p_layer3 = std::make_shared<ReLU>();
    shared_ptr<BaseLayer> p_layer4 = std::make_shared<Sigmoid>();
    // shared_ptr<BaseLayer> p_layer5 = std::make_shared<MyDL::Affine>(W, b);
    shared_ptr<BaseLayer> p_layer5 = std::make_shared<MyDL::Affine>(2, 2); // サイズを入力するようにしてみる
    shared_ptr<BaseLayer> p_layer6 = std::make_shared<SoftmaxWithLoss>();

    // AddLayer    layer1;
    // MulLayer    layer2;
    // ReLU        layer3;
    // Sigmoid     layer4;
    // MyDL::Affine    layer5(W, b);
    // SoftmaxWithLoss layer6;

    // I/O containers
    vector<MatrixXd> inputs;
    vector<MatrixXd> outputs;
    // MatrixXd X = -2 * MatrixXd::Identity(2, 2) + MatrixXd::Ones(2, 2);
    MatrixXd X = MatrixXd::Identity(2, 2);
    MatrixXd Y = MatrixXd::Ones(2, 2);
    vector<MatrixXd> dout;
    dout.push_back(MatrixXd::Ones(2, 2));

    // inputs
    inputs.push_back(X);
    inputs.push_back(Y);

    // layers
    // layers.push_back(std::move(p_layer1)); // Layerの派生クラス(Add)のポインタをコンテナに格納
    // layers.push_back(std::move(p_layer2)); // Layerの派生クラス(Mul)のポインタをコンテナに格納
    // layers.push_back(std::move(p_layer3)); // Layerの派生クラス(ReLU)のポインタをコンテナに格納
    // layers.push_back(std::move(p_layer4)); // Layerの派生クラス(Sigmoid)のポインタをコンテナに格納
    // layers.push_back(std::move(p_layer5)); // Layerの派生クラス(Affine)のポインタをコンテナに格納
    // layers.push_back(std::move(p_layer6)); // Layerの派生クラス(SoftmaxWithLoss)のポインタをコンテナに格納

    layers.push_back(p_layer1); // Layerの派生クラス(Add)のポインタをコンテナに格納
    layers.push_back(p_layer2); // Layerの派生クラス(Mul)のポインタをコンテナに格納
    layers.push_back(p_layer3); // Layerの派生クラス(ReLU)のポインタをコンテナに格納
    layers.push_back(p_layer4); // Layerの派生クラス(Sigmoid)のポインタをコンテナに格納
    layers.push_back(p_layer5); // Layerの派生クラス(Affine)のポインタをコンテナに格納
    layers.push_back(p_layer6); // Layerの派生クラス(SoftmaxWithLoss)のポインタをコンテナに格納

    if (auto affine = std::dynamic_pointer_cast<MyDL::Affine>(p_layer5))
    {
        params["W"] = affine->pW;
        params["b"] = affine->pb;
    }

    // layers_map["Add"] = p_layer1;
    // layers_map["Affine"] = p_layer5;

    // cout << "--- map forward ---" << endl;
    // outputs = layers_map["Add"]->forward(inputs);
    // cout << outputs[0] << endl;
    // outputs = layers_map["Affine"]->forward(inputs);
    // cout << outputs[0] << endl;

    cout << "----forward----" << endl;
    for (int i = 0; i < layers.size(); i++)
    {
        cout << "----layer" << i << "----" << endl;
        outputs = layers[i]->forward(inputs); // コンテナに格納したLayerのアドレスから、メンバ関数を呼び出し(ポリモーフィズム)

        cout << outputs[0] << endl;
    }

    // Affineレイヤのパラメータを更新できるのか確認したいので、backwardの処理を走らせた後に、
    // gradsの内容を加えるという処理をし、その処理の前後でパラメータの内容が変化しているかを確認する。
    cout << "----backward----" << endl;
    for (int i = 0; i < layers.size(); i++)
    {
        vector<MatrixXd> grads;
        cout << "----layer" << i << "----" << endl;
        if (i < layers.size() - 1)
        {
            grads = layers[i]->backward(dout);
        }
        else
        {
            vector<MatrixXd> init_dout = {MatrixXd::Ones(1, 1)};
            grads = layers[i]->backward(init_dout);
        }

        for (int j = 0; j < grads.size(); j++)
        {
            cout << grads[j] << endl;
        }
    }

    if (auto affine = std::dynamic_pointer_cast<MyDL::Affine>(layers[4]))
    {
        cout << "--- Affine Param b ---" << endl;
        cout << *(affine->pb) << endl;
        cout << "params[b]:" << endl;
        cout << *(params["b"]) << endl;
        cout << "--- Affine Param W ---" << endl;
        cout << *(affine->pW) << endl;
        cout << "params[W]:" << endl;
        cout << *(params["W"]) << endl;

        *(params["b"]) -= affine->db;
        *(params["W"]) -= affine->dW;
    }
    // 別のスコープでアクセスしたときに更新されているか？
    if (auto re_affine = std::dynamic_pointer_cast<MyDL::Affine>(layers[4]))
    {
        cout << "--- after update ---" << endl; // 内部で処理しないと、affineが宣言されていない、となる。スコープ外？
        cout << *(re_affine->pb) << endl;
        cout << *(re_affine->pW) << endl;
    }

    cout << "params[b]" << endl;
    cout << *(params["b"]) << endl;
    cout << "params[W]" << endl;
    cout << *(params["W"]) << endl;

    return 0;
}