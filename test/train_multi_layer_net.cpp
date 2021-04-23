#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "../include/sample_network.h"
#include "../include/optimizer.h"
#include "../datasets/include/mnist.h"
#include <matplotlibcpp.h>

using namespace Eigen;
namespace plt = matplotlibcpp;

int main()
{
    using std::cout;
    using std::endl;
    using std::unordered_map;
    using std::vector;
    using namespace MyDL;

    // ----------------------
    // parameters
    // ----------------------
    int NUM_DATAS = 60000;
    int epochs = 10;
    double learning_rate = 0.05; // 浮動小数点で宣言すること。うっかりint型にすると0になるので、勾配が更新できない

    int batch_size = 100;
    int input_size = 28 * 28;
    // vector<int> hidden_size = {100, 100, 100, 100, 100, 100};
    vector<int> hidden_size = {50, 25};
    // vector<int> hidden_size = {50}; // テスト用 パラメータが正しく更新できることは確認
    int output_size = 10;
    double weight_decay_lambda = 0.01;
    // double weight_decay_lambda = 0;

    int num_iters = epochs * NUM_DATAS / batch_size; // 今後のために、この辺の値をJSON等から読み込める仕組みを作る(=>アルゴリズムの実装が正しければ、パラメータは設定ファイルで管理)

    // ----------------------
    // Data Loader
    // ----------------------
    MatrixXd train_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd test_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd train_y = MatrixXd::Zero(batch_size, 10);
    MatrixXd test_y = MatrixXd::Zero(batch_size, 10);

    MnistEigenDataset mnist(batch_size);

    // ----------------------
    // DNN
    // ----------------------
    MultiLayerNet net(input_size, hidden_size, output_size, weight_decay_lambda=weight_decay_lambda);

    // ----------------------
    // Optimizer
    // ----------------------

    // SGD optimizer(0.01);
    // Momentum optimizer(0.01, 0.99);
    // AdaGrad optimizer(0.001);
    // RMSprop optimizer(0.001, 0.999);
    Adam optimizer(0.001, 0.9, 0.999);

    // ----------------------
    // データ格納用
    // ----------------------
    vector<MatrixXd> inputs, loss, val_inputs;
    unordered_map<string, MatrixXd> grads;
    double accuracy;
    // ----------------------
    // For Visualization
    // ----------------------
    vector<double> loss_history(num_iters), accuracy_history(num_iters / 10);
    vector<int> loss_counter(num_iters), accuracy_counter(num_iters / 10);

    // ----------------------
    // Learning Loop
    // ----------------------
    for (int i = 0; i < num_iters; i++)
    {
        mnist.next_train(train_X, train_y);
        inputs.push_back(train_X);
        // inputs.push_back(train_y);

        // Back Propagation
        grads = net.gradient(inputs, train_y);

        // 勾配更新 -> 後々この部分は Update という関数で記述することになる(引数はパラメータと勾配(grads))
        // for (auto grad : grads)
        // {
        //     string key = grad.first;
        //     *(net.params[key]) -= learning_rate * grad.second;
        // }

        // Update Prameters
        optimizer.update(net.params, grads);
        // cout << *(net.params["b1"]) << endl; // パラメータ更新の確認
        // if (auto affine = std::dynamic_pointer_cast<MyDL::Affine>(net._layers["Affine2"]))
        // {
        //     cout << *(affine->pb) << endl;
        // }

        // Loss Calculation
        loss = net.loss(inputs, train_y);
        loss_history[i] = loss[0](0);
        loss_counter[i] = i;

        cout << "iter" << i << " loss: " << loss[0] << endl;

        if (i % 10 == 0)
        {
            mnist.next_test(test_X, test_y);
            val_inputs.push_back(test_X);
            // val_inputs.push_back(test_y);
            accuracy = net.accuracy(val_inputs, test_y);
            cout << "accuracy: " << accuracy << endl;
            val_inputs.clear();
            accuracy_history[i / 10] = accuracy;
            accuracy_counter[i / 10] = i / 10;
        }

        inputs.clear();
    }

    // ----------------------
    // visualization
    // ----------------------
    plt::title("Loss History: Adam with Weight Decay");
    plt::plot(loss_counter, loss_history, "b");
    plt::grid(true);
    plt::save("mnist_learning_curve_Adam_with_weight_decay.png");

    plt::cla();
    plt::title("Accuracy History: Adam with Weight Decay");
    plt::plot(accuracy_counter, accuracy_history, "r");
    plt::grid(true);
    plt::save("mnist_accuracy_plot_Adam_with_weight_decay.png");

    return 0;
}