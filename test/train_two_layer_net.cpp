#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "../include/sample_network.h"
#include "../include/optimizer.h"
#include "../datasets/include/mnist.h"
#include "../matplotlibcpp.h"

using namespace Eigen;
namespace plt = matplotlibcpp;

int main()
{
    using std::cout;
    using std::endl;
    using std::vector;
    using std::unordered_map;
    using namespace MyDL;

    // ----------------------
    // parameters
    // ----------------------
    int num_iters = 6000; // 今後のために、この辺の値をJSON等から読み込める仕組みを作る(=>アルゴリズムの実装が正しければ、パラメータは設定ファイルで管理)
    double learning_rate = 0.05; // 浮動小数点で宣言すること。うっかりint型にすると0になるので、勾配が更新できない

    int batch_size = 100;
    int input_size = 28*28;
    int hidden_size = 50;
    int output_size =10;

    // ----------------------
    // Data Loader
    // ----------------------
    MatrixXd train_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd test_X  = MatrixXd::Zero(batch_size, input_size);
    MatrixXd train_y = MatrixXd::Zero(batch_size, 10);
    MatrixXd test_y  = MatrixXd::Zero(batch_size, 10);

    MnistEigenDataset mnist(batch_size);

    // ----------------------
    // DNN
    // ----------------------
    TwoLayerNet net(input_size, hidden_size, output_size);

    // ----------------------
    // Optimizer
    // ----------------------

    // SGD optimizer(0.05);
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
    vector<double> loss_history(num_iters), accuracy_history(num_iters/10);
    vector<int> loss_counter(num_iters), accuracy_counter(num_iters/10);

    // ----------------------
    // Learning Loop
    // ----------------------
    for (int i = 0; i < num_iters; i++)
    {
        mnist.next_train(train_X, train_y, true, true);
        inputs.push_back(train_X);
        inputs.push_back(train_y);

        // Back Propagation
        grads = net.gradient(inputs);

        // 勾配更新 -> 後々この部分は Update という関数で記述することになる(引数はパラメータと勾配(grads))
        // for (auto grad : grads)
        // {
        //     string key = grad.first;
        //     *(net.params[key]) -= learning_rate * grad.second;
        // }

        // Update Prameters
        optimizer.update(net.params, grads);

        // Loss Calculation
        loss = net.loss(inputs);
        loss_history[i] = loss[0](0);
        loss_counter[i] = i;

        cout << "iter" << i << " loss: " << loss[0] << endl;

        if (i % 10 == 0)
        {
            mnist.next_test(test_X, test_y, true, true);
            val_inputs.push_back(test_X);
            val_inputs.push_back(test_y);
            accuracy = net.accuracy(val_inputs);
            cout << "accuracy: " << accuracy << endl;
            val_inputs.clear();
            accuracy_history[i/10] = accuracy;
            accuracy_counter[i/10]= i / 10;
        }

        inputs.clear();
    }

    // ----------------------
    // visualization
    // ----------------------
    plt::title("Loss History: Adam with BatchNorm");
    plt::plot(loss_counter, loss_history, "b");
    plt::grid(true);
    plt::save("mnist_learning_curve_Adam_with_batch_norm.png");

    plt::cla();
    plt::title("Accuracy History: Adam with BatchNorm");
    plt::plot(accuracy_counter, accuracy_history, "r");
    plt::grid(true);
    plt::save("mnist_accuracy_plot_Adam_with_batch_norm.png");

    return 0;
}