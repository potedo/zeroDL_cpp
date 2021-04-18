#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "../simple_lib/include/two_layer_net.h"
#include "../datasets/include/mnist.h"
#include <matplotlibcpp.h>

using namespace Eigen;
namespace plt = matplotlibcpp;

int main()
{
    using std::cout;
    using std::endl;
    using std::map;
    using std::vector;
    using std::string;
    using namespace MyDL;

    // ハイパーパラメータ
    int num_iters = 3000;
    double learning_rate = 0.05;
    int batch_size = 100;
    int input_size = 28 * 28;
    int hidden_size = 100;
    int output_size = 10;

    // MNISTデータローダ
    MnistEigenDataset mnist(batch_size);

    // 各種変数初期化
    MatrixXd train_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd train_y = MatrixXd::Zero(batch_size, output_size);
    MatrixXd test_X = MatrixXd::Zero(batch_size, input_size);
    MatrixXd test_y = MatrixXd::Zero(batch_size, output_size);
    bool one_hot_label = true;

    // ネットワーク生成
    TwoLayerNet net(input_size, hidden_size, output_size, 0.01);

    // 最適化用
    map<string, MatrixXd> grads;
    double loss;
    double accuracy;

    // 学習経過プロット用
    vector<double> loss_history(num_iters);
    vector<int> plot_counter(num_iters);
    vector<double> accuracy_history(num_iters / 10);
    vector<int> accuracy_counter(num_iters / 10);

    // 最適化実行
    for (int i = 0; i < num_iters; i++){
        // 次のミニバッチ取得
        mnist.next_train(train_X, train_y, one_hot_label);
        
        // 勾配計算
        grads = net.gradient(train_X, train_y); // 内部で
        
        // 勾配更新
        for (auto i = grads.begin(); i != grads.end(); i++){
            net.params[i->first] -= learning_rate * grads[i->first];
        }

        // 損失プロット用
        loss = net.loss(train_X, train_y);
        loss_history[i] = loss;
        plot_counter[i] = i;

        cout << "iteration" << i << " loss: " << loss << endl;

        // 10step毎にaccuracy計測
        if (i % 10 == 0){
            mnist.next_test(test_X, test_y, one_hot_label);
            accuracy = net.accuracy(test_X, test_y);

            cout << "accuracy: " << accuracy << endl;

            accuracy_history[i/10] = accuracy;
            accuracy_counter[i/10] = i / 10;
        }
    }

    // visualize
    plt::title("Loss History");
    plt::plot(plot_counter, loss_history, "b");
    plt::grid(true);
    plt::save("ch4/mnist_learning_curve.png");

    plt::cla();
    plt::title("Accuracy History");
    plt::plot(accuracy_counter, accuracy_history, "r");
    plt::grid(true);
    plt::save("ch4/mnist_accuracy_plot.png");

    return 0;
}