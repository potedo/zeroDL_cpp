#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <unordered_map>
#include <Eigen/Dense>
#include <matplotlibcpp.h>
#include "../include/model.h"
#include "../datasets/include/mnist.h"
#include "../include/trainer.h"

using std::vector;
namespace plt = matplotlibcpp;

void search_train(double, double, int, vector<double> &, vector<double> &);
void print_vector(vector<double> &);


int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;
    using std::shared_ptr;
    using std::make_shared;
    using std::unordered_map;

    int optimization_trial = 2; //探索回数(テスト用の数値。しっかり確認するなら100回程度やる必要あり)
    // int optimization_trial = 100;
    int trial = 1;

    // 乱数発生のための設定
    std::random_device seed_gen;
    std::default_random_engine engine(seed_gen());
    std::uniform_real_distribution<> lr_dist(-8, -4);
    std::uniform_real_distribution<> lambda_dist(-6, -2);

    // 動作テスト。本来はこれらのパラメータをランダム化してseach_trainに突っ込む
    // 突っ込む内容は、10^(-8) ~ 10^(-4)
    // 10^(-6) ~ 10^(-2)

    unordered_map<string, vector<double>> train_acc_history_dictionary, test_acc_history_dictionary;
    unordered_map<string, double> learning_rate_dictionary, weight_decay_lambda_dictionary;

    while (trial <= optimization_trial )
    {
        vector<double> train_acc_history, test_acc_history;

        double learning_rate = std::pow(10, lr_dist(engine));           // -8 ~ -4
        double weight_decay_lambda = std::pow(10, lambda_dist(engine)); // -6, -2
        int epochs = 5;
        // int epochs = 50;

        cout << "Learning Rate: " << learning_rate << endl;
        cout << "Weight Decay Lambda: " << weight_decay_lambda << endl;

        search_train(learning_rate, weight_decay_lambda, epochs, train_acc_history, test_acc_history);

        cout << "===== Train accuracy =====" << endl;
        print_vector(train_acc_history);
        cout << "===== Test accuracy =====" << endl;
        print_vector(test_acc_history);

        train_acc_history_dictionary["trial" + std::to_string(trial)] = train_acc_history;
        test_acc_history_dictionary["trial" + std::to_string(trial)] = test_acc_history;
        learning_rate_dictionary["trial" + std::to_string(trial)] = learning_rate;
        weight_decay_lambda_dictionary["trial" + std::to_string(trial)] = weight_decay_lambda;

        // train_acc_history, test_acc_historyの中身を消去するコードを作成する
        train_acc_history.clear();
        train_acc_history.shrink_to_fit();
        test_acc_history.clear();
        test_acc_history.shrink_to_fit();

        trial++;
    }

    // 学習結果を描画

    for (auto item : train_acc_history_dictionary)
    {
        string trial_num = item.first;
        string lr_str = std::to_string(learning_rate_dictionary[trial_num]);
        string lambda_str = std::to_string(weight_decay_lambda_dictionary[trial_num]);
        vector<int> x_axis(item.second.size());
        std::iota(x_axis.begin(), x_axis.end(), 1);
        
        plt::title("accuracy history of " + trial_num);
        plt::plot(x_axis, train_acc_history_dictionary[trial_num], "b");
        plt::plot(x_axis, test_acc_history_dictionary[trial_num], "r");
        plt::grid(true);

        plt::save("Hyperparameter Tuning " + trial_num);
        plt::cla();

        cout << "=== " << trial_num << "===" << endl;
        cout << "Learning Rate: " << lr_str << endl;
        cout << "Weight Decay Lambda: " << lambda_str << endl;
        cout << endl;
    }

    return 0;
}


void search_train(double learning_rate,
                  double weight_decay_lambda,
                  int epochs,
                  vector<double> & train_acc_history,
                  vector<double> & test_acc_history)
{
    using namespace Eigen;
    using namespace MyDL;
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using std::make_shared;

    int input_size = 28 * 28;
    vector<int> hidden_list = {50};
    int output_size = 10;
    int batch_size = 100;
    string activation = "relu";
    string weight_initializer = "he";
    bool use_dropout = false;
    double dropout_ratio = 0.5;
    bool use_batchnorm = false;

    auto model = make_shared<MultiLayerModel>(input_size,
                                              hidden_list, 
                                              output_size, 
                                              weight_decay_lambda,
                                              activation,
                                              weight_initializer,
                                              use_dropout,
                                              dropout_ratio,
                                              use_batchnorm);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);
    auto optimizer = make_shared<SGD>(learning_rate);
    // Trainer trainer(model, optimizer, dataset, epochs=epochs, false);
    Trainer trainer(model, optimizer, dataset, epochs=epochs, true);

    trainer.train();

    train_acc_history = trainer.train_acc_history;
    test_acc_history  = trainer.test_acc_history;
}


void print_vector(vector<double> & vec)
{
    for (const auto &item : vec)
    {
        std::cout << item << " ";
    }
    std::cout << std::endl;
}