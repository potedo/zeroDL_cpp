#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "../include/trainer.h"
#include "../include/optimizer.h"
#include "../include/model.h"
#include "../include/utils.h"
#include "../datasets/include/mnist.h"
#include <matplotlibcpp.h>
#include <picojson.h>

int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::string;
    using std::vector;
    using std::iota;
    using std::shared_ptr;
    using std::make_shared;
    namespace plt = matplotlibcpp;

    string filepath = "./test/test_trainer.json";

    picojson::object obj;

    int err;

    err = load_json(filepath, obj);

    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;
    double learning_rate;
    int epochs;

    // リストを読み込む形になっていないのでそこは再度作成する
    if (err)
    {
        input_size = 28 * 28;
        hidden_size = 50;
        output_size = 10;
        batch_size = 100;
        learning_rate = 0.1;
        epochs = 2;
    }
    else
    {
        input_size = obj["input_size"].get<double>();
        hidden_size = obj["hidden_size"].get<double>();
        output_size = obj["output_size"].get<double>();
        batch_size = obj["batch_size"].get<double>();
        learning_rate = obj["learning_rate"].get<double>();
        epochs = obj["epochs"].get<double>();
    }

    cout << "====== Training Parameters ======" << endl;
    cout << "Input Size: " << input_size << endl;
    cout << "Hidden Size: " << hidden_size << endl;
    cout << "Output Size: " << output_size << endl;
    cout << "Batch Size: " << batch_size << endl;
    cout << "Leaning Rate: " << learning_rate << endl;
    cout << "Number of Epochs: " << epochs << endl;

    // model_creator()とか実装して、それに基づいてshared_ptrを返すようにした方が作りやすい？(I/Fを気にしなくて済む)
    // 可変長引数の仕組みとテンプレート関数の仕組みを確認して作成する。一旦無視してやってみる。
    // auto model = make_shared<TwoLayerMLP>(input_size, hidden_size, output_size);
    vector<int> hidden_list; // MultiLayerModelに差し替えるとき必要
    hidden_list.push_back(hidden_size);
    hidden_list.push_back(hidden_size);
    hidden_list.push_back(hidden_size);
    // auto model = make_shared<MultiLayerModel>(input_size, hidden_list, output_size, 0.1, "sigmoid", "sigmoid");
    auto model = make_shared<MultiLayerModel>(input_size,
                                              hidden_list, 
                                              output_size, 
                                              0.01, 
                                              "relu", 
                                              "relu",
                                              true,
                                              0.5,
                                              true);
    auto optimizer = make_shared<SGD>(learning_rate);
    // auto optimizer = make_shared<Adam>(learning_rate*0.01);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);

    Trainer trainer(model, optimizer, dataset, epochs=epochs);

    // cout << *(model->params["b1"]) << endl;

    // // model->get_params()の返り値を利用している場合、ちゃんとモデルのパラメータが更新されるかの確認
    // for (int i=0; i < 100; i++){
    //     trainer.train_step();
    // }

    cout << "======= Training Start! =======" << endl;
    
    trainer.train();

    cout << "Train acc history: "; 
    for (const auto &item : trainer.train_acc_history)
    {
        cout << item << " ";
    }
    cout << endl;

    cout << "Test acc history: ";
    for (const auto &item : trainer.test_acc_history)
    {
        cout << item << " ";
    }
    cout << endl;

    // ------------------------------------------
    //              Visualization
    // ------------------------------------------
    vector<int> loss_counter(trainer.train_loss_history.size());
    iota(loss_counter.begin(), loss_counter.end(), 1);
    plt::title("Loss History");
    plt::plot(loss_counter, trainer.train_loss_history, "b");
    plt::grid(true);
    plt::save("Mnist_Learning_curve_by_Trainer_Class");

    return 0;
}