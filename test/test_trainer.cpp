#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "../include/trainer.h"
#include "../include/optimizer.h"
#include "../include/model.h"
#include "../datasets/include/mnist.h"
#include <matplotlibcpp.h>

int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::vector;
    using std::iota;
    using std::shared_ptr;
    using std::make_shared;
    namespace plt = matplotlibcpp;

    int input_size = 28*28;
    int hidden_size = 50;
    int output_size = 10;
    int batch_size = 100;
    double learning_rate = 0.1;

    int epochs = 2;

    auto model = make_shared<TwoLayerMLP>(input_size, hidden_size, output_size);
    auto optimizer = make_shared<SGD>(learning_rate);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);

    Trainer trainer(model, optimizer, dataset, epochs=epochs);

    // cout << *(model->params["b1"]) << endl;

    // // model->get_params()の返り値を利用している場合、ちゃんとモデルのパラメータが更新されるかの確認
    // for (int i=0; i < 100; i++){
    //     trainer.train_step();
    // }

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