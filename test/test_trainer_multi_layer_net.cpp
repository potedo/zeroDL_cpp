#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../include/trainer.h"
#include "../include/model.h"
#include "../include/optimizer.h"
#include "../datasets/include/mnist.h"

void display_history(MyDL::Trainer& trainer);

int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::shared_ptr;
    using std::make_shared;

    int batch_size = 100;
    int input_size = 28*28;
    vector<int> hidden_size = {50};
    int output_size = 10;
    double lambda = 2.0;

    double learning_rate = 0.1;
    int epochs = 2;

    auto model = make_shared<MultiLayerModel>(input_size, hidden_size, output_size, lambda);
    auto optimizer = make_shared<SGD>(learning_rate);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);

    Trainer trainer(model, optimizer, dataset, epochs=epochs);

    trainer.train();

    display_history(trainer);

    return 0;
}


// util.cppを作成し、その中に記述する？
void display_history(MyDL::Trainer& trainer)
{
    using std::cout;
    using std::endl;

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
}