#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "../include/trainer.h"
#include "../include/sample_network.h"
#include "../datasets/include/mnist.h"

int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::shared_ptr;
    using std::make_shared;

    int input_size = 28*28;
    int hidden_size = 50;
    int output_size = 10;
    int batch_size = 10;

    auto model = make_shared<TwoLayerMLP>(input_size, hidden_size, output_size);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);

    Trainer trainer(model, dataset);

    trainer.train_step();

    return 0;
}