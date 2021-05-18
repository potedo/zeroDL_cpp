#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "../include/trainer.h"
#include "../include/model.h"
#include "../include/optimizer.h"
#include "../datasets/include/mnist.h"

int main()
{
    using namespace Eigen;
    using namespace MyDL;
    using std::cout;
    using std::endl;
    using std::make_shared;
    using std::shared_ptr;

    int batch_size = 100;
    int input_channels = 1;
    int input_height = 28;
    int input_width = 28;
    // int filter_num = 30;
    int filter_num = 10;
    int filter_size = 5;
    int pad = 0;
    int stride = 1;
    int hidden_size = 100;
    int output_size = 10;
    double weight_init_std = 0.01;

    double learning_rate = 0.001;
    int epochs = 10;

    auto model = make_shared<SimpleConvModel>(input_channels, 
                                              input_height, 
                                              input_width, 
                                              filter_num, 
                                              filter_size, 
                                              pad,
                                              stride,
                                              hidden_size,
                                              output_size,
                                              weight_init_std);
    auto optimizer = make_shared<Adam>(learning_rate);
    auto dataset = make_shared<MnistEigenDataset>(batch_size);

    Trainer trainer(model, optimizer, dataset, epochs = epochs);

    trainer.train();

    return 0;
}