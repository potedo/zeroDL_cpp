#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include "../include/trainer.h"
#include "../include/model.h"
#include "../include/dataset.h"

namespace MyDL{

    using namespace Eigen;
    using std::cout;
    using std::endl;
    using std::shared_ptr;

    Trainer::Trainer(shared_ptr<BaseModel> model, shared_ptr<Dataset> dataset)
    {
        _model = model;
        _dataset = dataset;
    }

    void Trainer::train_step(void)
    {
        // デバッグ用
        cout << "train step began" << endl;

        vector<MatrixXd> input, output;
        MatrixXd train_X, train_y;

        _dataset->next_train(train_X, train_y);

        input.push_back(train_X);

        output = _model->predict(input);

        // デバッグ用
        cout << output[0] << endl;
    }

}