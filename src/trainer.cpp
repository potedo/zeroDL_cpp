#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include "../include/trainer.h"
#include "../include/model.h"
#include "../include/dataset.h"
#include "../include/optimizer.h"

namespace MyDL{

    using namespace Eigen;
    using std::cout;
    using std::endl;
    using std::vector;
    using std::string;
    using std::unordered_map;
    using std::shared_ptr;

    Trainer::Trainer(shared_ptr<BaseModel> model, shared_ptr<Optimizer> optimizer, shared_ptr<Dataset> dataset, int epochs, bool verbose)
    {
        _model = model;
        _optimizer = optimizer;
        _dataset = dataset;
        _epochs = epochs;
        _verbose = verbose;
        _train_size = _dataset->get_train_size();
        _batch_size = _dataset->get_batch_size();
        _iter_per_epoch = (_train_size + _batch_size - 1) / _batch_size; // 切り上げ
        _max_iter = _epochs * _iter_per_epoch;

    }

    void Trainer::train_step(void)
    {
        vector<MatrixXd> inputs, loss, val_inputs;
        unordered_map<string, MatrixXd> grads;
        unordered_map<string, shared_ptr<MatrixXd>> params;
        MatrixXd train_X, train_y, test_X, test_y;

        _dataset->next_train(train_X, train_y);

        inputs.push_back(train_X);

        grads = _model->gradient(inputs, train_y);
        params = _model->get_params();

        _optimizer->update(params, grads);

        loss = _model->loss(inputs, train_y);
        train_loss_history.push_back(double(loss[0](0)));

        if (_verbose)
        {
            // デバッグ用(verboseオプションで表示するようにしても良いかも)
            cout << "Train loss: " << loss[0] << endl;
        }

        if (_current_iter % _iter_per_epoch == 0)
        {
            _current_epoch += 1;

            // accuracyの確認(内部的にエポックごとに確認する機構を設ける)
            _dataset->next_test(test_X, test_y);

            val_inputs.push_back(test_X);

            double train_acc, test_acc;
            train_acc = _model->accuracy(inputs, train_y);
            test_acc = _model->accuracy(val_inputs, test_y);

            train_acc_history.push_back(train_acc);
            test_acc_history.push_back(test_acc);

            if (_verbose)
            {
                cout << "=== epoch: " << _current_epoch << ", train acc: " << train_acc << ", test acc:" << test_acc << "===" << endl;
            }

        }

        _current_iter += 1;

    }

    void Trainer::train(void)
    {
        for (int i=0; i<_max_iter; i++)
        {
            this->train_step();
        }

        MatrixXd test_X, test_y;
        vector<MatrixXd> val_inputs;

        // 本当はすべてのテストデータを突っ込む形にしたい
        // -> ループで実装し、得られたaccuracyの平均を求めればOK?
        _dataset->next_test(test_X, test_y);
        val_inputs.push_back(test_X);

        double test_acc = _model->accuracy(val_inputs, test_y);

        if (_verbose)
        {
            cout << "========== Final Test Accuracy ==========" << endl;
            cout << "test acc: " << test_acc << endl;
        }

    }

}