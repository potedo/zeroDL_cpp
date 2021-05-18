#ifndef _DATASET_H_
#define _DATASET_H_

#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;

    // -----------------------------------------
    // Datasetの抽象基底クラス(Trainer実装のために作成)
    // -----------------------------------------
    class Dataset{

        public:
            virtual ~Dataset(){};
            virtual void next_train(MatrixXd &, MatrixXd &) = 0;
            virtual void next_test(MatrixXd &, MatrixXd &) = 0;
            virtual int get_train_size(void) = 0;
            // get_test_size()も実装する
            virtual int get_batch_size(void) = 0;
    };

}

#endif //_DATASET_H_