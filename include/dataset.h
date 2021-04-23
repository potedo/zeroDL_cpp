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
            virtual void next_train(MatrixXd &, MatrixXd &) = 0;
            virtual void next_test(MatrixXd &, MatrixXd &) = 0;

    };

}

#endif //_DATASET_H_