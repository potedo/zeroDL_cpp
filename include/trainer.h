#ifndef _TRAINER_H_
#define _TRAINER_H_

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "model.h"
#include "dataset.h"

namespace MyDL{

    using namespace Eigen;
    using std::vector;
    using std::shared_ptr;

    class Trainer
    {
        private:
            // NetWorkを引数にとるので、Modelクラスが必要になる
            shared_ptr<BaseModel> _model;
            shared_ptr<Dataset> _dataset;
            bool _one_hot_label;
            bool _normalize;
        
        public:
            Trainer(shared_ptr<BaseModel>, shared_ptr<Dataset>);
            void train_step(void);

    };


}


#endif //_TRAINER_H_