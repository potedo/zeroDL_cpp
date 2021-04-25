#ifndef _TRAINER_H_
#define _TRAINER_H_

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "model.h"
#include "dataset.h"
#include "optimizer.h"

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
            shared_ptr<Optimizer> _optimizer;
            bool _one_hot_label;
            bool _normalize;

            int _epochs;
            bool _verbose;
            int _train_size;
            int _batch_size;
            int _max_iter;
            int _iter_per_epoch;
            int _current_iter = 0;
            int _current_epoch = 0;

        public:
            vector<double> train_loss_history, train_acc_history, test_acc_history;

        public:
            Trainer(shared_ptr<BaseModel>, shared_ptr<Optimizer>, shared_ptr<Dataset>, int epochs=10, bool verbose=true);
            void train_step(void); // 後々privateメソッドにしても良いかも(これを直接は使わない)
            void train(void);
            // historyを取得する関数も追加する
    };


}


#endif //_TRAINER_H_