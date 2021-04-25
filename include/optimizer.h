#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <Eigen/Dense>

namespace MyDL
{
    using namespace Eigen;
    using std::string;
    using std::shared_ptr;
    using std::unordered_map;

    class Optimizer
    {
        public:
            virtual void update(unordered_map<string, shared_ptr<MatrixXd>>&, unordered_map<string, MatrixXd>&)=0;
    };

    class SGD: public Optimizer
    {
    private:
        double _learning_rate;

    public:
        SGD(double learning_rate = 0.1);
        void update(unordered_map<string, MatrixXd>&, unordered_map<string, MatrixXd>&); // 動作確認用
        void update(unordered_map<string, shared_ptr<MatrixXd>>&, unordered_map<string, MatrixXd>&); // DNN用
    };


    class Momentum: public Optimizer
    {
        private:
            double _learning_rate;
            double _momentum;
            unordered_map<string, MatrixXd> _v;

        public:
            Momentum(double learning_rate=0.1, double momentum=0.9);
            void update(unordered_map<string, MatrixXd>&, unordered_map<string, MatrixXd>&);
            void update(unordered_map<string, shared_ptr<MatrixXd>> &, unordered_map<string, MatrixXd> &);
    };


    class AdaGrad: public Optimizer
    {
        private:
            double _learning_rate;
            unordered_map<string, MatrixXd> _h;
        public:
            AdaGrad(double learning_rate);
            void update(unordered_map<string, MatrixXd> &, unordered_map<string, MatrixXd> &);
            void update(unordered_map<string, shared_ptr<MatrixXd>> &, unordered_map<string, MatrixXd> &);
    };


    class RMSprop: public Optimizer
    {
        private:
            double _learning_rate;
            double _decay_rate;
            unordered_map<string, MatrixXd> _h;
        public:
            RMSprop(double learning_rate, double decay_rate);
            void update(unordered_map<string, MatrixXd>&, unordered_map<string, MatrixXd>&);
            void update(unordered_map<string, shared_ptr<MatrixXd>> &, unordered_map<string, MatrixXd> &);
    };


    class Adam: public Optimizer
    {
        private:
            double _learning_rate;
            double _beta1; // Momentumのパラメータ(慣性の指数移動平均)
            double _beta2; // RMSprop のパラメータ(勾配の振幅の移動平均)
            double _iter;
            unordered_map<string, MatrixXd> _m;
            unordered_map<string, MatrixXd> _v;
        public:
            Adam(double learning_rate=0.001, double beta1=0.9, double beta2=0.999);
            void update(unordered_map<string, MatrixXd>&, unordered_map<string, MatrixXd>&);
            void update(unordered_map<string, shared_ptr<MatrixXd>> &, unordered_map<string, MatrixXd> &);
    };

}

#endif // _OPTIMIZER_H_