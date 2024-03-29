#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "layer.h"

namespace MyDL{

    using namespace Eigen;
    using std::vector;
    using std::string;
    using std::shared_ptr;
    using std::unordered_map;

    // ----------------------------------------------------------------- 
    // Modelに必要なAPIを規定する抽象クラス → Trainerクラスの汎用実装に必要
    // -----------------------------------------------------------------
    
    class BaseModel
    {
        public:

        public:
            virtual vector<MatrixXd> predict(vector<MatrixXd>) = 0;
            virtual vector<MatrixXd> loss(vector<MatrixXd>, MatrixXd&) = 0;
            virtual double accuracy(vector<MatrixXd>, MatrixXd&) = 0;
            virtual unordered_map<string, MatrixXd> gradient(vector<MatrixXd>, MatrixXd&) = 0;
            virtual unordered_map<string, shared_ptr<MatrixXd>> get_params(void) = 0; // Trainerクラスでパラメータを呼び出すのに必要。
    };


    class TwoLayerMLP: public BaseModel
    {
        private:
            int _input_size;
            int _hidden_size;
            int _output_size;
            double _weight_init_std;
            unordered_map<string, shared_ptr<BaseLayer>> _layers; // BaseLayerのポインタとして各レイヤを保持(アップキャスト)
            vector<string> _layer_list;                           // 登録順にキーを保持
            shared_ptr<BaseLayer> _last_layer;                    // Lossレイヤ
        public:
            unordered_map<string, shared_ptr<MatrixXd>> params;

        public:
            TwoLayerMLP(int input_size = 1,
                        int hidden_size = 1,
                        int output_size = 1,
                        double weight_init_std = 0.01); // デフォルトコンストラクタ
            vector<MatrixXd> predict(vector<MatrixXd>);
            vector<MatrixXd> loss(vector<MatrixXd>, MatrixXd&); // 内部的にSoftmaxWithLossのforwardを利用するので返り値もvector型
            double accuracy(vector<MatrixXd>, MatrixXd&);
            unordered_map<string, MatrixXd> gradient(vector<MatrixXd>, MatrixXd&);
            // unordered_map<string, MatrixXd> numerical_gradient(vector<MatrixXd>);
            unordered_map<string, shared_ptr<MatrixXd>> get_params(void){return this->params;}; // Trainerクラスでパラメータを呼び出すのに必要。これを使えば直接model->paramsとかしなくて済むのでは
    };

    class MultiLayerModel: public BaseModel
    {
        private:
            int _input_size;
            vector<int> _hidden_size_list;
            int _output_size;
            vector<int> _all_size_list;
            int _hidden_layer_num;
            double _weight_decay_lambda;
            unordered_map<string, shared_ptr<BaseLayer>> _layers;
            vector<string> _layer_list;
            shared_ptr<BaseLayer> _last_layer;

        public:
            // unordered_map<string, shared_ptr<BaseLayer>> _layers; // デバッグ用にpublicへ移動
            unordered_map<string, shared_ptr<MatrixXd>> params;
        
        private:
            void _init_weight(string); // 引数は weight_initializer

        public:
            MultiLayerModel(const int,
                            const vector<int>, 
                            const int,
                            const double weight_decay_lambda = 0,
                            string activation = "relu",
                            const string weight_initializer="he",
                            const bool use_dropout = false,
                            const double dropout_ratio = 0.5,
                            const bool use_batchnorm = false); // 指定できる内容「relu」「he」「sigmoid」「xavier」)
            vector<MatrixXd> predict(vector<MatrixXd>);
            vector<MatrixXd> loss(vector<MatrixXd>, MatrixXd &);
            double accuracy(vector<MatrixXd>, MatrixXd &);
            unordered_map<string, MatrixXd> gradient(vector<MatrixXd>, MatrixXd &);
            unordered_map<string, shared_ptr<MatrixXd>> get_params(void){return this->params;};
    };

    
    class SimpleConvModel: public BaseModel
    {
        private:
            int _C;
            int _H;
            int _W;
            int _filter_num;
            int _filter_size;
            int _pad;
            int _stride;
            int _hidden_size;
            int _output_size;
            unordered_map<string, shared_ptr<BaseLayer>> _layers;
            vector<string> _layer_list;
            shared_ptr<BaseLayer> _last_layer;

        public:
            unordered_map<string, shared_ptr<MatrixXd>> params;

        public:
            SimpleConvModel(const int, // input_channels
                            const int, // input_height
                            const int, // input_width
                            const int, // filter num
                            const int, // filter size
                            const int, // pad
                            const int, // stride
                            const int, // hidden size
                            const int, // output_size
                            const double); // weight_init_std
            vector<MatrixXd> predict(vector<MatrixXd>);
            vector<MatrixXd> loss(vector<MatrixXd>, MatrixXd &);
            double accuracy(vector<MatrixXd>, MatrixXd &);
            unordered_map<string, MatrixXd> gradient(vector<MatrixXd>, MatrixXd &);
            unordered_map<string, shared_ptr<MatrixXd>> get_params(void) { return this->params; };
    };

}

#endif //_MODEL_H_