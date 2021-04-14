#ifndef _SAMPLE_NETWORK_H_
#define _SAMPLE_NETWORK_H_

#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "layer.h"

namespace MyDL{

    using namespace Eigen;
    using std::shared_ptr;// unique_ptrも検討 → 可能ならそちらの方がよい。むやみにshared_ptrを使用してはならない
    using std::string;
    using std::vector;
    using std::unordered_map;

    class TwoLayerNet
    {
        private:
            int _input_size;
            int _hidden_size;
            int _output_size;
            double _weight_init_std;
            unordered_map<string, shared_ptr<BaseLayer>> _layers; // BaseLayerのポインタとして各レイヤを保持(アップキャスト)
            vector<string> _layer_list;         // 登録順にキーを保持
            shared_ptr<BaseLayer> _last_layer;  // Lossレイヤ
        public:
            unordered_map<string, shared_ptr<MatrixXd>> params;

        public:
            TwoLayerNet(int input_size=1,
                        int hidden_size=1, 
                        int output_size=1, 
                        double weight_init_std=0.01); // デフォルトコンストラクタ
            vector<MatrixXd> predict(vector<MatrixXd>);
            vector<MatrixXd> loss(vector<MatrixXd>); // 内部的にSoftmaxWithLossのforwardを利用するので返り値もvector型
            double accuracy(vector<MatrixXd>);
            unordered_map<string, MatrixXd> gradient(vector<MatrixXd>);
            unordered_map<string, MatrixXd> numerical_gradient(vector<MatrixXd>);
    };


    class MultiLayerNet
    {
        private:
            int _input_size;
            vector<int> _hidden_size_list;
            int _output_size;
            int _hidden_layer_num;
            double _weight_decay_lambda;
            // unordered_map<string, shared_ptr<BaseLayer>> _layers;
            vector<string> _layer_list;
            shared_ptr<BaseLayer> _last_layer;

        public:
            unordered_map<string, shared_ptr<BaseLayer>> _layers; // デバッグ用にpublicへ移動
            unordered_map<string, shared_ptr<MatrixXd>> params;

        public:
            MultiLayerNet(const int, const vector<int>, const int, const double weight_decay_lambda=0);
            vector<MatrixXd> predict(vector<MatrixXd>);
            vector<MatrixXd> loss(vector<MatrixXd>, MatrixXd&);
            double accuracy(vector<MatrixXd>, MatrixXd&);
            unordered_map<string, MatrixXd> gradient(vector<MatrixXd>, MatrixXd&);
    };

    
    // -----------------------------------------------------------
    // AffineLayer デバッグ用クラス → 参照先の更新確認
    // -----------------------------------------------------------
    class DebugAffine
    {
        private:
            int _input_size;
            int _output_size;
            unordered_map<string, shared_ptr<BaseLayer>> _layers;
        public:
            unordered_map<string, MatrixXd> params;

        public:
            DebugAffine(int input_size=1,
                        int output_size=1,
                        double weight_init_std=0.01);
            void PrintLayerParams(void);       
    };

}



#endif // _SAMPLE_NETWORK_H_