#ifndef _LAYER_H_
#define _LAYER_H_

#include <Eigen/Dense>
#include <vector>
#include <memory>

namespace MyDL{

    using namespace Eigen;
    using std::vector;
    using std::shared_ptr;

    class Config // Singletonで実装: レイヤ全体の順伝播モードを規定
    {
        private:
            bool _train_flg = true; // 静的メンバ

        private:
            Config(){}; // Singletonパターン(非公開メンバ関数としてコンストラクタ指定)

        public:
            static Config& getInstance(){
                static Config config;
                return config;
            }

            void set_flag(bool flg){
                _train_flg = flg;
            }

            bool get_flag(void){
                return _train_flg;
            }
    };


    class BaseLayer{
        public:
            virtual vector<MatrixXd> forward(vector<MatrixXd>) = 0;
            virtual vector<MatrixXd> backward(vector<MatrixXd>) = 0;
    };

    // SoftmaxWithLossのdout初期値として使用
    const vector<MatrixXd> init_dout = {MatrixXd::Ones(1, 1)};


    class AddLayer: public BaseLayer
    {
        // この層に関しては、微分をそのまま流すので内部に順伝播の情報を保持する必要はない
        private:

        public:
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd>);
    };

    class MulLayer: public BaseLayer
    {
        private:
            MatrixXd _x, _y; // 微分計算に入力の情報が両方必要

        public:
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd>);
    };


    class ReLU: public BaseLayer
    {
        private:
            MatrixXd _mask; // 微分計算にmasking情報が必要
        public:
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd>);
    };


    class Sigmoid: public BaseLayer
    {
        private:
            MatrixXd _y; // 微分計算に順伝播出力の情報が必要
        public:
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd>);
    };


    class Affine: public BaseLayer
    {
        private:
            MatrixXd _X; // 微分計算に順伝播入力の情報が必要

        public:
            shared_ptr<MatrixXd> pW, pb; // 数値微分の際に直接弄る必要がある
            MatrixXd _W, _b; // 内部にコピーを保持 → 動作確認用。これを使うのは非推奨
            MatrixXd dW, db; // 直接これらの値を取得し、gradsという変数に格納する@gradientメソッド

        public:
            Affine(){};
            Affine(const int, const int, const double weight_init_std=0.01); // 内部でEigen Matrixのインスタンスを生成し、パラメータとして保持
            Affine(const shared_ptr<MatrixXd>, const shared_ptr<MatrixXd>); // ポインタを渡す。DNNを構築する場合はこちら推奨
            Affine(MatrixXd&, MatrixXd&); // 参照を渡して、内部でコピーを保持。非推奨。
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd>);
    };


    class SoftmaxWithLoss: public BaseLayer
    {
        private:
            double _loss;
            MatrixXd _Y;
            MatrixXd _t;
        public:
            vector<MatrixXd> forward(vector<MatrixXd>);
            vector<MatrixXd> backward(vector<MatrixXd> dout=init_dout);
    };


    class BatchNorm: public BaseLayer
    {
        private:
            VectorXd _avg_mean;  // 推論時に使用する平均・指数移動平均
            VectorXd _avg_var;   // 推論時に使用する分散・指数移動平均
            double   _momentum;  // 指数移動平均の重み
            MatrixXd _Xc;        // 内部で使用する「平均0化した入力値」
            MatrixXd _Xn;        // 内部で使用する「正規化された入力値」
            VectorXd _std;       // 内部で使用する「入力の標準偏差」
            int      _batch_size;// 内部で使用する「ミニバッチのサイズ」

        public:
            shared_ptr<MatrixXd> pgamma, pbeta; // 横ベクトル → (1, dim)という指定にすること。
            MatrixXd dgamma, dbeta;

        public:
            BatchNorm(){};
            BatchNorm(const int, const double weight_init_std=0.01, const double momentum=0.9);
            BatchNorm(const shared_ptr<MatrixXd>, const shared_ptr<MatrixXd>, double momentum=0.9);
            vector<MatrixXd> forward(vector<MatrixXd>); // 推論用 -> train_flgを別管理にするか、APIをすべて統一するか？
            vector<MatrixXd> backward(vector<MatrixXd>);
    };

}



#endif // _LAYER_H_