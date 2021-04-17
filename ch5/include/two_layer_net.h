#ifndef _TWO_LAYER_NET_H_
#define _TWO_LAYER_NET_H_

#include <map>
#include <string>
#include <Eigen/Dense>

namespace MyDL{

    using namespace Eigen;
    using std::string;
    using std::map;

    class TwoLayerNet{
        private:
            int _input_size;
            int _hidden_size;
            int _output_size;
            double _weight_init_std;
            map<string, MatrixXd> _cache; // 逆伝播に使用するための計算結果キャッシュ

        public:
            map<string, MatrixXd> params; // MLPのパラメータ(最適化するときに取り出すのでpublic変数に)

        public:
            TwoLayerNet(); // デフォルトコンストラクタ
            TwoLayerNet(int, int, int, double); // 引数付きコンストラクタ(入力層と隠れ層のunit数を指定)
            MatrixXd predict(MatrixXd&); // 推論処理
            double loss(MatrixXd&, MatrixXd&); // 損失関数
            double accuracy(MatrixXd&, MatrixXd&); // 精度
            map<string, MatrixXd> numerical_gradient(MatrixXd&, MatrixXd&); // 勾配計算(数値微分：非推奨)
            map<string, MatrixXd> gradient(MatrixXd&, MatrixXd&); // 勾配計算(計算グラフ → 誤差逆伝播法)
    };

}
#endif // _TWO_LAYER_NET_H_