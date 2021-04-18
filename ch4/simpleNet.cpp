#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "../simple_lib/include/simple_activation.h"

using namespace Eigen;

double cross_entropy_error(MatrixXd &, MatrixXd &); // プロトタイプ宣言

class simpleNet
{
private:
    MatrixXd W = MatrixXd::Random(2, 3); // 乱数で初期化(本来はXavierの初期値とかにしたい？)

public:
    simpleNet(); // コンストラクタ
    MatrixXd predict(MatrixXd &);
    double loss(MatrixXd &, MatrixXd &);
    MatrixXd gradient(MatrixXd&, MatrixXd&);

    void print_W(void);
};


int main(){
    using std::cout;
    using std::endl;
    
    simpleNet net;
    net.print_W();

    MatrixXd X = MatrixXd::Zero(2, 1);
    MatrixXd P;

    X << 0.6, 0.9;

    P = net.predict(X);
    cout << "--- predict result ---" << endl;
    cout << P << endl;
    cout << "--- softmax ---" << endl;
    cout << MyDL::softmax(P) << endl;

    MatrixXd t = MatrixXd::Zero(1, 3);
    t << 0, 0, 1; // この書き方をするには、上記のようにメモリ確保をしておく必要がある(でないとセグフォになる)

    double loss;
    loss = net.loss(X, t);

    cout << "--- loss ---" << endl;
    cout << loss << endl;

    MatrixXd dW = MatrixXd::Zero(2, 3);

    dW = net.gradient(X, t);

    cout << "--- dW ---" << endl;
    cout << dW << endl; // 0.2, 0.2, -0.4; 0.3, 0.3, -0.6 (Wをすべて1で初期化した場合)が正解

    return 0;
}

simpleNet::simpleNet(){
    // W << 0.47355232, 0.9977393, 0.84668094, 0.85557411, 0.03563661, 0.69422093;
    W << 1,1,1,1,1,1;
}

MatrixXd simpleNet::predict(MatrixXd& X){
    return X.transpose() * W;
}

double simpleNet::loss(MatrixXd& X, MatrixXd& t){
    MatrixXd Y, Z;
    Z = simpleNet::predict(X);
    Y = MyDL::softmax(Z);

    double loss;
    loss = cross_entropy_error(Y, t);

    return loss;
}

void simpleNet::print_W(void){
    using std::cout;
    using std::endl;

    cout << " --- simpleNet parameter W --- " << endl;
    cout << "W = " << W << endl;
}

MatrixXd simpleNet::gradient(MatrixXd& x, MatrixXd& t)
{
    double h = 1e-4;
    MatrixXd grad = MatrixXd::Zero(2, 3);

    for (int i = 0; i < 6; i++)
    {
        double tmp_val = W(i);
        double f_xh1;
        double f_xh2;
        // f(x+h) の計算
        W(i) = tmp_val + h; // 変数xのi番目の要素だけ増分を取った形に変更
        f_xh1 = simpleNet::loss(x, t);
        // f(x-h)の計算
        W(i) = tmp_val - h;
        f_xh2 = simpleNet::loss(x, t);

        grad(i) = (f_xh1 - f_xh2) / (2 * h);
        W(i) = tmp_val; // 元の値に戻す
    }

    return grad;
}

// one-hot labelバージョンの のミニバッチ実装
double cross_entropy_error(MatrixXd &y, MatrixXd &t)
{
    int batch_size = y.rows();
    double ret = (t.array() * y.array().log()).sum() / batch_size;
    return -ret;
}
