#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

double cross_entropy_error(MatrixXd&, MatrixXd&); // プロトタイプ宣言
double cross_entropy_error(MatrixXd&, VectorXd&); // プロトタイプ宣言

int main(){
    using std::cout;
    using std::endl;

    MatrixXd y = MatrixXd::Zero(3, 5);
    MatrixXd t = MatrixXd::Zero(3, 5);
    double cross_entropy;

    t(0, 2) = 1;
    t(1, 4) = 1;
    t(2, 0) = 1;

    y << 0.1, 0.4, 0.1, 0.1, 0.1,  0.3, 0.1, 0.1, 0.1, 0.4,  0.3, 0.2, 0.2, 0.1, 0.2;

    cross_entropy = cross_entropy_error(y, t);
    cout << "cross entoropy: " << cross_entropy << endl;

    VectorXd t_label = VectorXd::Zero(3);
    double cross_entropy_label;

    t_label << 2, 4, 0;

    cross_entropy_label = cross_entropy_error(y, t_label);
    cout << "cross entropy label: " << cross_entropy_label << endl;

    return 0;
}


// one-hot labelバージョンの のミニバッチ実装
double cross_entropy_error(MatrixXd& y, MatrixXd& t){
    int batch_size = y.rows();
    double ret = (t.array() * y.array().log()).sum() / batch_size;
    return -ret;
}

// 通常ラベルバージョンのミニバッチ実装
double cross_entropy_error(MatrixXd& y, VectorXd& t){
    int batch_size = y.rows();
    VectorXd associated_label_vector = VectorXd::Zero(batch_size);
    // 使っているEigenのバージョンが3.3.9以上なら、 Fancy Indexが使えるはずだが、どうも違うようなのでこれで対応
    for(int i=0; i<batch_size; i++){
        associated_label_vector(i) = y(i, t(i));
    }
    double ret = associated_label_vector.array().log().sum() / batch_size;
    return -ret;
}