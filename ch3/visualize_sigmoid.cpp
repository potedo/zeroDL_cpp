#include <matplotlibcpp.h>
#include "../simple_lib/include/simple_activation.h"
#include <Eigen/Dense>
#include <vector>

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(){
    using std::vector;

    VectorXd x = VectorXd::LinSpaced(101, -5, 5);
    VectorXd y;

    y = x.unaryExpr([](double p){return MyDL::sigmoid<double>(p);});

    // 変換する要素数で初期化する必要あり
    vector<double> x_vec(101), y_vec(101);

    // 直接EigenのVectorを扱えないので、STLのコンテナに移し替えて使う
    Map<VectorXd>(&x_vec[0], 101) = x;
    Map<VectorXd>(&y_vec[0], 101) = y;

    plt::named_plot("sigmoid", x_vec, y_vec);
    plt::grid(true);
    plt::legend();
    plt::save("./ch3/images/sigmoid.png");

    return 0;
}