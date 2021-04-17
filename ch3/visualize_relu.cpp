#include <vector>
#include <Eigen/Dense>
#include "../ch5/include/simple_activation.h"
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;
using namespace Eigen;

int main(){
    using std::vector;

    int n = 201;
    VectorXd x = VectorXd::LinSpaced(n, -10, 10);
    VectorXd y;

    y = x.unaryExpr([](double p){return MyDL::relu<double>(p);});

    // plotのために、STLコンテナに詰め替え
    vector<double> xs(n), ys(n);
    Map<VectorXd>(&xs[0], n) = x;
    Map<VectorXd>(&ys[0], n) = y;

    plt::named_plot("ReLU", xs, ys, "--b");
    plt::grid(true);
    plt::legend();
    plt::save("ch3/images/relu.png");

    return 0;
}