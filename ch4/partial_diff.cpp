#include <iostream>
#include <vector>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

double numerical_diff(double (*f)(double), double x); // fは関数ポインタ, xはその引数
double function2(std::vector<double> x){
    return x[0]*x[0] + x[1]*x[1];
}

double function2_tmp1(double x0); // 偏微分用の関数 → x0に関する偏微分
double function2_tmp2(double x1); // x1に関する偏微分


int main(){
    using std::cout;
    using std::endl;
    using std::vector;

    cout << "--- Partial Differential x0 ---" << endl;
    cout << numerical_diff(&function2_tmp1, 3) << endl;
    cout << "--- Partial Differential x1 ---" << endl;
    cout << numerical_diff(&function2_tmp2, 4) << endl;

    vector<vector<double>> x, y, z;
    for (double i=-5; i<=5; i+=0.25){
        vector<double> x_row, y_row, z_row;
        for (double j=-5; j<=5; j+=0.25){
            vector<double> tmp;
            x_row.push_back(i);
            y_row.push_back(j);
            tmp.push_back(i);
            tmp.push_back(j);
            z_row.push_back(function2(tmp));
        }
        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    plt::plot_surface(x, y, z);
    plt::save("plot_parabola.png");

    return 0;
}

double numerical_diff(double (*f)(double), double x)
{
    double h = 1e-4;
    return (f(x + h) - f(x - h)) / (2 * h);
}

// x0 = x0, x1 = 4の場合の関数。ここからx0に関して微分する
double function2_tmp1(double x0){
    return x0*x0 + 4*4;
}

// x0 = 3, x1 = x1の場合の関数。ここから、x1に関して微分する
double function2_tmp2(double x1){
    return 3*3 + x1 * x1;
}