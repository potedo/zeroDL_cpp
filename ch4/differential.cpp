#include <iostream>
#include <vector>
#include "../matplotlibcpp.h"

namespace plt = matplotlibcpp;

double numerical_diff(double (*f)(double), double x); // fは関数ポインタ, xはその引数
double function1(double);

int main(){
    using std::cout;
    using std::endl;
    using std::vector;

    cout << "differential @ x = 5" << endl;
    cout << numerical_diff(&function1, 5) << endl;

    cout << "differential @ x = 10" << endl;
    cout << numerical_diff(&function1, 10) << endl;

    double diff_5 = numerical_diff(&function1, 5);

    int n = 200;
    vector<double> x(n), y(n), z(n);

    for (int i=0; i<n; i++){
        x[i] = (double)0.1 * i;
        y[i] = function1(0.1 * i);
        z[i] = diff_5 * (x[i] - 5) + function1(5); 
    }

    plt::named_plot("function1", x, y, "r");
    plt::named_plot("differential line", x, z, "--b");
    plt::grid(true);
    plt::title("function1");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::legend();
    plt::save("ch4/function1.png");

    return 0;
}

double numerical_diff(double (*f)(double), double x){
    double h = 1e-4;
    return (f(x+h)-f(x-h)) / (2*h);
}

double function1(double x){
    return 0.01*(x*x) + 0.1*x;
}