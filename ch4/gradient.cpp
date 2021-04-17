#include <iostream>
#include <vector>
#include "../matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<double> numerical_gradient(double (*f)(std::vector<double>), std::vector<double>);

double function2(std::vector<double> x)
{
    return x[0] * x[0] + x[1] * x[1];
}

int main(){
    using std::cout;
    using std::endl;
    using std::vector;

    vector<double> x(2);
    vector<double> grad(2);
    x[0] = 3;
    x[1] = 4;

    cout << " --- function2 gradient @ (3, 4) --- " << endl;
    grad = numerical_gradient(&function2, x);

    for (int i=0; i<grad.size(); i++){
        cout << grad[i] << " ";
    }
    cout << endl;

    x[0] = 0;
    x[1] = 2;

    cout << " --- function2 gradient @ (0, 2) --- " << endl;
    grad = numerical_gradient(&function2, x);

    for (int i = 0; i < grad.size(); i++)
    {
        cout << grad[i] << " ";
    }
    cout << endl;

    x[0] = 3;
    x[1] = 0;

    cout << " --- function2 gradient @ (3, 0) --- " << endl;
    grad = numerical_gradient(&function2, x);

    for (int i = 0; i < grad.size(); i++)
    {
        cout << grad[i] << " ";
    }
    cout << endl;


    vector<double> _x, y, u, v; // xの再宣言になってしまうのでxだけ_xにしている
    vector<double> tmp_grad(2);
    vector<double> tmp_x(2);
    for (double i = -2; i <= 2; i+=0.25){
        for(double j = -2; j <= 2; j+=0.25){
            _x.push_back(i);
            y.push_back(j);
            tmp_x[0] = i;
            tmp_x[1] = j;

            tmp_grad = numerical_gradient(&function2, tmp_x);

            u.push_back(-tmp_grad[0]);
            v.push_back(-tmp_grad[1]);
        }
    }

    plt::quiver(_x, y, u, v);
    plt::save("sample_gradient.png");

    return 0;
}



std::vector<double> numerical_gradient(double (*f)(std::vector<double>), std::vector<double> x)
{
    using std::vector;
    
    double h = 1e-4;
    int size_x = x.size();
    vector<double> grad(size_x);

    for (int i = 0; i < size_x; i++)
    {
        double tmp_val = x[i];
        double f_xh1;
        double f_xh2;
        // f(x+h) の計算
        x[i] = tmp_val + h; // 変数xのi番目の要素だけ増分を取った形に変更
        f_xh1 = f(x);
        // f(x-h)の計算
        x[i] = tmp_val - h;
        f_xh2 = f(x);

        grad[i] = (f_xh1 - f_xh2) / (2*h);
        x[i] = tmp_val; // 元の値に戻す        
    }

    return grad;
}