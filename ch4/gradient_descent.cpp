#include <iostream>
#include <vector>

using std::vector;
using std::cout;
using std::endl;

vector<double> numerical_gradient(double (*f)(vector<double>), vector<double>);
vector<double> gradient_descent(double (*f)(vector<double>),
                                vector<double>&,
                                double, 
                                int);

double function2(std::vector<double> x)
{
    return x[0] * x[0] + x[1] * x[1];
}


int main(){
    vector<double> init_x(2);
    vector<double> x(2);
    init_x[0] = -3.0; // push_backではダメ → init_x(2)で、2つ分の領域を確保している場合、push_backでは、3爪の領域に内容が追加されてしまう。
    init_x[1] = 4.0;

    x = gradient_descent(&function2, init_x, 0.1, 100);

    cout << "--- optimized result ---" << endl;
    cout << "x = (";
    for (int i = 0; i < x.size(); i++){
        cout << x[i] << " ";
    }
    cout << ")" << endl;

}



vector<double> gradient_descent(double (*f)(vector<double>),
                                vector<double>& init_x,
                                double lr=0.01,
                                int step_num=100)
{
    vector<double> x(2);
    vector<double> grad;

    // 初期値を一時変数に代入
    for (int i = 0; i < init_x.size(); i++){
        x[i] = init_x[i];
    }

    for (int i = 0; i < x.size(); i++)
    {
        cout << x[i] << endl;
    }

    // メインの更新処理
    for (int i = 0; i < step_num; i++){
        grad = numerical_gradient(f, x);
        
        for (int j = 0; j < grad.size(); j++){
            x[j] -= lr * grad[j];
        }

        // 標準出力に経過を表示
        cout << "step" << i << ": (";
        for (int j = 0; j < grad.size(); j++){
            cout << x[j] << " ";
        }
        cout << ")" << endl;
    }

    return x;
}

vector<double> numerical_gradient(double (*f)(vector<double>), vector<double> x)
{
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

        grad[i] = (f_xh1 - f_xh2) / (2 * h);
        x[i] = tmp_val; // 元の値に戻す
    }

    return grad;
}