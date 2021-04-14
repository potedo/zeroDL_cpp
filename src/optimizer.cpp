#include <string>
#include <unordered_map>
#include <Eigen/Dense>
#include <cmath>
#include "../include/optimizer.h"

namespace MyDL
{

    // ------------------------------------------------------------
    //                  SGD
    // ------------------------------------------------------------
    SGD::SGD(double learning_rate): _learning_rate(learning_rate)
    {
    }

    void SGD::update(unordered_map<string, MatrixXd>& params, unordered_map<string, MatrixXd>& grads)
    {
        for (auto grad : grads)
        {
            params[grad.first] -= _learning_rate * grad.second;
        }
    }

    void SGD::update(unordered_map<string, shared_ptr<MatrixXd>> & params, unordered_map<string, MatrixXd> &grads)
    {
        for (auto grad : grads)
        {
            *(params[grad.first]) -= _learning_rate * grad.second;
        }
    }

    // ------------------------------------------------------------
    //                  Momentum
    // ------------------------------------------------------------
    Momentum::Momentum(double learning_rate, double momentum) : _learning_rate(learning_rate), _momentum(momentum)
    {
    }

    void Momentum::update(unordered_map<string, MatrixXd>& params, unordered_map<string, MatrixXd>& grads)
    {
        if (_v.empty())
        {
            for (auto param : params)
            {
                _v[param.first] = MatrixXd::Zero(param.second.rows(), param.second.cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            _v[key] = _momentum * _v[key] - _learning_rate * grads[key];
            params[key] += _v[key];
        }
    }

    void Momentum::update(unordered_map<string, shared_ptr<MatrixXd>> &params, unordered_map<string, MatrixXd> &grads)
    {
        if (_v.empty())
        {
            for (auto param : params)
            {
                MatrixXd tmp_mat = *(param.second);
                _v[param.first] = MatrixXd::Zero(tmp_mat.rows(), tmp_mat.cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            _v[key] = _momentum * _v[key] - _learning_rate * grads[key];
            *(params[key]) += _v[key];
        }
    }

    // ------------------------------------------------------------
    //                  AdaGrad
    // ------------------------------------------------------------
    AdaGrad::AdaGrad(double learning_rate) : _learning_rate(learning_rate)
    {
    }

    void AdaGrad::update(unordered_map<string, MatrixXd> &params, unordered_map<string, MatrixXd> & grads)
    {
        if (_h.empty())
        {
            for (auto param : params)
            {
                _h[param.first] = MatrixXd::Zero(param.second.rows(), param.second.cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            // 勾配変化の大きかったパラメータの二乗平方根で割る → 要は勾配の絶対値が大きかったパラメータの更新幅を小さくするということ
            // ※ 厳密には勾配の絶対値で割っているわけではない(直前の内部状態に加算して平方根を取っている)
            _h[key].array() += grads[key].array() * grads[key].array();
            params[key].array() -= _learning_rate * _h[key].unaryExpr([](double p){return 1/(sqrt(p)+1e-7);}).array() * grads[key].array();
        }
    }

    void AdaGrad::update(unordered_map<string, shared_ptr<MatrixXd>> &params, unordered_map<string, MatrixXd> &grads)
    {
        if (_h.empty())
        {
            for (auto param : params)
            {
                MatrixXd tmp_mat = *(param.second);
                _h[param.first] = MatrixXd::Zero(tmp_mat.rows(), tmp_mat.cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            // 勾配変化の大きかったパラメータの二乗平方根で割る → 要は勾配の絶対値が大きかったパラメータの更新幅を小さくするということ
            // ※ 厳密には勾配の絶対値で割っているわけではない(直前の内部状態に加算して平方根を取っている)
            _h[key].array() += grads[key].array() * grads[key].array();
            params[key]->array() -= _learning_rate * _h[key].unaryExpr([](double p) { return 1 / (sqrt(p) + 1e-7); }).array() * grads[key].array();
        }
    }

    // ------------------------------------------------------------
    //                  RMSProp
    // ------------------------------------------------------------
    RMSprop::RMSprop(double learning_rate, double decay_rate): _learning_rate(learning_rate), _decay_rate(decay_rate)
    {
    }

    void RMSprop::update(unordered_map<string, MatrixXd>& params, unordered_map<string, MatrixXd>& grads)
    {
        if (_h.empty())
        {
            for (auto param : params)
            {
                _h[param.first] = MatrixXd::Zero(param.second.rows(), param.second.cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            // AdaGradの内部状態hを指数移動平均に置き換えたものがRMSprop -> 指数移動平均の減衰率(平滑化係数)がdecay_rate
            _h[key].array() *= _decay_rate;
            _h[key].array() += (1 - _decay_rate) * grads[key].array() * grads[key].array();
            params[key].array() -= _learning_rate * _h[key].unaryExpr([](double p) { return 1 / (sqrt(p) + 1e-7); }).array() * grads[key].array();
        }
    }

    void RMSprop::update(unordered_map<string, shared_ptr<MatrixXd>> &params, unordered_map<string, MatrixXd> &grads)
    {
        if (_h.empty())
        {
            for (auto param : params)
            {
                _h[param.first] = MatrixXd::Zero(param.second->rows(), param.second->cols());
            }
        }

        for (auto param : params)
        {
            string key = param.first;
            // AdaGradの内部状態hを指数移動平均に置き換えたものがRMSprop -> 指数移動平均の減衰率(平滑化係数)がdecay_rate
            _h[key].array() *= _decay_rate;
            _h[key].array() += (1 - _decay_rate) * grads[key].array() * grads[key].array();
            params[key]->array() -= _learning_rate * _h[key].unaryExpr([](double p) { return 1 / (sqrt(p) + 1e-7); }).array() * grads[key].array();
        }

    }

    // ------------------------------------------------------------
    //                  Adam
    // ------------------------------------------------------------
    Adam::Adam(double learning_rate, double beta1, double beta2): _learning_rate(learning_rate), _beta1(beta1), _beta2(beta2)
    {
        _iter = 0;
    }

    void Adam::update(unordered_map<string, MatrixXd>& params, unordered_map<string, MatrixXd>& grads)
    {
        if (_m.empty())
        {
            for (auto param : params)
            {
                _m[param.first] = MatrixXd::Zero(param.second.rows(), param.second.cols());
                _v[param.first] = MatrixXd::Zero(param.second.rows(), param.second.cols());
            }
        }

        _iter++;
        double lr_t = _learning_rate * sqrt(1.0 - std::pow(_beta2, _iter)) / (1.0 - std::pow(_beta1, _iter));

        for (auto param : params)
        {
            string key = param.first;
            _m[key] += (1 - _beta1) * (grads[key] - _m[key]);
            _v[key] += (1 - _beta2) * (grads[key].unaryExpr([](double p){return p*p;}) - _v[key]);

            params[key].array() -= lr_t * _m[key].array() * _v[key].unaryExpr([](double p){return 1 / (sqrt(p) + 1e-7);}).array();
        }
    }

    void Adam::update(unordered_map<string, shared_ptr<MatrixXd>>& params, unordered_map<string, MatrixXd>& grads)
    {
        if (_m.empty())
        {
            for (auto param : params)
            {
                _m[param.first] = MatrixXd::Zero(param.second->rows(), param.second->cols());
                _v[param.first] = MatrixXd::Zero(param.second->rows(), param.second->cols());
            }
        }

        _iter++;
        double lr_t = _learning_rate * sqrt(1.0 - std::pow(_beta2, _iter)) / (1.0 - std::pow(_beta1, _iter));

        for (auto param : params)
        {
            string key = param.first;
            _m[key] += (1 - _beta1) * (grads[key] - _m[key]);
            _v[key] += (1 - _beta2) * (grads[key].unaryExpr([](double p) { return p * p; }) - _v[key]);

            params[key]->array() -= lr_t * _m[key].array() * _v[key].unaryExpr([](double p) { return 1 / (sqrt(p) + 1e-7); }).array();
        }
    
    }

}