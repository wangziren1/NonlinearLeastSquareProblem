#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <iomanip>
#include "yaml-cpp/yaml.h"

using namespace std;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 5, 5> Matrix5d;

struct Data {
  double t;
  double value;
};

// g(x;t) = x1 + t*x2 + t*t*x3 + x4*exp(-x5*t)
inline double FunctionG(const Vector5d& true_model, double t) {
  double x1 = true_model(0);
  double x2 = true_model(1);
  double x3 = true_model(2);
  double x4 = true_model(3);
  double x5 = true_model(4);
  return x1 + t * x2 + t * t * x3 + x4 * exp(-x5 * t);
}

vector<Data> GenerateData(const Vector5d& true_model) {
  // unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::default_random_engine generator(seed);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, 0.2);
  int data_size = 100;
  vector<Data> data;
  data.reserve(data_size);
  double t = 0;
  for (int i = 0; i < data_size; ++i) {
    double true_data = FunctionG(true_model, t);
    data.push_back({t, true_data + distribution(generator)});
    // data.push_back({t, true_data});
    // cout << distribution(generator) << endl;
    // cout << data.back().t << " " << data.back().value << endl;
    t += 1;
  }
  return data;
}

inline Vector5d ResidualDerivative(const Vector5d& param, double t) {
  double x4 = param(3);
  double x5 = param(4);
  Vector5d residual_derivative;
  residual_derivative(0) = 1;
  residual_derivative(1) = t;
  residual_derivative(2) = t * t;
  residual_derivative(3) = exp(-x5 * t);
  residual_derivative(4) = x4 * exp(-x5 * t) * (-t);
  return residual_derivative;
}

inline double Residual(const Vector5d& param, double t, double y) {
  return FunctionG(param, t) - y;
}

Vector5d Optimize(Vector5d start_point, const vector<Data> data) {
  for (int i = 0; i < 15; ++i) {
    Matrix5d transposeJ_J = Matrix5d::Zero();
    Vector5d transposeJ_residual = Vector5d::Zero();
    double f = 0;
    for (auto& p : data) {
      Vector5d residual_derivative = ResidualDerivative(start_point, p.t);
      transposeJ_J += residual_derivative * residual_derivative.transpose();
      double residual = Residual(start_point, p.t, p.value);
      transposeJ_residual += residual * residual_derivative;
      f += residual * residual;
    }
    Vector5d p_gauss_newton = transposeJ_J.colPivHouseholderQr().
        solve(-transposeJ_residual);
    // Don't use inverse(), because it may produce false result.
    // Vector5d p_gauss_newton = transposeJ_J.inverse() * (-transposeJ_residual);
    f = 0.5 * f;
    cout << i << " f:" << setprecision(6) << f << " step norm:" << 
        p_gauss_newton.norm() << " parameter:" << start_point.transpose() 
        << endl;
    start_point += p_gauss_newton;
  }
  return start_point;
}

void ParseParam(const YAML::Node& model_param, Vector5d& true_model,
    Vector5d& start_point) {
  true_model << model_param["x1"].as<double>(), 
                model_param["x2"].as<double>(),
                model_param["x3"].as<double>(),
                model_param["x4"].as<double>(),
                model_param["x5"].as<double>();
  start_point << model_param["start_x1"].as<double>(),
                 model_param["start_x2"].as<double>(),
                 model_param["start_x3"].as<double>(),
                 model_param["start_x4"].as<double>(),
                 model_param["start_x5"].as<double>();
}

int main(int argc, char** argv) {
  YAML::Node model_param = YAML::LoadFile("../data.yaml");
  Vector5d true_model;
  Vector5d start_point;
  ParseParam(model_param, true_model, start_point);
  vector<Data> data = GenerateData(true_model);
  Vector5d optimized_model = Optimize(start_point, data);
  cout << "true model:     " << true_model.transpose() << endl;
  cout << "optimized model:" << optimized_model.transpose() << endl;
}