#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <random>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

namespace py = pybind11;
using namespace Eigen;

// Definición de la red Elman RNN
class ElmanRNN {
public:
    MatrixXd W_xh, W_hh, W_hy;
    VectorXd hidden;
    int hidden_size;

    ElmanRNN(int input_size, int hidden_size, int output_size)
        : hidden_size(hidden_size),
          W_xh(MatrixXd::Random(hidden_size, input_size)),
          W_hh(MatrixXd::Random(hidden_size, hidden_size)),
          W_hy(MatrixXd::Random(output_size, hidden_size)),
          hidden(VectorXd::Zero(hidden_size)) {}

    MatrixXd forward(const MatrixXd& input) {
        int n_samples = input.rows();
        MatrixXd output(n_samples, W_hy.rows());
        
        for (int i = 0; i < n_samples; ++i) {
            hidden = (W_xh * input.row(i).transpose() + W_hh * hidden).array().tanh();
            output.row(i) = (W_hy * hidden).transpose();
        }
        return output;
    }

    void train(const MatrixXd& input, const MatrixXd& target, int epochs, double lr, std::vector<double>& loss_values) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            MatrixXd output = forward(input);
            MatrixXd error = target - output;
            double loss = error.squaredNorm() / input.rows();
            loss_values.push_back(loss);

            // Backpropagation (simplificado)
            MatrixXd dW_hy = error.transpose() * hidden.transpose();
            W_hy += lr * dW_hy;
        }
    }
};

std::tuple<py::array_t<double>, std::vector<double>> train_rnn(
    py::array_t<double> input_data, py::array_t<double> target_data,
    int input_size, int hidden_size, int output_size, int epochs, double lr) {

    auto input_buf = input_data.request();
    auto target_buf = target_data.request();
    double* input_ptr = static_cast<double*>(input_buf.ptr);
    double* target_ptr = static_cast<double*>(target_buf.ptr);
    int n_samples = input_buf.shape[0];

    Map<MatrixXd> input_matrix(input_ptr, n_samples, input_size);
    Map<MatrixXd> target_matrix(target_ptr, n_samples, output_size);

    ElmanRNN rnn(input_size, hidden_size, output_size);
    std::vector<double> loss_values;
    rnn.train(input_matrix, target_matrix, epochs, lr, loss_values);
    
    MatrixXd output = rnn.forward(input_matrix);
    py::array_t<double> result({n_samples, output_size}, output.data());

    return std::make_tuple(result, loss_values);
}

PYBIND11_MODULE(elman_rnn, m) {
    m.def("train_rnn", &train_rnn, "Train an Elman RNN using Eigen");
}
