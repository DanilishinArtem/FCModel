#pragma once
#include "Model.h"

enum class Activation{
    ReLU,
    Softmax
};

class FFNode : public Node {
public:
    FFNode(Model & model, string name, Activation activation, uint16_t output_size, uint16_t input_size);

    void init(mt19937& rne) override;

    // The input data should have size input_size
    void forward(float* inputs) override;

    // The gradient data should have size output_size
    void reverse(float* gradients) override;

    size_t param_count() const noexcept{
        // Weight matrix entries + bias entries
        return (input_size_ + 1) * output_size_;
    }

    float* param(size_t index);
    float* gradient(size_t index);

    void print() const override;

private:
    Activation activation_;
    uint16_t output_size_;
    uint16_t input_size_;

    // Node parameters ------>
    vector<float> weights_;
    vector<float> biases_;
    vector<float> activations_;

    // Loss gradients ------>
    vector<float> weight_gradients_;
    vector<float> bias_gradients_;
    vector<float> activation_gradients_;

    vector<float> input_gradients_;
    float* last_input_;
};