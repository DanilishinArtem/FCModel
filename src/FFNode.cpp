#include "FFNode.h"


FFNode::FFNode(Model& model, 
                string name, 
                Activation activation, 
                uint16_t output_size, 
                uint16_t input_size) 
                : Node{model, std::move(name)}, 
                activation_{activation},
                output_size_{output_size},
                input_size_{input_size}
{
    printf("%s: %d -> %d\n", name_.c_str(), input_size_, output_size_);

    // The weight parameters of a FF-layer are an NxM matrix
    weights_.resize(output_size_ * input_size_);

    // Each node in this layer is assigned a bias (so that zero is not necessarily mapped to zero)
    biases_.resize(output_size_);

    // The outputs of each neuron within the layer is an "activation" in neuroscience parlance
    activations_.resize(output_size_);

    activation_gradients_.resize(output_size_);
    weight_gradients_.resize(output_size_ * input_size_);
    bias_gradients_.resize(output_size_);
    input_gradients_.resize(input_size_);
}



void FFNode::init(mt19937& rne){
    float sigma;
    switch (activation_){
        case Activation::ReLU:
            sigma = sqrt(2.0 / static_cast<float>(input_size_));
            break;
        case Activation::Softmax:
        default:
            sigma = sqrt(1.0 / static_cast<float>(input_size_));
            break;
    }

    auto dist = normal_distribution<float>(0.0, sigma);
    
    for(float& w : weights_){
        w = dist(rne);
    }

    for(float& b : biases_){
        b = 0.01;
    }
}

void FFNode::forward(float* inputs){
    // Remember te last input data for backpropagation later
    last_input_ = inputs;

    for(size_t i = 0; i != output_size_; i++){
        // For each output vector, compute the dot product of the input data with the weight vector add the bias
        float z{0.0};

        size_t offset = i * input_size_;

        for(size_t j = 0; j != input_size_; j++){
            z += weights_[offset + j] * inputs[j];
        }

        // Add neuron bias
        z += biases_[i];

        switch(activation_){
            case Activation::ReLU:
                activations_[i] = max(z, float{0.0});
                break;
            case Activation::Softmax:
            default:
                activations_[i] = exp(z);
                break;
        }
    }

    if (activation_ == Activation::Softmax){
        // softmax(z)_i = exp(z_i) / sum_j exp(z_j)
        float sum_exp_z{0.0};
        for(size_t i = 0; i != output_size_; i++){
            sum_exp_z += activations_[i];
        }
        float inv_sum_exp_z = float{1.0} / sum_exp_z;
        for(size_t i = 0; i != output_size_; i++){
            activations_[i] *= inv_sum_exp_z;
        }
    }

    // Forward activation data to all subsequent nodes in the computational graph
    for(Node* subsequents : subsequents_){
        // pass a pointer to the first element of the vector inputs
        subsequents->forward(activations_.data());
    }
}

void FFNode::reverse(float* gradients){
    // We receive a vector of output_size_ gradients of the loss function with respect to the activations of this node.
    // We need to compute the gradients of the loss function with respect to each parameter in the node (all weights and biases).
    // In addition, we need to compute the gradients with respect to the inputs in order to propagate the gradients further.

    // Noration:
    // Subscripts on any of the following vector and matrix quantities are used to specify a specific element of the vector or matrix.
    // - I is the input vector
    // - W is the weight matrix
    // - B is the bias vector
    // - Z = W * I + B
    // - A is our activation function (ReLU or softmax)
    // - L is the total loss (cost)

    // The gradient we receive from the subsequent is dJ/dg(Z) which we can use to compute dJ/dW_{i,j}, dJ/dB_i, and dJ/dI_i

    // First, we compute dJ/dz as dJ/dg(z) * dg(z)/dz and store it in out activations array
    for(size_t i = 0; i != output_size_; i++){
        // dg(z)/dz
        float activation_grad{0.0};
        switch(activation_){
            case Activation::ReLU:
                if(activations_[i] > float(0.0)){
                    activation_grad = float{0.0};
                }else{
                    activation_grad = float{1.0};
                }
                // dJ/dz = dJ/dg(z) * dg(z)/dz
                activation_gradients_[i] = gradients[i] * activation_grad;
                break;
            case Activation::Softmax:
            default:
                for(size_t j = 0; j != output_size_; j++){
                    if (i == j){
                        activation_grad += activations_[i] * (float{1.0} - activations_[i]) * gradients[j];
                    }else{
                        activation_grad += -activations_[i] * activations_[j] * gradients[j];
                    }
                }
                activation_gradients_[i] = activation_grad;
                break;
        }
    }

    for(size_t i = 0; i != output_size_; i++){
        // Next, let's cumpute the partial dJ/db_i. If we hold all the weights and imputs
        // constant, it's clear that dz/db_i is just 1 (consider differentiating the line
        // mx + b with respect to b). Thus, dJ/db_i = dJ/dg(z_i) * dg(z_i)/dz_i * 1
        bias_gradients_[i] += activation_gradients_[i];
    }

    fill(input_gradients_.begin(), input_gradients_.end(), 0.0);

    // To compute dz/dI_i, recall that z_i = \sum_i W_i * I_i + B_i. That is, the precursor to each activation is
    // a dot-product between a weight vector and the input plus a bias. Thus, dz/dI_i must be the sum of all
    // weights that were scaled by I_i during the forward pass.
    for(size_t i = 0; i != output_size_; i++){
        size_t offset = i * input_size_;
        for(size_t j = 0; j != input_size_; j++){
            input_gradients_[j] += weights_[offset + j] * activation_gradients_[i];
        }
    }

    for(size_t i = 0; i != input_size_; i++){
        for(size_t j = 0; j != output_size_; j++){
            // Each individual weight shows up in the equation for z once and is scaled by the corresponding input.
            // Thus, dJ/dw_i = dJ/dg(z_i) * dg(z_i)/dz_i * dz_i/dw_ij where the last factor is equal to the input scaled by w_ij
            weight_gradients_[j * input_size_ + i] += last_input_[i] * activation_gradients_[j];
        }
    }

    for(Node* node : antecedents_){
        // Forward loss gradients with respect to the inputs to the previous node
        node->reverse(input_gradients_.data());
    }
}

float* FFNode::param(size_t index){
    if(index < weights_.size()){
        return &weights_[index];
    }
    return &biases_[index - weights_.size()];
}

float* FFNode::gradient(size_t index){
    if(index < weights_.size()){
        return &weight_gradients_[index];
    }
    return &bias_gradients_[index - weights_.size()];
}

void FFNode::print() const{
    printf("%s\n", name_.c_str());

    // Consider the input samples as column vectors, and visualize the weights as matrix
    // transforming vectors with input_size dimension to size_ dimension
    printf("Weights (%d X %d)\n", output_size_, input_size_);
    for(size_t i = 0; i != output_size_; i++){
        size_t offset = i * input_size_;
        for(size_t j = 0; j != input_size_; j++){
            printf("\t[%zu]%f", offset + j, weights_[offset + j]);
        }
        printf("\n");
    }
    printf("Biases (%d x 1)\n", output_size_);
    for(size_t i = 0; i != output_size_; i++){
        printf("\t%f\n", biases_[i]);
    }
    printf("\n");
}