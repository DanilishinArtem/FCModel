#include "CCELossNode.h"
#include <limits>

CCELossNode::CCELossNode(Model& model, string name, uint16_t input_size, size_t batch_size): Node{model, std::move(name)}, input_size_{input_size}, inv_batch_size_{float{1.0} / static_cast<float>(batch_size)}{
    // When we deliver a gradient bach, we deliver just the loss gradient with respect
    // to any input and the index that was "hot" in the second argument.
    gradients_.resize(input_size_);
}

void CCELossNode::forward(float* data){
    // The cross-entropy categorical loss is defined as -\sum_i(q_i * log(p_i))
    // where p_i is the predicted prebability and q_i is the expected prebability

    // In information theory, by convention, lim_{x approaches 0}{x log(x)} = 0

    float max{0.0};
    size_t max_index;

    loss_ = float{0.0};
    for(size_t i = 0; i != input_size_; i++){
        if(data[i] > max){
            max_index = i;
            max = data[i];
        }

        // Because the target vector is one-hot encoded, most of these terms will be zero,
        // but we leave the full calculation here to be explicit and in the event we want 
        // to compute losses against probability distributions that arent one-hot. In practice,
        // a faster code path should be employed if the targets are knoen to be one-hot distributions.

        // Prevent undefined results when taking the log of zero
        loss_ -= target_[i] * log(std::max(data[i], numeric_limits<float>::epsilon()));

        if(target_[i] != float{0.0}){
            active_ = i;
        }
    }

    if(max_index == active_){
        ++correct_;
    }else{
        ++incorrect_;
    }

    cummulative_loss_ += loss_;

    // Store the data pointer to compute gradients later
    last_input_ = data;
}

void CCELossNode::reverse(float* data){
    // dJ/dq_i = d(-\sum_i(p_i log(q_i)))/dq_i = -1/q_j where j is the index of the correct
    // classification (loss gradient for single sample).

    // Note the normalization factor where we multiply by the inverse batch size. 
    // This ensures that losses cpmputed by the network are similar in scale irrespectie of the batch size.

    for(size_t i = 0; i != input_size_; i++){
        gradients_[i] = -inv_batch_size_ * target_[i] / last_input_[i];
    }

    for(Node* node : antecedents_){
        node->reverse(gradients_.data());
    }
}

void CCELossNode::print() const {
    printf("Avg loss: %f\t%f%% correct\n", avg_loss(), accuracy() * 100.0);
}

float CCELossNode::accuracy() const {
    return static_cast<float>(correct_) / static_cast<float>(correct_ + incorrect_);
}

float CCELossNode::avg_loss() const {
    return cummulative_loss_ / static_cast<float>(correct_ + incorrect_);
}

void CCELossNode::reset_score() {
    cummulative_loss_ = 0.0;
    correct_ = 0;
    incorrect_ = 0;
}