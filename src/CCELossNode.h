#pragma once
#include "Model.h"

// Categorical cross-entropy loss node
// Assumes input data is "one-hot encoded", with size equal to the number of possible classifications,
// where the "answer" has a single "1" (aka hot value) in one of the classification 
// positions and zero everywhere else.

class CCELossNode : public Node{
public:
    CCELossNode(Model& model, string name, uint16_t input_size, size_t batch_size);

    // No initialization is needed for this node
    void init(mt19937&) override {};

    void forward(float* inputs) override;
    // As a loss node, the argiment to this method is ignored (the gradient of the loss with respect to 
    // itself is unity)
    void reverse(float* gradients = nullptr) override;

    void print() const override;

    void set_target(float const* target){
        target_ = target;
    }

    float accuracy() const;
    float avg_loss() const;
    void reset_score();

private:
    uint16_t input_size_;

    // We minimize the average loss, not the net loss so that the losses 
    // prodused do not scale with batch size (which allows us to keep training parameters constant)

    float inv_batch_size_;
    float loss_;
    float const* target_;
    float* last_input_;
    // Stores the last active classificatin in the target one-hot encoding
    size_t active_;
    float cummulative_loss_{0.0};
    // Store running counts of correct and incorrect predictions
    size_t correct_ = 0;
    size_t incorrect_ = 0;
    vector<float> gradients_;
};