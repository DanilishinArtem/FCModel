#include "GDOptimizer.h"
#include "Model.h"
#include <cmath>

GDOptimizer::GDOptimizer(float eta): eta_{eta}{}

void GDOptimizer::train(Node& node){
    size_t param_count = node.param_count();
    for(size_t i = 0; i != param_count; i++){
        float& params = *node.param(i);
        float& gradient = *node.gradient(i);
        params -= eta_ * gradient;
        // Rest the gradient which will be accumulated again in the next training epoch
        gradient = float{0.0};
    }
}