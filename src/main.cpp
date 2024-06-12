#include "CCELossNode.h"
#include "FFNode.h"
#include "GDOptimizer.h"
#include "MNIST.h"
#include "Model.h"
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>

static constexpr size_t batch_size = 80;

Model create_model(ifstream& images, ifstream& labels, MNIST** mnist, CCELossNode** loss){
    
    // Here we create a simple fully-cobbected feedforwrd neural network
    Model model{"ff"};

    *mnist = &model.add_node<MNIST>(images, labels);

    FFNode& hidden = model.add_node<FFNode>("hidden", Activation::ReLU, 32, 784);

    FFNode& output = model.add_node<FFNode>("output", Activation::Softmax, 10, 32);

    *loss = &model.add_node<CCELossNode>("loss", 10, batch_size);
    (*loss)->set_target((*mnist)->label());

    // The structure of our compurational graph is completely sequential. In fact, the fully connected node
    // and loss node we'he implemented here do not support multiple inputs. Consider adding nodes that
    // support "skip" connections that forward outputs from earlier nodes to downstream nodes that aren't
    // directly adjacent (such skip nodes are used in the ResNet architecture)

    model.create_edge(hidden, **mnist);
    model.create_edge(output, hidden);
    model.create_edge(**loss, output);
    return model;
}

void train(char* argv[]){
    // Uncomment tot debug floating point instability in the network
    // feenableexcept(FE_INVALID | FE_OVERFLOW);

    printf("Executing training routine\n");

    ifstream images{
        std::filesystem::path{argv[0]} / "train-images-idx3-ubyte",
        std::ios::binary
    };

    ifstream labels{
        std::filesystem::path{argv[0]} / "train-labels-idx1-ubyte",
        std::ios::binary
    };

    MNIST* mnist;
    CCELossNode* loss;
    Model model = create_model(images, labels, &mnist, &loss);

    model.init();

    // The gradient descent optimizer is stateless, but other optimizers may not be.
    // Some optimizers need to track "momentum" or gradient histories.
    // Others may slow the learning rate for each parameter at different rates
    // depending on various factors

    GDOptimizer optimizer{float{0.3}};

    // Here, hardcoded the number of batched to train on. In practice, training should
    // halt when the average loss begins to vascillate, indicating that the model is starting 
    // to overfit the data. Implement some form of loss-improvement measure to determine when 
    // this inflection point occurs and stop accordingly.

    size_t i = 0;
    for(i=0; i!=256; ++i){
        loss->reset_score();
        for(size_t j = 0; j != batch_size; ++j){
            mnist->forward();
            loss->reverse();
        }
        model.train(optimizer);
    }

    printf("Run %zu batches (%zu samples each)\n", i, batch_size);

    // Pring the average loss computed in the final batch
    loss->print();

    ofstream out{
        std::filesystem::current_path() / (model.name() + ".params"),
        std::ios::binary
    };
    model.save(out);
}

void evaluate(char* argv[]){
    printf("Executing evaluatin routine\n");

    ifstream images{
        std::filesystem::path{argv[0]} / "t10k-images-idx3-ubyte",
        std::ios::binary
    };

    ifstream labels{
        std::filesystem::path{argv[0]} / "t10k-labels-idx1-ubyte",
        std::ios::binary
    };

    MNIST* mnist;
    CCELossNode* loss;
    // For the data to be loaded properly, the model myst be constructed in the same manner
    // as it was constructed during training.

    Model model = create_model(images, labels, &mnist, &loss);

    // Instead of initializing the parameters randompy, here we load it from 
    // disk (saved from a previous training run)
    std::ifstream params_file{std::filesystem::path{argv[1]}, std::ios::binary};
    model.load(params_file);

    // Evaluate all 10 000 images in the test set and compute the loss average
    for(size_t i = 0; i != mnist->size(); i++){
        mnist->forward();
    }
    loss->print();
}

int main(int argc, char* argv[]){
    if(argc < 2){
        printf("Supported commands include:\ntrain\nevaluate\n");
        return 1;
    }

    if(strcmp(argv[1], "train") == 0){
        train(argv + 2);
    }else if(strcmp(argv[1], "evaluate") == 0){
        evaluate(argv + 2);
    }else{
        printf("Argument %s is an unrecognized directive.\n", argv[1]);
    }
    return 0;
}


