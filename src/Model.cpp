#include "Model.h"

Node::Node(Model& model, string name) : model_{model}, name_{std::move(name)}{}

Model::Model(string name) : name_{std::move(name)}{}

void Model::create_edge(Node& dst, Node& src){
    // No validation is done to ensure the edge doesn't already exist
    dst.antecedents_.push_back(&src);
    src.subsequents_.push_back(&dst);
}

mt19937::result_type Model::init(mt19937::result_type seed)
{
    if (seed == 0)
    {
        // Generate a new random seed from the host random device
        std::random_device rd{};
        seed = rd();
    }
    // std::printf("Initializing model parameters with seed: %u\n", seed);

    mt19937 rne{seed};

    for (auto& node : nodes_)
    {
        node->init(rne);
    }

    return seed;
}

void Model::train(Optimizer& optimizer){
    for(auto&& node : nodes_){
        optimizer.train(*node);
    }
}

void Model::print() const {
    // Invoke "print" on each node in the order added
    for(auto&& node : nodes_){
        node->print();
    }
}

void Model::save(ofstream& out){
    // To save the model to disk, we emplay a very simple scheme. All nodes are looped through in the order they
    // were added to the model. Then, all advertised learnable parameters are serialized in host byte-order to the 
    // supplied output stream

    // This simplistic method of saving the model to disk isn't very robust or practical in the real world.
    // For one thing, it contains no reflection data about the topology of the model. Loading the data relies
    // on the model being constructed in the same mabber in was trained on.
    // Furthermore, the data will be parsed incorrectly if the program is recompiled to operate with a 
    // different precision. Adopting a more sibseble serialization scheme is left as an exercise.

    for(auto& node : nodes_){
        size_t param_count = node->param_count();
        for(size_t i = 0; i != param_count; i++){
            out.write(reinterpret_cast<char const*>(node->param(i)), sizeof(float));
        }
    }
}

void Model::load(ifstream& in){
    for(auto& node : nodes_){
        size_t param_count = node->param_count();
        for(size_t i = 0; i != param_count; i++){
            in.read(reinterpret_cast<char*>(node->param(i)), sizeof(float));
        }
    }
}