#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include <memory>
#include <fstream>

using namespace std;

// To be defined later. This class encapsulates all the nodes in our graph
// TODO implement this in the cpp file
class Model;

class Node{
public:
    Node(Model& model, string name);

    // Nodes must describe how they should be initialized
    virtual void init(mt19937& rne) = 0;

    // During forward propagation, nodes transform input data and feed results to all subsequent nodes
    virtual void forward(float* inputs) = 0;

    // During reverse propagation, nodes receive loss gradients to its previous outputs and compute gradients with respect to each tunable parameter
    virtual void reverse(float* gradients) = 0; 

    // If the node has tunable parameters, this method should be overridden to reflect the quantity of tunable parameters
    virtual size_t param_count() const noexcept {return 0;}

    // Accessor for parameter by index
    virtual float* param(size_t index) {return nullptr;}

    // Access for loss-gradient with respect tot a parameter specified by index
    virtual float* gradient(size_t index) {return nullptr;}

    // Human-readable name for debugging purposes
    string const& name() const noexcept {return name_;}

    // Information dump for debugging purposes
    virtual void print() const = 0;

protected:
    friend class Model;

    Model& model_;
    string name_;
    // Nodes that precede this node in the computational graph
    vector<Node*> antecedents_; // предыдущие узлы
    // Nodes that succeed this node in the compuational graph
    vector<Node*> subsequents_; // следующие узлы
};

// Base class of optimizer used to train a model
class Optimizer{
public:
    virtual void train(Node& node) = 0;
};

class Model{
public:
    Model(string name);

    // Add a node to the model, forwarding arguments to the node's constructor
    template<typename Node_t, typename... T>
    Node_t& add_node(T&&... args){
        // emplace_back() instead of push_back() because we can initialize the node inplace
        // anothe words, emplace_back creates the object at the end of the vector 
        nodes_.emplace_back(make_unique<Node_t>(*this, forward<T>(args)...));
        // nodes_.emplace_back(make_unique<Node_t>(args));
        return reinterpret_cast<Node_t&>(*nodes_.back());
    }

    // Create a dependency berween two constituent nodes
    void create_edge(Node& dst, Node& src);

    // Initialize the parameters of all nodes with the provided seed. If the 
    // seed os 0, a new random seed is chosen instead. Returns the seed used.
    mt19937::result_type init(mt19937::result_type seed = 0);

    // Adjust all model parameters of constituent nodes using the provided optimizer (shown later)
    void train(Optimizer& optimizer);

    string const& name() const noexcept{
        return name_;
    }

    void print() const;

    // Routines for saving and loading model parameters to and from disk
    void save(ofstream& out);
    void load(ifstream& in);

private:
    friend class Node;
    string name_;
    vector<unique_ptr<Node>> nodes_;
};