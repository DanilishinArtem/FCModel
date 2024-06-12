add_executable(
    nn
    main.cpp
    FFNode.cpp
    MNIST.cpp
    Model.cpp
    GDOptimizer.cpp
    CCELossNode.cpp
)

target_compile_features(nn PUBLIC cxx_std_17)