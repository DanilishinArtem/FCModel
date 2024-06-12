#pragma once

#include "Model.h"
#include <fstream>

class MNIST : public Node
{
public:
    constexpr static size_t DIM = 28 * 28;

    MNIST(Model& model, std::ifstream& images, std::ifstream& labels);

    void init(mt19937&) override
    {}

    // As this is an input node, the argument to this function is ignored
    void forward(float* data = nullptr) override;
    // Backpropagation is a no-op for input nodes as there are no parameters to
    // update
    void reverse(float* data = nullptr) override
    {}

    // Parse the next image and label into memory
    void read_next();

    void print() const override;

    [[nodiscard]] size_t size() const noexcept
    {
        return image_count_;
    }

    [[nodiscard]] float const* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] float* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] float* label() noexcept
    {
        return label_;
    }

    [[nodiscard]] float const* label() const noexcept
    {
        return label_;
    }

    // Quick ASCII visualization of the last read image. For best results,
    // ensure that your terminal font is a monospace font.
    void print_last();

private:
    std::ifstream& images_;
    std::ifstream& labels_;
    uint32_t image_count_;
    // Data from the images file is read as one-byte unsigned values which are
    // converted to num_t after
    char buf_[DIM];
    // All images are resized (with antialiasing) to a 28 x 28 row-major raster
    float data_[DIM];
    // One-hot encoded label
    float label_[10];
};