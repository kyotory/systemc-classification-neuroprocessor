#ifndef NEURO_NETWORK_H
#define NEURO_NETWORK_H

#include <cstddef>
#include <string>
#include <vector>

struct NetworkSpec {
    std::string name;
    std::size_t input_size = 0;
    std::vector<std::size_t> layer_sizes;
    std::vector<float> inputs;
    std::size_t image_width = 0;
    std::size_t image_height = 0;
    std::vector<std::string> output_labels;
    std::vector<std::vector<float>> weights;
    bool weights_generated = false;
    std::string weights_source;

    bool has_class_labels() const { return !output_labels.empty(); }

    static NetworkSpec load_from_file(const std::string& path);
    void validate() const;
    std::size_t shared_cells_required() const;
};

#endif
