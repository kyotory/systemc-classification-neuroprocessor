#include "network.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

using WeightMap = std::map<std::size_t, std::vector<float>>;

bool load_weight_file(const std::filesystem::path& path, WeightMap& layers) {
    std::ifstream input(path);
    if (!input) return false;

    std::string raw_line;
    while (std::getline(input, raw_line)) {
        raw_line = trim(raw_line);
        if (raw_line.empty() || raw_line[0] == '#') continue;

        std::istringstream line(raw_line);
        std::string key;
        line >> key;
        constexpr const char* prefix = "weights_layer_";
        if (key.rfind(prefix, 0) != 0) return false;

        std::size_t layer_index = 0;
        try {
            layer_index = static_cast<std::size_t>(std::stoul(key.substr(std::string(prefix).size())));
        } catch (...) {
            return false;
        }

        std::vector<float> values;
        float value = 0.0F;
        while (line >> value) values.push_back(value);
        if (values.empty() || !layers.emplace(layer_index, std::move(values)).second) return false;
    }
    return !layers.empty();
}

bool weights_match_architecture(const WeightMap& map, const NetworkSpec& network) {
    if (map.size() != network.layer_sizes.size()) return false;
    std::size_t fan_in = network.input_size;
    for (std::size_t layer = 0; layer < network.layer_sizes.size(); ++layer) {
        const auto found = map.find(layer);
        if (found == map.end() || found->second.size() != fan_in * network.layer_sizes[layer]) {
            return false;
        }
        fan_in = network.layer_sizes[layer];
    }
    return true;
}

void generate_weights(NetworkSpec& network) {
    std::mt19937 generator(20261013U);
    network.weights.clear();
    network.weights.reserve(network.layer_sizes.size());

    std::size_t fan_in = network.input_size;
    for (const std::size_t fan_out : network.layer_sizes) {
        const float stddev = std::sqrt(2.0F / static_cast<float>(fan_in + fan_out));
        std::normal_distribution<float> distribution(0.0F, stddev);
        std::vector<float> layer_weights(fan_in * fan_out);
        for (float& weight : layer_weights) weight = distribution(generator);
        network.weights.push_back(std::move(layer_weights));
        fan_in = fan_out;
    }
    network.weights_generated = true;
    network.weights_source = "deterministically generated (untrained)";
}
}

NetworkSpec NetworkSpec::load_from_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open network file: " + path);

    NetworkSpec network;
    network.name = std::filesystem::path(path).stem().string();
    std::string raw_line;
    std::size_t line_number = 0;

    while (std::getline(input, raw_line)) {
        ++line_number;
        raw_line = trim(raw_line);
        if (raw_line.empty() || raw_line[0] == '#') continue;

        std::istringstream line(raw_line);
        std::string key;
        line >> key;
        if (key == "name") {
            std::getline(line, network.name);
            network.name = trim(network.name);
            if (network.name.empty()) {
                throw std::runtime_error(path + ":" + std::to_string(line_number) + ": expected a name");
            }
        } else if (key == "image_size") {
            if (!(line >> network.image_width >> network.image_height) ||
                network.image_width == 0 || network.image_height == 0) {
                throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                         ": expected image_size width height");
            }
        } else if (key == "output_labels") {
            std::string label;
            while (line >> label) network.output_labels.push_back(label);
            if (network.output_labels.empty()) {
                throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                         ": output_labels must not be empty");
            }
        } else if (key == "layer_sizes") {
            std::size_t width = 0;
            while (line >> width) network.layer_sizes.push_back(width);
            if (network.layer_sizes.empty()) {
                throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                         ": layer_sizes must contain at least one width");
            }
        } else if (key == "inputs") {
            float value = 0.0F;
            while (line >> value) network.inputs.push_back(value);
        } else {
            throw std::runtime_error(path + ":" + std::to_string(line_number) +
                                     ": unknown setting '" + key + "'");
        }
    }

    network.input_size = network.inputs.size();
    if (network.input_size == 0) throw std::runtime_error(path + ": inputs must contain values");

    std::filesystem::path weight_path(path);
    weight_path.replace_extension(".weights");
    WeightMap loaded;
    if (load_weight_file(weight_path, loaded) && weights_match_architecture(loaded, network)) {
        network.weights.resize(network.layer_sizes.size());
        for (std::size_t layer = 0; layer < network.layer_sizes.size(); ++layer) {
            network.weights[layer] = loaded.at(layer);
        }
        network.weights_generated = false;
        network.weights_source = weight_path.string();
    } else {
        generate_weights(network);
    }

    network.validate();
    return network;
}

void NetworkSpec::validate() const {
    if (input_size == 0 || inputs.size() != input_size) {
        throw std::runtime_error(name + ": no valid input vector");
    }
    if (layer_sizes.empty()) throw std::runtime_error(name + ": network must have at least one layer");
    if (weights.size() != layer_sizes.size()) {
        throw std::runtime_error(name + ": number of weight matrices does not match layer count");
    }
    if ((image_width == 0) != (image_height == 0)) {
        throw std::runtime_error(name + ": image_width and image_height must be specified together");
    }
    if (image_width != 0 && image_width * image_height != input_size) {
        throw std::runtime_error(name + ": image_size does not match number of inputs");
    }

    std::size_t previous_width = input_size;
    for (std::size_t layer = 0; layer < layer_sizes.size(); ++layer) {
        if (layer_sizes[layer] == 0) throw std::runtime_error(name + ": layer widths must be positive");
        const std::size_t expected = previous_width * layer_sizes[layer];
        if (weights[layer].size() != expected) {
            throw std::runtime_error(name + ": weight matrix dimensions do not match layer_sizes");
        }
        previous_width = layer_sizes[layer];
    }
    if (!output_labels.empty() && output_labels.size() != layer_sizes.back()) {
        throw std::runtime_error(name + ": output_labels count must match final layer width");
    }
}

std::size_t NetworkSpec::shared_cells_required() const {
    std::size_t total = input_size;
    for (const auto width : layer_sizes) total += width;
    return total;
}
