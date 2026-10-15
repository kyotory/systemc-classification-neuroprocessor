#ifndef NEURO_MEMORY_H
#define NEURO_MEMORY_H

#include <systemc>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

SC_MODULE(SharedMemory) {
    explicit SharedMemory(sc_core::sc_module_name name, std::size_t capacity);

    void clear();
    void write(std::size_t address, float value);
    float read(std::size_t address) const;
    std::size_t capacity() const;
    std::size_t used_cells() const;
    std::size_t payload_bytes() const;
    std::size_t backing_array_bytes() const;

private:
    std::vector<float> data_;
    std::vector<std::uint8_t> used_;
};

SC_MODULE(ConfigurationMemory) {
    explicit ConfigurationMemory(sc_core::sc_module_name name);

    void clear();
    void write(std::size_t address, std::size_t value);
    std::size_t read(std::size_t address) const;
    std::size_t used_cells() const;
    std::size_t payload_bytes() const;
    std::size_t backing_array_bytes() const;

private:
    std::vector<std::size_t> data_;
    std::vector<std::uint8_t> used_;
};

SC_MODULE(LocalWeightMemory) {
    explicit LocalWeightMemory(sc_core::sc_module_name name);

    void clear();
    void write_weight(std::size_t layer, std::size_t neuron,
                      std::size_t index, float value);
    float read_weight(std::size_t layer, std::size_t neuron,
                      std::size_t index) const;
    std::size_t row_size(std::size_t layer, std::size_t neuron) const;
    std::size_t row_count() const;
    std::size_t stored_weight_count() const;
    std::size_t payload_bytes() const;
    std::size_t value_buffer_bytes() const;

private:
    using Key = std::pair<std::size_t, std::size_t>;
    std::map<Key, std::vector<float>> rows_;
};

#endif
