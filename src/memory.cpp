#include "memory.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

SharedMemory::SharedMemory(sc_core::sc_module_name name, std::size_t capacity)
    : sc_module(name), data_(capacity, 0.0F), used_(capacity, 0) {
    if (capacity == 0) throw std::invalid_argument("SharedMemory capacity must be positive");
}

void SharedMemory::clear() {
    std::fill(data_.begin(), data_.end(), 0.0F);
    std::fill(used_.begin(), used_.end(), 0);
}

void SharedMemory::write(std::size_t address, float value) {
    if (address >= data_.size()) throw std::out_of_range("SharedMemory write address out of range");
    data_[address] = value;
    used_[address] = 1;
}

float SharedMemory::read(std::size_t address) const {
    if (address >= data_.size() || !used_[address]) {
        throw std::out_of_range("SharedMemory read from an unused/out-of-range address");
    }
    return data_[address];
}


std::size_t SharedMemory::capacity() const { return data_.size(); }

std::size_t SharedMemory::used_cells() const {
    return static_cast<std::size_t>(std::count(used_.begin(), used_.end(), std::uint8_t{1}));
}

std::size_t SharedMemory::payload_bytes() const { return used_cells() * sizeof(float); }

std::size_t SharedMemory::backing_array_bytes() const {
    return data_.capacity() * sizeof(float) + used_.capacity() * sizeof(std::uint8_t);
}

ConfigurationMemory::ConfigurationMemory(sc_core::sc_module_name name) : sc_module(name) {}

void ConfigurationMemory::clear() {
    data_.clear();
    used_.clear();
}

void ConfigurationMemory::write(std::size_t address, std::size_t value) {
    if (address >= data_.size()) {
        data_.resize(address + 1, 0);
        used_.resize(address + 1, 0);
    }
    data_[address] = value;
    used_[address] = 1;
}

std::size_t ConfigurationMemory::read(std::size_t address) const {
    if (address >= data_.size() || !used_[address]) {
        throw std::out_of_range("ConfigurationMemory read from an unused/out-of-range address");
    }
    return data_[address];
}

std::size_t ConfigurationMemory::used_cells() const {
    return static_cast<std::size_t>(std::count(used_.begin(), used_.end(), std::uint8_t{1}));
}

std::size_t ConfigurationMemory::payload_bytes() const {
    return used_cells() * sizeof(std::size_t);
}

std::size_t ConfigurationMemory::backing_array_bytes() const {
    return data_.capacity() * sizeof(std::size_t) + used_.capacity() * sizeof(std::uint8_t);
}


LocalWeightMemory::LocalWeightMemory(sc_core::sc_module_name name) : sc_module(name) {}

void LocalWeightMemory::clear() { rows_.clear(); }

void LocalWeightMemory::write_weight(std::size_t layer, std::size_t neuron,
                                     std::size_t index, float value) {
    auto& row = rows_[Key{layer, neuron}];
    if (index >= row.size()) row.resize(index + 1, 0.0F);
    row[index] = value;
}

float LocalWeightMemory::read_weight(std::size_t layer, std::size_t neuron,
                                     std::size_t index) const {
    const auto found = rows_.find(Key{layer, neuron});
    if (found == rows_.end() || index >= found->second.size()) {
        throw std::out_of_range("LocalWeightMemory read from an absent weight");
    }
    return found->second[index];
}

std::size_t LocalWeightMemory::row_size(std::size_t layer, std::size_t neuron) const {
    const auto found = rows_.find(Key{layer, neuron});
    if (found == rows_.end()) return 0;
    return found->second.size();
}

std::size_t LocalWeightMemory::row_count() const { return rows_.size(); }

std::size_t LocalWeightMemory::stored_weight_count() const {
    std::size_t total = 0;
    for (const auto& row : rows_) total += row.second.size();
    return total;
}

std::size_t LocalWeightMemory::payload_bytes() const {
    return stored_weight_count() * sizeof(float);
}

std::size_t LocalWeightMemory::value_buffer_bytes() const {
    std::size_t total = 0;
    for (const auto& row : rows_) total += row.second.capacity() * sizeof(float);
    return total;
}

