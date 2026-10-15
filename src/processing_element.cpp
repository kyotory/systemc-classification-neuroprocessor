#include "processing_element.h"

#include <algorithm>
#include <stdexcept>

ProcessingElement::ProcessingElement(sc_core::sc_module_name name,
                                     const sc_core::sc_time& local_memory_latency)
    : sc_module(name),
      local_memory_link("local_memory_link", local_memory_latency) {}

void ProcessingElement::clear_input_buffer() { input_buffer_.clear(); }

void ProcessingElement::receive_input(float value) { input_buffer_.push_back(value); }

std::vector<float> ProcessingElement::read_local_weights(
    LocalWeightMemory& local_memory, std::size_t layer, std::size_t neuron,
    std::size_t expected_size) {
    std::vector<float> values;
    values.reserve(expected_size);
    local_memory_link.record_transactions(expected_size);
    for (std::size_t i = 0; i < expected_size; ++i) {
        values.push_back(local_memory.read_weight(layer, neuron, i));
        ++weight_reads_;
    }
    return values;
}

float ProcessingElement::compute_mac(const std::vector<float>& weights) {
    if (input_buffer_.size() != weights.size()) {
        throw std::invalid_argument("PE input buffer and local weights must have the same size");
    }

    constexpr std::size_t MULTIPLIER_COUNT = 4;
    float accumulator = 0.0F;
    ++neurons_processed_;
    mac_operations_ += input_buffer_.size();
    mac_groups_ += (input_buffer_.size() + MULTIPLIER_COUNT - 1) / MULTIPLIER_COUNT;

    for (std::size_t base = 0; base < input_buffer_.size(); base += MULTIPLIER_COUNT) {
        float products[MULTIPLIER_COUNT] = {0.0F, 0.0F, 0.0F, 0.0F};
        const std::size_t count = std::min(MULTIPLIER_COUNT, input_buffer_.size() - base);
        for (std::size_t lane = 0; lane < count; ++lane) {
            products[lane] = input_buffer_[base + lane] * weights[base + lane];
        }
        const float partial_sum = products[0] + products[1] + products[2] + products[3];
        accumulator += partial_sum;
    }
    return accumulator;
}

void ProcessingElement::note_dispatch() { ++dispatched_neurons_; }
void ProcessingElement::note_shared_input_read() { ++bus_input_reads_; }
void ProcessingElement::note_weight_write() { ++weight_writes_; }
void ProcessingElement::note_mac_result_transfer() { ++mac_result_transfers_; }

std::size_t ProcessingElement::neurons_processed() const { return neurons_processed_; }
std::size_t ProcessingElement::mac_operations() const { return mac_operations_; }
std::size_t ProcessingElement::mac_groups() const { return mac_groups_; }
std::size_t ProcessingElement::weight_reads() const { return weight_reads_; }
std::size_t ProcessingElement::weight_writes() const { return weight_writes_; }
std::size_t ProcessingElement::bus_input_reads() const { return bus_input_reads_; }
std::size_t ProcessingElement::dispatched_neurons() const { return dispatched_neurons_; }
std::size_t ProcessingElement::mac_result_transfers() const { return mac_result_transfers_; }
