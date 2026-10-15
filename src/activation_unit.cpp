#include "activation_unit.h"

#include <cmath>
#include <stdexcept>

ActivationUnit::ActivationUnit(sc_core::sc_module_name name) : sc_module(name) {}

void ActivationUnit::receive_mac_result(float weighted_sum) {
    pending_weighted_sum_ = weighted_sum;
    has_pending_value_ = true;
    ++bus_input_count_;
}

float ActivationUnit::activate_received() {
    if (!has_pending_value_) {
        throw std::logic_error("APE activation requested before receiving a MAC result");
    }

    const float accumulator = pending_weighted_sum_;
    if (accumulator >= 0.0F) {
        const float z = std::exp(-accumulator);
        output_value_ = 1.0F / (1.0F + z);
    } else {
        const float z = std::exp(accumulator);
        output_value_ = z / (1.0F + z);
    }
    has_pending_value_ = false;
    ++activation_count_;
    return output_value_;
}

float ActivationUnit::output_value() const { return output_value_; }
void ActivationUnit::note_bus_output() { ++bus_output_count_; }
std::size_t ActivationUnit::activation_count() const { return activation_count_; }
std::size_t ActivationUnit::bus_input_count() const { return bus_input_count_; }
std::size_t ActivationUnit::bus_output_count() const { return bus_output_count_; }
