#ifndef NEURO_ACTIVATION_UNIT_H
#define NEURO_ACTIVATION_UNIT_H

#include <systemc>
#include <cstddef>

SC_MODULE(ActivationUnit) {
    explicit ActivationUnit(sc_core::sc_module_name name);

    void receive_mac_result(float weighted_sum);
    float activate_received();
    float output_value() const;
    void note_bus_output();

    std::size_t activation_count() const;
    std::size_t bus_input_count() const;
    std::size_t bus_output_count() const;

private:
    float pending_weighted_sum_ = 0.0F;
    float output_value_ = 0.0F;
    bool has_pending_value_ = false;
    std::size_t activation_count_ = 0;
    std::size_t bus_input_count_ = 0;
    std::size_t bus_output_count_ = 0;
};

#endif
