#ifndef NEURO_PROCESSING_ELEMENT_H
#define NEURO_PROCESSING_ELEMENT_H

#include <systemc>
#include <cstddef>
#include <vector>

#include "memory.h"
#include "point_to_point_link.h"

SC_MODULE(ProcessingElement) {
    explicit ProcessingElement(sc_core::sc_module_name name,
                               const sc_core::sc_time& local_memory_latency);

    PointToPointLink local_memory_link;

    void clear_input_buffer();
    void receive_input(float value);
    std::vector<float> read_local_weights(LocalWeightMemory& local_memory,
                                          std::size_t layer, std::size_t neuron,
                                          std::size_t expected_size);

    float compute_mac(const std::vector<float>& weights);

    void note_dispatch();
    void note_shared_input_read();
    void note_weight_write();
    void note_mac_result_transfer();

    std::size_t neurons_processed() const;
    std::size_t mac_operations() const;
    std::size_t mac_groups() const;
    std::size_t weight_reads() const;
    std::size_t weight_writes() const;
    std::size_t bus_input_reads() const;
    std::size_t dispatched_neurons() const;
    std::size_t mac_result_transfers() const;

private:
    std::vector<float> input_buffer_;
    std::size_t neurons_processed_ = 0;
    std::size_t mac_operations_ = 0;
    std::size_t mac_groups_ = 0;
    std::size_t weight_reads_ = 0;
    std::size_t weight_writes_ = 0;
    std::size_t bus_input_reads_ = 0;
    std::size_t dispatched_neurons_ = 0;
    std::size_t mac_result_transfers_ = 0;
};

#endif
