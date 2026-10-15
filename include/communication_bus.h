#ifndef NEURO_COMMUNICATION_BUS_H
#define NEURO_COMMUNICATION_BUS_H

#include <systemc>
#include <cstddef>

#include "memory.h"
#include "processing_element.h"
#include "activation_unit.h"

SC_MODULE(CommunicationBus) {
    CommunicationBus(sc_core::sc_module_name name,
                     const sc_core::sc_time& transaction_latency);

    void transaction();
    void write_shared(SharedMemory& memory, std::size_t address, float value);
    float read_shared(SharedMemory& memory, std::size_t address);

    void dispatch_neuron_to_pe(ProcessingElement& pe);
    void schedule_neuron_dispatch(ProcessingElement& pe);
    void broadcast_shared_input(sc_core::sc_vector<ProcessingElement>& pe_array,
                                SharedMemory& memory, std::size_t address);
    void write_local_weight(ProcessingElement& pe, LocalWeightMemory& memory,
                            std::size_t layer, std::size_t neuron,
                            std::size_t index, float value);
    void transfer_mac_result_to_ape(ProcessingElement& pe, ActivationUnit& ape,
                                    float weighted_sum);
    float transfer_ape_result_to_cu(ActivationUnit& ape);

    std::size_t transaction_count() const;
    const sc_core::sc_time& transaction_latency() const;

private:
    sc_core::sc_time latency_;
    std::size_t transaction_count_ = 0;
};

#endif
