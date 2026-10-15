#include "communication_bus.h"

#include <stdexcept>

CommunicationBus::CommunicationBus(sc_core::sc_module_name name,
                                   const sc_core::sc_time& transaction_latency)
    : sc_module(name), latency_(transaction_latency) {
    if (latency_ <= sc_core::SC_ZERO_TIME) {
        throw std::invalid_argument("Communication transaction latency must be positive");
    }
}

void CommunicationBus::transaction() {
    ++transaction_count_;
    sc_core::wait(latency_);
}

void CommunicationBus::write_shared(SharedMemory& memory,
                                    std::size_t address, float value) {
    transaction();
    memory.write(address, value);
}

float CommunicationBus::read_shared(SharedMemory& memory, std::size_t address) {
    transaction();
    return memory.read(address);
}

void CommunicationBus::dispatch_neuron_to_pe(ProcessingElement& pe) {
    transaction();
    pe.note_dispatch();
}

void CommunicationBus::schedule_neuron_dispatch(ProcessingElement& pe) {
    ++transaction_count_;
    pe.note_dispatch();
}

void CommunicationBus::broadcast_shared_input(
    sc_core::sc_vector<ProcessingElement>& pe_array,
    SharedMemory& memory, std::size_t address) {
    if (pe_array.size() == 0) throw std::runtime_error("Cannot broadcast input without PEs");
    transaction();
    const float value = memory.read(address);
    for (std::size_t i = 0; i < pe_array.size(); ++i) {
        pe_array[i].note_shared_input_read();
        pe_array[i].receive_input(value);
    }
}

void CommunicationBus::write_local_weight(ProcessingElement& pe,
                                          LocalWeightMemory& memory,
                                          std::size_t layer,
                                          std::size_t neuron,
                                          std::size_t index,
                                          float value) {
    transaction();
    memory.write_weight(layer, neuron, index, value);
    pe.note_weight_write();
}

void CommunicationBus::transfer_mac_result_to_ape(ProcessingElement& pe,
                                                   ActivationUnit& ape,
                                                   float weighted_sum) {
    transaction(); 
    pe.note_mac_result_transfer();
    ape.receive_mac_result(weighted_sum);
}

float CommunicationBus::transfer_ape_result_to_cu(ActivationUnit& ape) {
    transaction();
    ape.note_bus_output();
    return ape.output_value();
}

std::size_t CommunicationBus::transaction_count() const { return transaction_count_; }
const sc_core::sc_time& CommunicationBus::transaction_latency() const { return latency_; }
