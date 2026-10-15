#ifndef NEURO_IO_CONTROLLER_H
#define NEURO_IO_CONTROLLER_H

#include <systemc>
#include <cstddef>
#include <vector>

#include "network.h"
#include "memory.h"
#include "processing_element.h"
#include "communication_bus.h"

SC_MODULE(IOController) {
    explicit IOController(sc_core::sc_module_name name);

    void load_network(const NetworkSpec& network,
                      CommunicationBus& bus,
                      SharedMemory& shared_memory,
                      sc_core::sc_vector<ProcessingElement>& pe_array,
                      sc_core::sc_vector<LocalWeightMemory>& local_weight_memories);

    std::vector<float> read_output(CommunicationBus& bus,
                                  SharedMemory& shared_memory,
                                  std::size_t output_base,
                                  std::size_t output_size);
};

#endif
