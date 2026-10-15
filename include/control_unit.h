#ifndef NEURO_CONTROL_UNIT_H
#define NEURO_CONTROL_UNIT_H

#include <systemc>
#include <cstddef>
#include <vector>

#include "network.h"
#include "memory.h"
#include "communication_bus.h"
#include "point_to_point_link.h"
#include "processing_element.h"
#include "activation_unit.h"

struct ExecutionResult {
    std::size_t output_base = 0;
    std::size_t output_size = 0;
};

SC_MODULE(ControlUnit) {
    explicit ControlUnit(sc_core::sc_module_name name);

    void load_configuration(const NetworkSpec& network,
                            ConfigurationMemory& configuration_memory,
                            PointToPointLink& configuration_link);
    void signal_io_controller(PointToPointLink& io_control_link);

    ExecutionResult execute(CommunicationBus& bus,
                            SharedMemory& shared_memory,
                            ConfigurationMemory& configuration_memory,
                            PointToPointLink& configuration_link,
                            PointToPointLink& cu_shared_link,
                            sc_core::sc_vector<ProcessingElement>& pe_array,
                            sc_core::sc_vector<ActivationUnit>& ape_array,
                            sc_core::sc_vector<LocalWeightMemory>& local_weight_memories);
};

#endif
