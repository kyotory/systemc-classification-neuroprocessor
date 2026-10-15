#ifndef NEURO_PROCESSOR_H
#define NEURO_PROCESSOR_H

#include <systemc>
#include <cstddef>
#include <vector>

#include "network.h"
#include "memory.h"
#include "communication_bus.h"
#include "point_to_point_link.h"
#include "processing_element.h"
#include "activation_unit.h"
#include "io_controller.h"
#include "control_unit.h"

SC_MODULE(NeuroProcessor) {
    SC_HAS_PROCESS(NeuroProcessor);

    NeuroProcessor(sc_core::sc_module_name name,
                   std::size_t pe_count,
                   NetworkSpec network);

private:
    void run();
    void print_input_image() const;
    void print_classification_result(const std::vector<float>& output) const;
    void print_measurements(const sc_core::sc_time& model_start,
                           const sc_core::sc_time& after_load,
                           const sc_core::sc_time& after_forward_pass,
                           const sc_core::sc_time& model_end) const;
    void print_core_load() const;
    void print_memory_usage() const;

    std::size_t pe_count_;
    NetworkSpec network_;

    CommunicationBus bus;
    PointToPointLink io_control_link;
    PointToPointLink configuration_link;
    PointToPointLink cu_shared_link;
    SharedMemory shared_memory;
    ConfigurationMemory configuration_memory;
    sc_core::sc_vector<LocalWeightMemory> local_weight_memories;
    sc_core::sc_vector<ProcessingElement> pe_array;
    sc_core::sc_vector<ActivationUnit> ape_array;
    IOController io_controller;
    ControlUnit control_unit;
};

#endif
