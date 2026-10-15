#include "io_controller.h"

#include <stdexcept>

IOController::IOController(sc_core::sc_module_name name) : sc_module(name) {}

void IOController::load_network(const NetworkSpec& network,
                                CommunicationBus& bus,
                                SharedMemory& shared_memory,
                                sc_core::sc_vector<ProcessingElement>& pe_array,
                                sc_core::sc_vector<LocalWeightMemory>& local_weight_memories) {
    if (pe_array.size() == 0) throw std::runtime_error("Cannot load network without PE elements");

    shared_memory.clear();
    if (local_weight_memories.size() != pe_array.size()) {
        throw std::runtime_error("Each PE must have a separate local weight memory");
    }
    for (std::size_t i = 0; i < local_weight_memories.size(); ++i) {
        local_weight_memories[i].clear();
    }

    for (std::size_t i = 0; i < network.inputs.size(); ++i) {
        bus.write_shared(shared_memory, i, network.inputs[i]);
    }

    std::size_t previous_width = network.input_size;
    for (std::size_t layer = 0; layer < network.layer_sizes.size(); ++layer) {
        const std::size_t output_width = network.layer_sizes[layer];
        for (std::size_t neuron = 0; neuron < output_width; ++neuron) {
            const std::size_t pe_index = neuron % pe_array.size();
            ProcessingElement& pe = pe_array[pe_index];
            LocalWeightMemory& local_memory = local_weight_memories[pe_index];
            const std::size_t row_base = neuron * previous_width;
            for (std::size_t i = 0; i < previous_width; ++i) {
                bus.write_local_weight(pe, local_memory, layer, neuron, i,
                                       network.weights[layer][row_base + i]);
            }
        }
        previous_width = output_width;
    }
}

std::vector<float> IOController::read_output(CommunicationBus& bus,
                                             SharedMemory& shared_memory,
                                             std::size_t output_base,
                                             std::size_t output_size) {
    std::vector<float> output;
    output.reserve(output_size);
    for (std::size_t i = 0; i < output_size; ++i) {
        output.push_back(bus.read_shared(shared_memory, output_base + i));
    }
    return output;
}
