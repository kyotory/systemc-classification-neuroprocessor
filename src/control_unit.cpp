#include "control_unit.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

ControlUnit::ControlUnit(sc_core::sc_module_name name) : sc_module(name) {}

void ControlUnit::load_configuration(const NetworkSpec& network,
                                     ConfigurationMemory& configuration_memory,
                                     PointToPointLink& configuration_link) {
    configuration_memory.clear();
    configuration_link.transaction();
    configuration_memory.write(0, network.input_size);
    configuration_link.transaction();
    configuration_memory.write(1, network.layer_sizes.size());
    for (std::size_t layer = 0; layer < network.layer_sizes.size(); ++layer) {
        configuration_link.transaction();
        configuration_memory.write(2 + layer, network.layer_sizes[layer]);
    }
}

void ControlUnit::signal_io_controller(PointToPointLink& io_control_link) {
    io_control_link.transaction();
}

/* прочитать конфигурацию

для каждого слоя:
    загрузить входной вектор слоя в PE
    назначить нейроны вычислительным элементам
    учесть время диспетчеризации и чтения весов
    рассчитать взвешенные суммы
    передать суммы в APE
    применить функцию активации
    вернуть результаты в CU
    записать результаты в Shared Memory

вернуть адрес начала и размер выходного слоя*/

ExecutionResult ControlUnit::execute(
    CommunicationBus& bus,
    SharedMemory& shared_memory,
    ConfigurationMemory& configuration_memory,
    PointToPointLink& configuration_link,
    PointToPointLink& cu_shared_link,
    sc_core::sc_vector<ProcessingElement>& pe_array,
    sc_core::sc_vector<ActivationUnit>& ape_array,
    sc_core::sc_vector<LocalWeightMemory>& local_weight_memories) {
    if (pe_array.size() == 0 || ape_array.size() == 0 ||
        local_weight_memories.size() != pe_array.size()) {
        throw std::runtime_error("ControlUnit requires matching PE, APE and local-memory arrays");
    }

    configuration_link.transaction();
    const std::size_t input_width = configuration_memory.read(0);
    configuration_link.transaction();
    const std::size_t layer_count = configuration_memory.read(1);
    if (layer_count == 0) throw std::runtime_error("Configuration contains no layers");

    std::vector<std::size_t> layer_sizes(layer_count);
    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        configuration_link.transaction();
        layer_sizes[layer] = configuration_memory.read(2 + layer);
        if (layer_sizes[layer] == 0) throw std::runtime_error("Layer width must be positive");
    }

    struct NeuronTask {
        std::size_t neuron = 0;
        std::size_t pe_index = 0;
        std::size_t ape_index = 0;
        std::vector<float> weights;
    };

    std::size_t source_base = 0;
    std::size_t destination_base = input_width;
    std::size_t current_width = input_width;

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        const std::size_t output_width = layer_sizes[layer];

        for (std::size_t pe = 0; pe < pe_array.size(); ++pe) {
            pe_array[pe].clear_input_buffer();
        }
        for (std::size_t i = 0; i < current_width; ++i) {
            bus.broadcast_shared_input(pe_array, shared_memory, source_base + i);
        }
        
        std::vector<NeuronTask> tasks;
        tasks.reserve(output_width);
        std::vector<std::size_t> pe_ready_cycle(pe_array.size(), 0);
        std::size_t bus_ready_cycle = 0;
        std::size_t layer_makespan = 0;

        for (std::size_t neuron = 0; neuron < output_width; ++neuron) {
            const std::size_t pe_index = neuron % pe_array.size();
            const std::size_t ape_index = neuron % ape_array.size();
            ProcessingElement& pe = pe_array[pe_index];
            LocalWeightMemory& local_memory = local_weight_memories[pe_index];

            const std::size_t dispatch_start =
                std::max(bus_ready_cycle, pe_ready_cycle[pe_index]);
            const std::size_t dispatch_finish = dispatch_start + 1;
            bus_ready_cycle = dispatch_finish;
            bus.schedule_neuron_dispatch(pe);

            const std::vector<float> weights = pe.read_local_weights(
                local_memory, layer, neuron, current_width);
            pe_ready_cycle[pe_index] = dispatch_finish + current_width;
            layer_makespan = std::max(layer_makespan, pe_ready_cycle[pe_index]);
            tasks.push_back(NeuronTask{neuron, pe_index, ape_index, weights});
        }

        if (layer_makespan != 0) {
            sc_core::wait(bus.transaction_latency() *
                          static_cast<double>(layer_makespan));
        }

        for (const NeuronTask& task : tasks) {
            ProcessingElement& pe = pe_array[task.pe_index];
            ActivationUnit& ape = ape_array[task.ape_index];
            const float weighted_sum = pe.compute_mac(task.weights);
            bus.transfer_mac_result_to_ape(pe, ape, weighted_sum);
            ape.activate_received();
            const float result_for_cu = bus.transfer_ape_result_to_cu(ape);
            cu_shared_link.transaction();
            shared_memory.write(destination_base + task.neuron, result_for_cu);
        }

        source_base = destination_base;
        destination_base += output_width;
        current_width = output_width;
    }

    const std::size_t final_width = layer_sizes.back();
    const std::size_t final_base = destination_base - final_width;
    return ExecutionResult{final_base, final_width};
}
