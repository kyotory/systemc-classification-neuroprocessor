#include "neuro_processor.h"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {
const sc_core::sc_time kTransactionLatency(1, sc_core::SC_NS);
}

NeuroProcessor::NeuroProcessor(sc_core::sc_module_name name,
                               std::size_t pe_count,
                               NetworkSpec network)
    : sc_module(name),
      pe_count_(pe_count),
      network_(std::move(network)),
      bus("bus", kTransactionLatency),
      io_control_link("cu_io_controller_link", kTransactionLatency),
      configuration_link("cu_configuration_memory_link", kTransactionLatency),
      cu_shared_link("cu_shared_memory_link", kTransactionLatency),
      shared_memory("shared_memory", network_.shared_cells_required()),
      configuration_memory("configuration_memory"),
      local_weight_memories("local_weight_memories", pe_count),
      pe_array("pe_array", pe_count,
          [](const char* module_name, std::size_t) {
              return new ProcessingElement(module_name, kTransactionLatency);
          }),
      ape_array("ape_array", pe_count),
      io_controller("io_controller"),
      control_unit("control_unit") {
    if (pe_count_ == 0) {
        throw std::invalid_argument("PE count must be positive");
    }
    if (!network_.has_class_labels()) {
        throw std::invalid_argument("Classification requires output labels");
    }
    SC_THREAD(run);
}

void NeuroProcessor::run() {
    const sc_core::sc_time model_start = sc_core::sc_time_stamp();

    control_unit.load_configuration(network_, configuration_memory, configuration_link);

    control_unit.signal_io_controller(io_control_link);
    io_controller.load_network(network_, bus, shared_memory, pe_array, local_weight_memories);
    const sc_core::sc_time after_load = sc_core::sc_time_stamp();

    const ExecutionResult result = control_unit.execute(
        bus, shared_memory, configuration_memory,
        configuration_link, cu_shared_link, pe_array, ape_array, local_weight_memories);
    const sc_core::sc_time after_forward_pass = sc_core::sc_time_stamp();

    control_unit.signal_io_controller(io_control_link);
    const std::vector<float> output = io_controller.read_output(
        bus, shared_memory, result.output_base, result.output_size);
    const sc_core::sc_time model_end = sc_core::sc_time_stamp();

    print_classification_result(output);
    print_measurements(model_start, after_load, after_forward_pass, model_end);
    print_core_load();
    print_memory_usage();
    sc_core::sc_stop();
}

void NeuroProcessor::print_input_image() const {
    std::cout << "Input image (" << network_.image_width << "x"
              << network_.image_height << "; # = foreground, . = background):\n";
    for (std::size_t row = 0; row < network_.image_height; ++row) {
        for (std::size_t col = 0; col < network_.image_width; ++col) {
            const float pixel = network_.inputs[row * network_.image_width + col];
            std::cout << (pixel >= 0.5F ? "# " : ". ");
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

void NeuroProcessor::print_classification_result(const std::vector<float>& output) const {
    if (output.size() != network_.output_labels.size() || output.empty()) {
        throw std::runtime_error("Classifier output and output_labels have different sizes");
    }

    std::cout << "[1] Classification\n  ";
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (i) std::cout << " | ";
        std::cout << network_.output_labels[i] << '='
                  << std::fixed << std::setprecision(6) << output[i];
    }

    const auto best = std::max_element(output.begin(), output.end());
    const std::size_t predicted = static_cast<std::size_t>(std::distance(output.begin(), best));
    std::cout << "\n  Predicted: " << network_.output_labels[predicted]
              << " (score " << std::fixed << std::setprecision(6) << *best << ")\n";
}

void NeuroProcessor::print_core_load() const {
    std::cout << "\n[3] PE communication cycles for neuron outputs\n";

    for (std::size_t i = 0; i < pe_array.size(); ++i) {
        const auto& pe = pe_array[i];
        const std::size_t pe_comm_cycles = pe.dispatched_neurons()
                                         + pe.weight_reads()
                                         + pe.mac_result_transfers();

        std::cout << "  PE[" << i << "]: " << pe_comm_cycles << " cycles\n";
    }
}

void NeuroProcessor::print_measurements(const sc_core::sc_time& model_start,
                                        const sc_core::sc_time& after_load,
                                        const sc_core::sc_time& after_forward_pass,
                                        const sc_core::sc_time& model_end) const {
    const sc_core::sc_time network_output_time = after_forward_pass - after_load;
    const sc_core::sc_time total_time = model_end - model_start;
    const double cycle_ns = kTransactionLatency.to_seconds();
    const auto to_cycles = [cycle_ns](const sc_core::sc_time& time) {
        return static_cast<unsigned long long>(time.to_seconds() / cycle_ns + 0.5);
    };

    std::size_t local_weight_transactions = 0;
    for (std::size_t i = 0; i < pe_array.size(); ++i) {
        local_weight_transactions += pe_array[i].weight_reads();
    }
    const std::size_t all_link_transactions = bus.transaction_count()
        + io_control_link.transactions()
        + configuration_link.transactions()
        + cu_shared_link.transactions()
        + local_weight_transactions;
    const auto elapsed_cycles = to_cycles(total_time);

    std::cout << "\n[2] Communication cycles\n"
              << "  Forward pass: " << to_cycles(network_output_time) << " cycles\n"
              << "  Total (load + forward pass + output read): "
              << elapsed_cycles << " cycles\n"
              << "  Communication transactions (all links): "
              << all_link_transactions << '\n';
    if (all_link_transactions > elapsed_cycles) {
        std::cout << "  Parallel transaction overlap: "
                  << (all_link_transactions - elapsed_cycles) << " cycle-equivalents\n";
    }
}

void NeuroProcessor::print_memory_usage() const {
    std::size_t local_weight_payload = 0;
    for (std::size_t i = 0; i < pe_array.size(); ++i) {
        local_weight_payload += local_weight_memories[i].payload_bytes();
    }
    const std::size_t shared_payload = shared_memory.payload_bytes();
    const std::size_t configuration_payload = configuration_memory.payload_bytes();
    const std::size_t logical_total = shared_payload + configuration_payload + local_weight_payload;

    std::cout << "\n[4] Memory\n  Shared=" << shared_payload
              << " B, config=" << configuration_payload
              << " B, weights=" << local_weight_payload
              << " B, total=" << logical_total << " B ("
              << std::fixed << std::setprecision(2)
              << static_cast<double>(logical_total) / 1024.0 << " KiB)\n";
}
