#include "neuro_processor.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr const char* kDefaultInputFile = "input/shape_classifier.txt";

}

int sc_main(int argc, char* argv[]) {
    try {
        std::size_t pe_count = 4;
        std::string input_file = kDefaultInputFile;
        bool input_file_provided = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--pe") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--pe requires a positive integer");
                }
                pe_count = static_cast<std::size_t>(std::stoul(argv[++i]));
                if (pe_count == 0) {
                    throw std::runtime_error("--pe must be greater than zero");
                }
            } else if (!arg.empty() && arg[0] == '-') {
                throw std::runtime_error("Unknown option: " + arg);
            } else {
                if (input_file_provided) {
                    throw std::runtime_error("Specify only one classifier input file");
                }
                input_file = arg;
                input_file_provided = true;
            }
        }

        NetworkSpec network = NetworkSpec::load_from_file(input_file);
        if (network.image_width == 0 || network.image_height == 0) {
            throw std::runtime_error(
                "The classifier file must specify image_size width height");
        }
        if (!network.has_class_labels()) {
            throw std::runtime_error(
                "The classifier file must specify output_labels, for example: circle square triangle");
        }

        NeuroProcessor processor("neuro_processor", pe_count, std::move(network));
        sc_core::sc_start();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
