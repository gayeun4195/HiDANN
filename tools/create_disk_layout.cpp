#include "disk_utils.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void usage(const char *argv0) {
    std::cerr << "Usage: " << argv0
              << " float <base.fbin> <graph.mem.index> <out_prefix_or_disk.index>\n";
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        usage(argv[0]);
        return 1;
    }

    const std::string dtype = argv[1];
    const std::string base = argv[2];
    const std::string graph = argv[3];
    std::string out = argv[4];
    if (out.size() < 11 || out.substr(out.size() - 11) != "_disk.index") {
        out += "_disk.index";
    }

    try {
        if (dtype != "float") {
            throw std::invalid_argument("artifact layout helper currently supports dtype=float only");
        }
        diskann::create_disk_layout<float>(base, graph, out);
    } catch (const std::exception &e) {
        std::cerr << "create_disk_layout failed: " << e.what() << "\n";
        return 2;
    }

    std::cout << "disk_index=" << out << "\n";
    return 0;
}
