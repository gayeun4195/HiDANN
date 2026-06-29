// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <string>

#include "partition.h"

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cout << "Usage:\n"
                  << argv[0]
                  << " datatype<int8/uint8/float> <data_path> <ids_uint32_bin> <output_data_bin>"
                  << std::endl;
        return -1;
    }

    const std::string data_path(argv[2]);
    const std::string ids_path(argv[3]);
    const std::string output_path(argv[4]);

    if (std::string(argv[1]) == std::string("float"))
        return retrieve_shard_data_from_ids<float>(data_path, ids_path, output_path);
    if (std::string(argv[1]) == std::string("int8"))
        return retrieve_shard_data_from_ids<int8_t>(data_path, ids_path, output_path);
    if (std::string(argv[1]) == std::string("uint8"))
        return retrieve_shard_data_from_ids<uint8_t>(data_path, ids_path, output_path);

    std::cout << "Unsupported data format. Use float/int8/uint8." << std::endl;
    return -1;
}
