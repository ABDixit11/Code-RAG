#include <iostream>
#include <fstream>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace fs = std::filesystem;
using json = nlohmann::json;

int main() {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = false;  // Open existing DB, don't create a new one

    // Path to the RocksDB database
    std::string db_path = "/app/testdb";
    std::string output_dir = "/app/batch_json_files";

    // Ensure the output directory exists
    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
        std::cout << "Created directory: " << output_dir << std::endl;
    }

    // Open the RocksDB database
    rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);
    if (!status.ok()) {
        std::cerr << "Error opening RocksDB: " << status.ToString() << std::endl;
        return 1;
    }

    // Fetch rows from RocksDB in batches
    const size_t batch_size = 10000;
    size_t current_batch_size = 0;
    size_t batch_index = 0;

    std::unique_ptr<rocksdb::Iterator> it(db->NewIterator(rocksdb::ReadOptions()));
    it->SeekToFirst();

    json batchData = json::array();  // Start with an empty JSON array

    while (it->Valid()) {
        std::string key = it->key().ToString();
        std::string value = it->value().ToString();

        // Add the current key-value pair as a new entry in the JSON array
        json entry;
        entry["func_code_string"] = value;  // Add value to JSON
        batchData.push_back(entry);  // Add to array

        current_batch_size++;

        // Save the batch when the batch size is reached
        if (current_batch_size >= batch_size) {
            std::string batch_file = output_dir + "/batch_" + std::to_string(batch_index) + ".json";
            std::ofstream outFile(batch_file);
            outFile << batchData.dump(4);  // Save batch as formatted JSON
            outFile.close();

            std::cout << "Saved batch " << batch_index << " with " << current_batch_size << " records to " << batch_file << std::endl;

            batchData.clear();  // Reset the array for the next batch
            current_batch_size = 0;
            batch_index++;
        }

        it->Next();
    }

    // Save any remaining records
    if (current_batch_size > 0) {
        std::string batch_file = output_dir + "/batch_" + std::to_string(batch_index) + ".json";
        std::ofstream outFile(batch_file);
        outFile << batchData.dump(4);
        outFile.close();

        std::cout << "Saved final batch " << batch_index << " with " << current_batch_size << " records to " << batch_file << std::endl;
    }

    delete db;
    return 0;
}
