#include <rocksdb/db.h>
#include <iostream>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>  // For JSON serialization

using json = nlohmann::json;

// Function to generate the key
std::string generate_key(const std::string& repository_name, const std::string& func_path_in_repository, const std::string& func_name) {
    return repository_name + ":" + func_path_in_repository + ":" + func_name;
}

void store_in_rocksdb(rocksdb::DB* db, const json& row, size_t index) {
    // Check if the row is a string (i.e., JSON is stringified)
    json actual_row = row;
    if (row.is_string()) {
        try {
            actual_row = json::parse(row.get<std::string>());
        } catch (const json::parse_error& e) {
            std::cerr << "Error parsing stringified JSON at index " << index << ": " << e.what() << std::endl;
            return;  // Skip this row if it cannot be parsed
        }
    }

    // Check if all required fields are present
    if (!actual_row.contains("repository_name") || !actual_row.contains("func_path_in_repository") || !actual_row.contains("func_name")) {
        std::cerr << "Missing required fields in row at index " << index << std::endl;
        
        // Print the row to debug
        std::cerr << "Row data: " << actual_row.dump(4) << std::endl;

        // Skip this row if missing fields
        return;
    }

    std::string repository_name = actual_row["repository_name"];
    std::string func_path_in_repository = actual_row["func_path_in_repository"];
    std::string func_name = actual_row["func_name"];

    // Generate the key
    std::string key = generate_key(repository_name, func_path_in_repository, func_name);

    // Serialize the row data to JSON string
    std::string value = actual_row.dump();

    // Write to RocksDB
    rocksdb::Status status = db->Put(rocksdb::WriteOptions(), key, value);
    if (!status.ok()) {
        std::cerr << "Error writing to RocksDB: " << status.ToString() << std::endl;
    }
}

int main() {
    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;

    // Open the RocksDB database
    rocksdb::Status status = rocksdb::DB::Open(options, "testdb", &db);
    if (!status.ok()) {
        std::cerr << "Unable to open RocksDB: " << status.ToString() << std::endl;
        return 1;
    }

    // Load the dataset JSON file
    std::ifstream file("all_rows_serialized.json");
    json dataset;
    file >> dataset;  // Read JSON data from the file

    // Iterate over each row and store it in RocksDB
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& row = dataset[i];
        store_in_rocksdb(db, row, i);
        std::cout << "Progress: " << i + 1 << "/" << dataset.size() << " entries added to RocksDB." << std::endl;
    }

    std::cout << "Data stored successfully in RocksDB." << std::endl;

    // Close the database
    delete db;

    return 0;
}
