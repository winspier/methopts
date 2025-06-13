#include "LinearRegressionSGD.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <data.tsv> <output_dir_or_file>" << std::endl;
        return 1;
    }

    const std::string data_file  = argv[1];
    const std::string out_arg    = argv[2];

    std::ifstream infile(data_file);
    if (!infile.is_open()) {
        std::cerr << "Failed to open data file: " << data_file << std::endl;
        return 1;
    }

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::string line;

    std::getline(infile, line);

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (std::getline(iss, token, '\t')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 3) continue;
        try {
            double x1     = std::stod(tokens[0]);
            double x2     = std::stod(tokens[1]);
            double target = std::stod(tokens[2]);
            X.push_back({x1, x2});
            y.push_back(target);
        } catch (...) {
            std::cerr << "Warning: skipping invalid row: " << line << std::endl;
        }
    }
    infile.close();

    std::cout << "Loaded " << X.size() << " samples." << std::endl;
    if (X.empty()) {
        std::cerr << "ERROR: No data loaded! Check your dataset file format." << std::endl;
        return 1;
    }

    const double learning_rate = 0.01;
    const int    max_iters     = 1000;
    Vec lower_bounds = {-10.0, -10.0};
    Vec upper_bounds = { 10.0,  10.0};

    LinearRegressionSGD model(learning_rate, max_iters, lower_bounds, upper_bounds);
    Vec initial_beta = {0.0, 0.0};

    Vec beta = model.fit(X, y, initial_beta);

    std::cout << "Trained beta parameters:" << std::endl;
    for (size_t i = 0; i < beta.size(); ++i) {
        std::cout << "  beta[" << i << "] = " << beta[i] << std::endl;
    }

    fs::path out_path = out_arg;
    fs::path beta_path;

    if (fs::exists(out_path) && fs::is_directory(out_path)) {
        beta_path = out_path / "beta.txt";
    } else if (out_path.has_extension()) {
        beta_path = out_path;
    } else {
        beta_path = out_path / "beta.txt";
    }

    if (beta_path.has_parent_path()) {
        fs::create_directories(beta_path.parent_path());
    }

    std::ofstream outfile(beta_path);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << beta_path << std::endl;
        return 1;
    }
    for (double b : beta) {
        outfile << b << "\n";
    }
    outfile.close();

    std::cout << "Saved beta to " << beta_path << std::endl;
    return 0;
}
