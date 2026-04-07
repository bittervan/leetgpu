#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t error__ = (call);                                                    \
        if (error__ != cudaSuccess) {                                                          \
            std::ostringstream oss__;                                                          \
            oss__ << "CUDA error: " << cudaGetErrorString(error__) << " at " << __FILE__       \
                  << ":" << __LINE__;                                                          \
            throw std::runtime_error(oss__.str());                                             \
        }                                                                                      \
    } while (false)

namespace fs = std::filesystem;

struct ReductionCase {
    std::string name;
    int N = 0;
    std::vector<float> input;
};

struct CompareResult {
    bool ok = true;
    float expected = 0.0f;
    float actual = 0.0f;
    float abs_diff = 0.0f;
};

extern "C" void solve(const float* input, float* output, int N);

float cpu_reference(const ReductionCase& test_case) {
    double sum = 0.0;
    for (const float value : test_case.input) {
        sum += static_cast<double>(value);
    }
    return static_cast<float>(sum);
}

CompareResult compare_outputs(float expected, float actual, float abs_tol = 1e-5f, float rel_tol = 1e-6f) {
    CompareResult result;
    result.expected = expected;
    result.actual = actual;
    result.abs_diff = std::fabs(expected - actual);

    const float limit = abs_tol + rel_tol * std::fabs(expected);
    if (!std::isfinite(actual) || result.abs_diff > limit) {
        result.ok = false;
    }

    return result;
}

ReductionCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    ReductionCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.N)) {
        throw std::runtime_error("Failed to read N from: " + path.string());
    }
    if (test_case.N <= 0) {
        throw std::runtime_error("Invalid reduction size in: " + path.string());
    }

    test_case.input.resize(static_cast<size_t>(test_case.N));
    for (float& value : test_case.input) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough input values in: " + path.string());
        }
    }

    return test_case;
}

std::vector<fs::path> discover_test_files(const fs::path& test_dir) {
    std::vector<fs::path> files;
    if (!fs::exists(test_dir) || !fs::is_directory(test_dir)) {
        return files;
    }

    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

ReductionCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> size_dist(1, 65536);
    std::uniform_int_distribution<int> value_dist(-16, 16);

    ReductionCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.N = size_dist(rng);
    test_case.input.resize(static_cast<size_t>(test_case.N));

    for (float& value : test_case.input) {
        value = static_cast<float>(value_dist(rng));
    }

    return test_case;
}

fs::path default_test_dir(const char* argv0) {
    const fs::path exe_path = fs::weakly_canonical(fs::path(argv0));
    const fs::path exe_dir = exe_path.parent_path();

    const fs::path next_to_binary = exe_dir / "testdata";
    if (fs::exists(next_to_binary)) {
        return next_to_binary;
    }

    const fs::path cwd_dir = fs::current_path() / "reduction" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const ReductionCase& test_case) {
    const float expected = cpu_reference(test_case);
    float actual = std::numeric_limits<float>::quiet_NaN();

    const size_t input_bytes = test_case.input.size() * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    try {
        CUDA_CHECK(cudaMemcpy(d_input, test_case.input.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_output, &actual, sizeof(float), cudaMemcpyHostToDevice));

        solve(d_input, d_output, test_case.N);

        CUDA_CHECK(cudaMemcpy(&actual, d_output, sizeof(float), cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    const CompareResult compare = compare_outputs(expected, actual);
    if (!compare.ok) {
        std::cout << "[FAIL] " << test_case.name << " N=" << test_case.N << " expected=" << compare.expected
                  << " actual=" << compare.actual << " abs_diff=" << compare.abs_diff << '\n';
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " N=" << test_case.N << " sum=" << compare.actual
              << " abs_diff=" << compare.abs_diff << '\n';
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << argv0 << '\n';
    std::cout << "  " << argv0 << " --random_cases 100 --seed 42\n";
    std::cout << "  " << argv0 << " --case ./testdata/case_00_example_8.txt\n";
}

int main(int argc, char** argv) {
    try {
        fs::path test_dir = default_test_dir(argv[0]);
        std::vector<fs::path> explicit_cases;
        int random_cases = 20;
        uint32_t seed = 20260407u;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--test_dir") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --test_dir");
                }
                test_dir = argv[++i];
            } else if (arg == "--case") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --case");
                }
                explicit_cases.emplace_back(argv[++i]);
            } else if (arg == "--random_cases") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --random_cases");
                }
                random_cases = std::stoi(argv[++i]);
                if (random_cases < 0) {
                    throw std::runtime_error("--random_cases must be >= 0");
                }
            } else if (arg == "--seed") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("Missing value for --seed");
                }
                seed = static_cast<uint32_t>(std::stoul(argv[++i]));
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown argument: " + arg);
            }
        }

        std::cout << "Using test_dir: " << test_dir << '\n';
        std::cout << "Random seed: " << seed << ", random cases: " << random_cases << '\n';

        int total = 0;
        int passed = 0;

        if (!explicit_cases.empty()) {
            for (const auto& path : explicit_cases) {
                ++total;
                passed += run_case(load_case_from_file(path)) ? 1 : 0;
            }
        } else {
            const std::vector<fs::path> files = discover_test_files(test_dir);
            if (files.empty()) {
                std::cout << "[WARN] No fixed test files found in " << test_dir << '\n';
            }

            for (const auto& path : files) {
                ++total;
                passed += run_case(load_case_from_file(path)) ? 1 : 0;
            }
        }

        std::mt19937 rng(seed);
        for (int i = 0; i < random_cases; ++i) {
            ++total;
            passed += run_case(make_random_case(i, rng)) ? 1 : 0;
        }

        std::cout << "\nSummary: " << passed << "/" << total << " cases passed.\n";
        return passed == total ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] " << ex.what() << '\n';
        return 1;
    }
}
