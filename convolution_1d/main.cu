#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
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

struct ConvolutionCase {
    std::string name;
    int input_size = 0;
    int kernel_size = 0;
    std::vector<float> input;
    std::vector<float> kernel;
};

struct CompareResult {
    bool ok = true;
    int index = -1;
    float expected = 0.0f;
    float actual = 0.0f;
    float max_abs_diff = 0.0f;
};

extern "C" void solve(const float* input, const float* kernel, float* output, int input_size, int kernel_size);

std::vector<float> cpu_reference(const ConvolutionCase& test_case) {
    const int output_size = test_case.input_size - test_case.kernel_size + 1;
    std::vector<float> expected(static_cast<size_t>(output_size), 0.0f);

    for (int out_idx = 0; out_idx < output_size; ++out_idx) {
        float sum = 0.0f;
        for (int kernel_idx = 0; kernel_idx < test_case.kernel_size; ++kernel_idx) {
            sum += test_case.input[static_cast<size_t>(out_idx + kernel_idx)] *
                   test_case.kernel[static_cast<size_t>(kernel_idx)];
        }
        expected[static_cast<size_t>(out_idx)] = sum;
    }

    return expected;
}

CompareResult compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual, int output_size,
                              float abs_tol = 1e-4f, float rel_tol = 1e-4f) {
    CompareResult result;

    for (int i = 0; i < output_size; ++i) {
        const float ref = expected[static_cast<size_t>(i)];
        const float got = actual[static_cast<size_t>(i)];
        const float abs_diff = std::fabs(ref - got);
        const float limit = abs_tol + rel_tol * std::fabs(ref);

        result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);

        if (!std::isfinite(got) || abs_diff > limit) {
            result.ok = false;
            result.index = i;
            result.expected = ref;
            result.actual = got;
            return result;
        }
    }

    return result;
}

ConvolutionCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    ConvolutionCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.input_size >> test_case.kernel_size)) {
        throw std::runtime_error("Failed to read input_size kernel_size from: " + path.string());
    }
    if (test_case.input_size <= 0 || test_case.kernel_size <= 0 || test_case.kernel_size > test_case.input_size) {
        throw std::runtime_error("Invalid convolution sizes in: " + path.string());
    }

    test_case.input.resize(static_cast<size_t>(test_case.input_size));
    test_case.kernel.resize(static_cast<size_t>(test_case.kernel_size));

    for (float& value : test_case.input) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough input values in: " + path.string());
        }
    }
    for (float& value : test_case.kernel) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough kernel values in: " + path.string());
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

ConvolutionCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> input_size_dist(1, 4096);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);

    ConvolutionCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.input_size = input_size_dist(rng);

    const int kernel_limit = std::min(test_case.input_size, 128);
    std::uniform_int_distribution<int> kernel_size_dist(1, kernel_limit);
    test_case.kernel_size = kernel_size_dist(rng);

    test_case.input.resize(static_cast<size_t>(test_case.input_size));
    test_case.kernel.resize(static_cast<size_t>(test_case.kernel_size));

    for (float& value : test_case.input) {
        value = value_dist(rng);
    }
    for (float& value : test_case.kernel) {
        value = value_dist(rng);
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

    const fs::path cwd_dir = fs::current_path() / "convolution_1d" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const ConvolutionCase& test_case) {
    const int output_size = test_case.input_size - test_case.kernel_size + 1;
    const std::vector<float> expected = cpu_reference(test_case);
    std::vector<float> actual(static_cast<size_t>(output_size), 0.0f);

    const size_t input_bytes = test_case.input.size() * sizeof(float);
    const size_t kernel_bytes = test_case.kernel.size() * sizeof(float);
    const size_t output_bytes = actual.size() * sizeof(float);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_input, test_case.input.data(), input_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kernel, test_case.kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

        solve(d_input, d_kernel, d_output, test_case.input_size, test_case.kernel_size);

        CUDA_CHECK(cudaMemcpy(actual.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_kernel);
        cudaFree(d_output);
        throw;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    const CompareResult compare = compare_outputs(expected, actual, output_size);
    if (!compare.ok) {
        std::cout << "[FAIL] " << test_case.name << " input_size=" << test_case.input_size
                  << " kernel_size=" << test_case.kernel_size << " output_size=" << output_size
                  << " first mismatch at index " << compare.index << " expected=" << compare.expected
                  << " actual=" << compare.actual << " max_abs_diff=" << compare.max_abs_diff << '\n';
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " input_size=" << test_case.input_size
              << " kernel_size=" << test_case.kernel_size << " output_size=" << output_size
              << " max_abs_diff=" << compare.max_abs_diff << '\n';
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << argv0 << '\n';
    std::cout << "  " << argv0 << " --random_cases 100 --seed 42\n";
    std::cout << "  " << argv0 << " --case ./testdata/case_00_sanity_5_3.txt\n";
}

int main(int argc, char** argv) {
    try {
        fs::path test_dir = default_test_dir(argv[0]);
        std::vector<fs::path> explicit_cases;
        int random_cases = 20;
        uint32_t seed = 20260403u;

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
