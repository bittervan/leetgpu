#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
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

struct SoftmaxCase {
    std::string name;
    int N = 0;
    std::vector<float> input;
};

struct CompareResult {
    bool ok = true;
    int index = -1;
    float expected = 0.0f;
    float actual = 0.0f;
    float max_abs_diff = 0.0f;
    double actual_sum = 0.0;
};

extern "C" void solve(const float* input, float* output, int N);

std::vector<float> cpu_reference(const SoftmaxCase& test_case) {
    const auto max_iter = std::max_element(test_case.input.begin(), test_case.input.end());
    const float max_value = *max_iter;

    std::vector<double> exps(static_cast<size_t>(test_case.N), 0.0);
    double denom = 0.0;
    for (int i = 0; i < test_case.N; ++i) {
        const double shifted = static_cast<double>(test_case.input[static_cast<size_t>(i)]) - max_value;
        const double value = std::exp(shifted);
        exps[static_cast<size_t>(i)] = value;
        denom += value;
    }

    std::vector<float> expected(static_cast<size_t>(test_case.N), 0.0f);
    for (int i = 0; i < test_case.N; ++i) {
        expected[static_cast<size_t>(i)] = static_cast<float>(exps[static_cast<size_t>(i)] / denom);
    }
    return expected;
}

CompareResult compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual, int size,
                              float abs_tol = 5e-5f, float rel_tol = 5e-5f, double sum_tol = 5e-4) {
    CompareResult result;
    result.actual_sum = std::accumulate(actual.begin(), actual.end(), 0.0);

    for (int i = 0; i < size; ++i) {
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

    if (!std::isfinite(result.actual_sum) || std::fabs(result.actual_sum - 1.0) > sum_tol) {
        result.ok = false;
        result.index = -1;
        result.expected = 1.0f;
        result.actual = static_cast<float>(result.actual_sum);
        return result;
    }

    return result;
}

SoftmaxCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    SoftmaxCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.N)) {
        throw std::runtime_error("Failed to read N from: " + path.string());
    }
    if (test_case.N <= 0) {
        throw std::runtime_error("Invalid softmax size in: " + path.string());
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

SoftmaxCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> size_dist(1, 32768);
    std::uniform_real_distribution<float> value_dist(-20.0f, 20.0f);

    SoftmaxCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.N = size_dist(rng);
    test_case.input.resize(static_cast<size_t>(test_case.N));

    for (float& value : test_case.input) {
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

    const fs::path cwd_dir = fs::current_path() / "softmax" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const SoftmaxCase& test_case) {
    const std::vector<float> expected = cpu_reference(test_case);
    std::vector<float> actual(static_cast<size_t>(test_case.N), std::numeric_limits<float>::quiet_NaN());

    const size_t bytes = actual.size() * sizeof(float);

    float* d_input = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_input, test_case.input.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_output, actual.data(), bytes, cudaMemcpyHostToDevice));

        solve(d_input, d_output, test_case.N);

        CUDA_CHECK(cudaMemcpy(actual.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_input);
        cudaFree(d_output);
        throw;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    const CompareResult compare = compare_outputs(expected, actual, test_case.N);
    if (!compare.ok) {
        if (compare.index >= 0) {
            std::cout << "[FAIL] " << test_case.name << " N=" << test_case.N << " first mismatch at index "
                      << compare.index << " expected=" << compare.expected << " actual=" << compare.actual
                      << " max_abs_diff=" << compare.max_abs_diff << " sum=" << compare.actual_sum << '\n';
        } else {
            std::cout << "[FAIL] " << test_case.name << " N=" << test_case.N
                      << " invalid probability sum expected=1 actual=" << compare.actual
                      << " max_abs_diff=" << compare.max_abs_diff << '\n';
        }
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " N=" << test_case.N << " max_abs_diff=" << compare.max_abs_diff
              << " sum=" << compare.actual_sum << '\n';
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << argv0 << '\n';
    std::cout << "  " << argv0 << " --random_cases 100 --seed 42\n";
    std::cout << "  " << argv0 << " --case ./testdata/case_00_example_3.txt\n";
}

int main(int argc, char** argv) {
    try {
        fs::path test_dir = default_test_dir(argv[0]);
        std::vector<fs::path> explicit_cases;
        int random_cases = 20;
        uint32_t seed = 20260507u;

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
        return 2;
    }
}
