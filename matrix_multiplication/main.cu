#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
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

struct MatrixCase {
    std::string name;
    int M = 0;
    int N = 0;
    int K = 0;
    std::vector<float> A;
    std::vector<float> B;
};

struct CompareResult {
    bool ok = true;
    int row = -1;
    int col = -1;
    float expected = 0.0f;
    float actual = 0.0f;
    float max_abs_diff = 0.0f;
};

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K);

std::vector<float> cpu_reference(const MatrixCase& test_case) {
    std::vector<float> expected(static_cast<size_t>(test_case.M) * test_case.N, 0.0f);

    for (int row = 0; row < test_case.M; ++row) {
        for (int col = 0; col < test_case.N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < test_case.K; ++k) {
                sum += test_case.A[row * test_case.K + k] * test_case.B[k * test_case.N + col];
            }
            expected[row * test_case.N + col] = sum;
        }
    }

    return expected;
}

CompareResult compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual, int M, int N,
                              float abs_tol = 1e-4f, float rel_tol = 1e-4f) {
    CompareResult result;

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            const size_t idx = static_cast<size_t>(row) * N + col;
            const float ref = expected[idx];
            const float got = actual[idx];
            const float abs_diff = std::fabs(ref - got);
            const float limit = abs_tol + rel_tol * std::fabs(ref);

            result.max_abs_diff = std::max(result.max_abs_diff, abs_diff);

            if (!std::isfinite(got) || abs_diff > limit) {
                result.ok = false;
                result.row = row;
                result.col = col;
                result.expected = ref;
                result.actual = got;
                return result;
            }
        }
    }

    return result;
}

MatrixCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    MatrixCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.M >> test_case.N >> test_case.K)) {
        throw std::runtime_error("Failed to read M N K from: " + path.string());
    }
    if (test_case.M <= 0 || test_case.N <= 0 || test_case.K <= 0) {
        throw std::runtime_error("Invalid matrix shape in: " + path.string());
    }

    test_case.A.resize(static_cast<size_t>(test_case.M) * test_case.K);
    test_case.B.resize(static_cast<size_t>(test_case.K) * test_case.N);

    for (float& value : test_case.A) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough A values in: " + path.string());
        }
    }
    for (float& value : test_case.B) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough B values in: " + path.string());
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

MatrixCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> dim_dist(1, 64);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);

    MatrixCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.M = dim_dist(rng);
    test_case.N = dim_dist(rng);
    test_case.K = dim_dist(rng);
    test_case.A.resize(static_cast<size_t>(test_case.M) * test_case.K);
    test_case.B.resize(static_cast<size_t>(test_case.K) * test_case.N);

    for (float& value : test_case.A) {
        value = value_dist(rng);
    }
    for (float& value : test_case.B) {
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

    const fs::path cwd_dir = fs::current_path() / "matrix_multiplication" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const MatrixCase& test_case) {
    const std::vector<float> expected = cpu_reference(test_case);
    std::vector<float> actual(static_cast<size_t>(test_case.M) * test_case.N, 0.0f);

    const size_t bytes_a = test_case.A.size() * sizeof(float);
    const size_t bytes_b = test_case.B.size() * sizeof(float);
    const size_t bytes_c = actual.size() * sizeof(float);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes_a));
    CUDA_CHECK(cudaMalloc(&d_b, bytes_b));
    CUDA_CHECK(cudaMalloc(&d_c, bytes_c));

    try {
        CUDA_CHECK(cudaMemcpy(d_a, test_case.A.data(), bytes_a, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, test_case.B.data(), bytes_b, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_c, 0, bytes_c));

        solve(d_a, d_b, d_c, test_case.M, test_case.N, test_case.K);

        CUDA_CHECK(cudaMemcpy(actual.data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        throw;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    const CompareResult compare = compare_outputs(expected, actual, test_case.M, test_case.N);
    if (!compare.ok) {
        std::cout << "[FAIL] " << test_case.name << " shape=(" << test_case.M << ", " << test_case.N << ", "
                  << test_case.K << ") first mismatch at (" << compare.row << ", " << compare.col
                  << ") expected=" << compare.expected << " actual=" << compare.actual
                  << " max_abs_diff=" << compare.max_abs_diff << '\n';
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " shape=(" << test_case.M << ", " << test_case.N << ", "
              << test_case.K << ") max_abs_diff=" << compare.max_abs_diff << '\n';
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << argv0 << '\n';
    std::cout << "  " << argv0 << " --random_cases 100 --seed 42\n";
    std::cout << "  " << argv0 << " --case ./testdata/case_00_sanity_2x3x4.txt\n";
}

int main(int argc, char** argv) {
    try {
        fs::path test_dir = default_test_dir(argv[0]);
        std::vector<fs::path> explicit_cases;
        int random_cases = 20;
        uint32_t seed = 20260330u;

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
