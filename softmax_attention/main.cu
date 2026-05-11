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

struct AttentionCase {
    std::string name;
    int M = 0;
    int N = 0;
    int d = 0;
    std::vector<float> Q;
    std::vector<float> K;
    std::vector<float> V;
};

struct CompareResult {
    bool ok = true;
    int row = -1;
    int col = -1;
    float expected = 0.0f;
    float actual = 0.0f;
    float max_abs_diff = 0.0f;
};

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d);

std::vector<float> cpu_reference(const AttentionCase& test_case) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(test_case.d));
    std::vector<float> output(static_cast<size_t>(test_case.M) * test_case.d, 0.0f);
    std::vector<float> scores(static_cast<size_t>(test_case.N), 0.0f);

    for (int m = 0; m < test_case.M; ++m) {
        float max_score = -std::numeric_limits<float>::infinity();
        for (int n = 0; n < test_case.N; ++n) {
            float score = 0.0f;
            for (int k = 0; k < test_case.d; ++k) {
                score += test_case.Q[m * test_case.d + k] * test_case.K[n * test_case.d + k];
            }
            score *= scale;
            scores[n] = score;
            max_score = std::max(max_score, score);
        }

        double denom = 0.0;
        for (int n = 0; n < test_case.N; ++n) {
            denom += std::exp(static_cast<double>(scores[n]) - max_score);
        }

        for (int k = 0; k < test_case.d; ++k) {
            double value = 0.0;
            for (int n = 0; n < test_case.N; ++n) {
                const double weight = std::exp(static_cast<double>(scores[n]) - max_score) / denom;
                value += weight * static_cast<double>(test_case.V[n * test_case.d + k]);
            }
            output[m * test_case.d + k] = static_cast<float>(value);
        }
    }

    return output;
}

CompareResult compare_outputs(const std::vector<float>& expected, const std::vector<float>& actual, int rows,
                              int cols, float abs_tol = 1e-4f, float rel_tol = 1e-4f) {
    CompareResult result;

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const size_t idx = static_cast<size_t>(row) * cols + col;
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

AttentionCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    AttentionCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.M >> test_case.N >> test_case.d)) {
        throw std::runtime_error("Failed to read M N d from: " + path.string());
    }
    if (test_case.M <= 0 || test_case.N <= 0 || test_case.d <= 0) {
        throw std::runtime_error("Invalid attention shape in: " + path.string());
    }

    test_case.Q.resize(static_cast<size_t>(test_case.M) * test_case.d);
    test_case.K.resize(static_cast<size_t>(test_case.N) * test_case.d);
    test_case.V.resize(static_cast<size_t>(test_case.N) * test_case.d);

    for (float& value : test_case.Q) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough Q values in: " + path.string());
        }
    }
    for (float& value : test_case.K) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough K values in: " + path.string());
        }
    }
    for (float& value : test_case.V) {
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough V values in: " + path.string());
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

AttentionCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> m_dist(1, 64);
    std::uniform_int_distribution<int> n_dist(1, 64);
    std::uniform_int_distribution<int> d_dist(1, 32);
    std::uniform_real_distribution<float> value_dist(-2.0f, 2.0f);

    AttentionCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.M = m_dist(rng);
    test_case.N = n_dist(rng);
    test_case.d = d_dist(rng);
    test_case.Q.resize(static_cast<size_t>(test_case.M) * test_case.d);
    test_case.K.resize(static_cast<size_t>(test_case.N) * test_case.d);
    test_case.V.resize(static_cast<size_t>(test_case.N) * test_case.d);

    for (float& value : test_case.Q) value = value_dist(rng);
    for (float& value : test_case.K) value = value_dist(rng);
    for (float& value : test_case.V) value = value_dist(rng);

    return test_case;
}

fs::path default_test_dir(const char* argv0) {
    const fs::path exe_path = fs::weakly_canonical(fs::path(argv0));
    const fs::path exe_dir = exe_path.parent_path();

    const fs::path next_to_binary = exe_dir / "testdata";
    if (fs::exists(next_to_binary)) {
        return next_to_binary;
    }

    const fs::path cwd_dir = fs::current_path() / "softmax_attention" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const AttentionCase& test_case) {
    const std::vector<float> expected = cpu_reference(test_case);
    std::vector<float> actual(static_cast<size_t>(test_case.M) * test_case.d, 0.0f);

    const size_t bytes_q = test_case.Q.size() * sizeof(float);
    const size_t bytes_k = test_case.K.size() * sizeof(float);
    const size_t bytes_v = test_case.V.size() * sizeof(float);
    const size_t bytes_o = actual.size() * sizeof(float);

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_o = nullptr;

    CUDA_CHECK(cudaMalloc(&d_q, bytes_q));
    CUDA_CHECK(cudaMalloc(&d_k, bytes_k));
    CUDA_CHECK(cudaMalloc(&d_v, bytes_v));
    CUDA_CHECK(cudaMalloc(&d_o, bytes_o));

    try {
        CUDA_CHECK(cudaMemcpy(d_q, test_case.Q.data(), bytes_q, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, test_case.K.data(), bytes_k, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, test_case.V.data(), bytes_v, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_o, 0, bytes_o));

        solve(d_q, d_k, d_v, d_o, test_case.M, test_case.N, test_case.d);

        CUDA_CHECK(cudaMemcpy(actual.data(), d_o, bytes_o, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        throw;
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_o));

    const CompareResult compare = compare_outputs(expected, actual, test_case.M, test_case.d);
    if (!compare.ok) {
        std::cout << "[FAIL] " << test_case.name << " dims=(M=" << test_case.M << ", N=" << test_case.N
                  << ", d=" << test_case.d << ") first mismatch at (" << compare.row << ", " << compare.col
                  << ") expected=" << compare.expected << " actual=" << compare.actual
                  << " max_abs_diff=" << compare.max_abs_diff << '\n';
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " dims=(M=" << test_case.M << ", N=" << test_case.N
              << ", d=" << test_case.d << ") max_abs_diff=" << compare.max_abs_diff << '\n';
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
}

int main(int argc, char** argv) {
    try {
        fs::path test_dir = default_test_dir(argv[0]);
        std::vector<fs::path> explicit_cases;
        int random_cases = 20;
        uint32_t seed = 20260509u;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--test_dir") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --test_dir");
                test_dir = argv[++i];
            } else if (arg == "--case") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --case");
                explicit_cases.emplace_back(argv[++i]);
            } else if (arg == "--random_cases") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --random_cases");
                random_cases = std::stoi(argv[++i]);
                if (random_cases < 0) throw std::runtime_error("--random_cases must be >= 0");
            } else if (arg == "--seed") {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for --seed");
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
