#include <cuda_runtime.h>

#include <algorithm>
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

struct ImageCase {
    std::string name;
    int width = 0;
    int height = 0;
    std::vector<unsigned char> pixels;
};

struct CompareResult {
    bool ok = true;
    int x = -1;
    int y = -1;
    int channel = -1;
    int expected = -1;
    int actual = -1;
};

extern "C" void solve(unsigned char* image, int width, int height);

std::vector<unsigned char> cpu_reference(const ImageCase& test_case) {
    std::vector<unsigned char> expected = test_case.pixels;

    const size_t pixel_count = static_cast<size_t>(test_case.width) * test_case.height;
    for (size_t pixel_idx = 0; pixel_idx < pixel_count; ++pixel_idx) {
        const size_t base = pixel_idx * 4;
        for (int channel = 0; channel < 3; ++channel) {
            expected[base + channel] = static_cast<unsigned char>(255 - expected[base + channel]);
        }
    }

    return expected;
}

CompareResult compare_outputs(const std::vector<unsigned char>& expected, const std::vector<unsigned char>& actual,
                              int width, int height) {
    CompareResult result;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t base = (static_cast<size_t>(y) * width + x) * 4;
            for (int channel = 0; channel < 4; ++channel) {
                const unsigned char ref = expected[base + channel];
                const unsigned char got = actual[base + channel];
                if (ref != got) {
                    result.ok = false;
                    result.x = x;
                    result.y = y;
                    result.channel = channel;
                    result.expected = static_cast<int>(ref);
                    result.actual = static_cast<int>(got);
                    return result;
                }
            }
        }
    }

    return result;
}

ImageCase load_case_from_file(const fs::path& path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open test file: " + path.string());
    }

    ImageCase test_case;
    test_case.name = path.filename().string();

    if (!(fin >> test_case.width >> test_case.height)) {
        throw std::runtime_error("Failed to read width height from: " + path.string());
    }
    if (test_case.width <= 0 || test_case.height <= 0) {
        throw std::runtime_error("Invalid image shape in: " + path.string());
    }

    const size_t value_count = static_cast<size_t>(test_case.width) * test_case.height * 4;
    test_case.pixels.resize(value_count);

    for (size_t i = 0; i < value_count; ++i) {
        int value = -1;
        if (!(fin >> value)) {
            throw std::runtime_error("Not enough pixel values in: " + path.string());
        }
        if (value < 0 || value > 255) {
            throw std::runtime_error("Pixel value out of range [0, 255] in: " + path.string());
        }
        test_case.pixels[i] = static_cast<unsigned char>(value);
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

ImageCase make_random_case(int case_index, std::mt19937& rng) {
    std::uniform_int_distribution<int> width_dist(1, 128);
    std::uniform_int_distribution<int> height_dist(1, 128);
    std::uniform_int_distribution<int> value_dist(0, 255);

    ImageCase test_case;
    test_case.name = "random_" + std::to_string(case_index);
    test_case.width = width_dist(rng);
    test_case.height = height_dist(rng);
    test_case.pixels.resize(static_cast<size_t>(test_case.width) * test_case.height * 4);

    for (unsigned char& pixel : test_case.pixels) {
        pixel = static_cast<unsigned char>(value_dist(rng));
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

    const fs::path cwd_dir = fs::current_path() / "image_invert" / "testdata";
    if (fs::exists(cwd_dir)) {
        return cwd_dir;
    }

    return next_to_binary;
}

bool run_case(const ImageCase& test_case) {
    const std::vector<unsigned char> expected = cpu_reference(test_case);
    std::vector<unsigned char> actual = test_case.pixels;

    const size_t bytes = actual.size() * sizeof(unsigned char);

    unsigned char* d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_image, actual.data(), bytes, cudaMemcpyHostToDevice));

        solve(d_image, test_case.width, test_case.height);

        CUDA_CHECK(cudaMemcpy(actual.data(), d_image, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_image);
        throw;
    }

    CUDA_CHECK(cudaFree(d_image));

    const CompareResult compare = compare_outputs(expected, actual, test_case.width, test_case.height);
    if (!compare.ok) {
        static constexpr const char* kChannelNames[] = {"R", "G", "B", "A"};
        std::cout << "[FAIL] " << test_case.name << " shape=(" << test_case.height << ", " << test_case.width
                  << ") first mismatch at (x=" << compare.x << ", y=" << compare.y
                  << ", channel=" << kChannelNames[compare.channel] << ") expected=" << compare.expected
                  << " actual=" << compare.actual << '\n';
        return false;
    }

    std::cout << "[PASS] " << test_case.name << " shape=(" << test_case.height << ", " << test_case.width << ")\n";
    return true;
}

void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0 << " [--test_dir PATH] [--case FILE] [--random_cases N] [--seed SEED]\n";
    std::cout << "Examples:\n";
    std::cout << "  " << argv0 << '\n';
    std::cout << "  " << argv0 << " --random_cases 100 --seed 42\n";
    std::cout << "  " << argv0 << " --case ./testdata/case_00_example_1x2.txt\n";
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
