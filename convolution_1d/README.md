# Convolution 1D

这里只需要改 `solve.cu`。当前已经提供了 `solve` wrapper，你只需要实现 `convolution_1d_kernel`。

## 文件说明

- `solve.cu`: 你需要修改的唯一文件，主要补上 `convolution_1d_kernel`
- `main.cu`: 对拍 runner，包含 CPU 参考实现、固定测试和随机测试
- `CMakeLists.txt`: 题目目录自己的 CMake 配置
- `generate_test_data.py`: 生成固定测试数据
- `testdata/*.txt`: 固定输入数据

## 题目约定

- 测试框架采用 `valid 1D convolution` 定义：
- `output[i] = sum(input[i + j] * kernel[j])`，其中 `0 <= j < kernel_size`
- 输出长度是 `input_size - kernel_size + 1`
- `input`、`kernel`、`output` 都是 GPU 上的 device pointer

## 根目录 CMake 用法

从仓库根目录配置：

```bash
cmake -S . -B build
```

默认会在构建目录下生成 `compile_commands.json`，例如 `build/compile_commands.json`。

编译时会自动在构建目录下生成测试数据，再编译可执行文件：

```bash
cmake --build build
```

跑对拍：

```bash
cmake --build build --target convolution_1d_check
# 或
ctest --test-dir build --output-on-failure -R convolution_1d_compare
```

跑全部题目：

```bash
cmake --build build --target check
```

如果你希望普通 build 结束后自动对拍：

```bash
cmake -S . -B build -DLEETGPU_RUN_CHECK_ON_BUILD=ON
cmake --build build
```

## 只保留 CMake 构建

手动生成测试数据：

```bash
python3 convolution_1d/generate_test_data.py
```

## 测试逻辑

- 固定测试：读取 `testdata` 目录下的 `.txt`
- 随机测试：运行时按 seed 生成 `input` 和 `kernel`
- 参考答案：CPU 上按滑动窗口逐点累加
- 判定方式：比较 GPU 输出和 CPU 输出，使用 `1e-4` 绝对/相对误差
