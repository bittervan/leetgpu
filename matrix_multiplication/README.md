# Matrix Multiplication

这里只需要改 `solve.cu`。里面包含你要实现的 `matrix_multiplication_kernel`，以及调用它的 `solve` wrapper。

## 文件说明

- `solve.cu`: 你需要修改的唯一文件，包含 `matrix_multiplication_kernel` 和 `solve` wrapper
- `main.cu`: 对拍 runner，包含 CPU 参考实现、固定测试和随机测试
- `CMakeLists.txt`: 题目目录自己的 CMake 配置
- `generate_test_data.py`: 生成固定测试数据
- `testdata/*.txt`: 固定输入数据

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
cmake --build build --target check
# 或
ctest --test-dir build --output-on-failure
```

如果你希望普通 build 结束后自动对拍：

```bash
cmake -S . -B build -DLEETGPU_RUN_CHECK_ON_BUILD=ON
cmake --build build
```

## 只保留 CMake 构建

手动生成测试数据：

```bash
python3 matrix_multiplication/generate_test_data.py
```

## 测试逻辑

- 固定测试：读取 `testdata` 目录下的 `.txt`
- 随机测试：运行时按 seed 生成矩阵
- 参考答案：CPU 上用朴素三重循环计算
- 判定方式：比较 GPU 输出和 CPU 输出，使用 `1e-4` 绝对/相对误差
