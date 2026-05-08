# Softmax

这里只需要改 `solve.cu`。当前已经提供了 `solve` wrapper 和空的 `softmax_kernel`，你只需要实现 GPU softmax，把长度为 `N` 的 `float` 数组映射到 `output[0..N-1]`。

## 题目要求

- 输入：设备端 `input`，长度为 `N`
- 输出：设备端 `output`，长度为 `N`
- 运算：`output[i] = exp(input[i] - max(input)) / sum_j exp(input[j] - max(input))`
- 数值稳定性：必须使用 max trick，先减去输入最大值再做指数运算

## 文件说明

- `solve.cu`: 你需要修改的核心文件，包含 GPU softmax 实现
- `main.cu`: 对拍 runner，包含 CPU 参考实现、固定测试和随机测试
- `CMakeLists.txt`: 题目目录自己的 CMake 配置
- `generate_test_data.py`: 生成固定测试数据
- `testdata/*.txt`: 固定输入数据

## 根目录 CMake 用法

从仓库根目录配置：

```bash
cmake -S . -B build
```

编译并自动生成测试数据：

```bash
cmake --build build
```

跑 softmax 对拍：

```bash
cmake --build build --target softmax_check
# 或
ctest --test-dir build --output-on-failure -R softmax_compare
```

只跑某个 case：

```bash
./build/softmax/softmax_runner --case ./build/softmax/testdata/case_00_example_3.txt
```

## 测试逻辑

- 固定测试：覆盖题面样例、`N=1`、warp/block 边界、大正数/大负数以及稳定性 case
- 随机测试：运行时按 seed 生成输入，范围包含较大正负值，验证数值稳定性
- 参考答案：CPU 上先求 `max`，再用 `double` 计算分母和每个输出，最后转回 `float`
- 判定方式：
  - 每个元素和 CPU 结果按绝对误差 `5e-5`、相对误差 `5e-5` 比较
  - 额外检查输出是否全为有限值
  - 额外检查输出总和是否接近 `1.0`
