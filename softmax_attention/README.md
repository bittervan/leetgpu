# Softmax Attention

这里只需要改 `solve.cu`。当前已经提供了 `solve` wrapper，你只需要实现 row-wise softmax attention，把 `Q K^T` 做 softmax 后再乘 `V`，输出写到 `output`。

## 文件说明

- `solve.cu`: 你需要修改的唯一文件，补上 attention kernel
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

跑对拍：

```bash
cmake --build build --target softmax_attention_check
# 或
ctest --test-dir build --output-on-failure -R softmax_attention_compare
```

手动生成测试数据：

```bash
python3 softmax_attention/generate_test_data.py
```

## 测试逻辑

- 输入约定：`Q` 是 `M x d`，`K` 是 `N x d`，`V` 是 `N x d`
- 输出约定：`output` 是 `M x d`
- 参考答案：CPU 上先算 `scores = QK^T / sqrt(d)`，对每一行做 softmax，再乘 `V`
- 判定方式：逐元素比较输出，使用 `1e-4` 绝对/相对误差
- 固定测试：覆盖题面样例、小尺寸、warp 边界、block 边界、以及性能 case `512 x 256 x 128`
