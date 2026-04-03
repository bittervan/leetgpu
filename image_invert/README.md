# Image Invert

这里只需要改 `solve.cu`。当前已经提供了 `solve` wrapper，你只需要实现 `invert_kernel`。

## 文件说明

- `solve.cu`: 你需要修改的唯一文件，主要补上 `invert_kernel`
- `main.cu`: 对拍 runner，包含 CPU 参考实现、固定测试和随机测试
- `CMakeLists.txt`: 题目目录自己的 CMake 配置
- `generate_test_data.py`: 生成固定测试数据
- `testdata/*.txt`: 固定输入数据

## 题目约定

- 输入是一张 RGBA 图像，按行优先（row-major）展开
- 每个像素由 4 个 `unsigned char` 组成，顺序是 `R, G, B, A`
- `image` 总长度是 `width * height * 4`
- `image` 是 GPU 上的 device pointer
- `solve` 需要原地修改 `image`
- 只反转 `R`、`G`、`B` 三个通道：`channel = 255 - channel`
- `A` 通道必须保持不变

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
cmake --build build --target image_invert_check
# 或
ctest --test-dir build --output-on-failure -R image_invert_compare
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
python3 image_invert/generate_test_data.py
```

## 测试逻辑

- 维度约定：图像形状是 `height x width`
- 固定测试：读取 `testdata` 目录下的 `.txt`
- 随机测试：运行时按 seed 生成图像
- 参考答案：CPU 上逐像素计算 `RGBA -> (255-R, 255-G, 255-B, A)`
- 判定方式：逐像素逐通道精确比较 GPU 输出和 CPU 输出
