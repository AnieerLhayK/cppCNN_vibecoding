# 设计说明

## 设计目标

项目优先保证 CNN 数值核心清晰、可测试，并能在 Visual Studio 中演示。Qt 和 OpenCV 仅位于交互及图片边界，核心网络不依赖任何深度学习框架。

## 模块划分

- `cnn/Tensor`：CHW 连续 `float` 存储与边界检查。
- `cnn/ConvLayer`：多通道卷积及权重、偏置梯度。
- `cnn/Activation`：ReLU 和数值稳定 Softmax。
- `cnn/PoolingLayer`：最大池化及最大值索引反传。
- `cnn/FlattenLayer`：张量与向量形状转换。
- `cnn/FCLayer`：全连接前向、反向与参数更新。
- `cnn/Loss`：交叉熵、argmax 和准确率。
- `cnn/CNN`：LeNet 拓扑、Top-1、模型保存、加载与检查。
- `cnn/Trainer`：mini-batch SGD、L2 衰减和评估。
- `data/DataLoader`：GTSRB 原始目录、CSV 和子集映射。
- `image/ImageProcessor`：CLI 图片读取、缩放与归一化。
- `gui/ImageBridge`：`QImage` 到 `Tensor(3,32,32)` 的转换。
- `gui/ResourceLocator`：模型、标签和演示图片的跨目录查找。
- `gui/InferenceEngine`：一次推理、Top-K 排序和耗时统计。
- `gui/AppController`：GUI 状态、异步任务和错误反馈的协调器。
- `qml/`：只负责界面呈现与用户交互，不执行 CNN 数值计算。

训练属于开发者工作流，不嵌入 GUI。`scripts/train_model.ps1` 组合子集工具、CLI、CTest、评估和元数据记录；GUI 只加载经过开发者确认的模型。

## 网络数据流

```text
3x32x32
 -> Conv(3,6,5)  -> 6x28x28
 -> ReLU
 -> MaxPool(2,2) -> 6x14x14
 -> Conv(6,16,5) -> 16x10x10
 -> ReLU
 -> MaxPool(2,2) -> 16x5x5
 -> Flatten      -> 400
 -> FC            -> 120
 -> ReLU
 -> FC            -> class_count
 -> Softmax
```

## GUI 架构

界面采用 Qt Quick Controls 2 的 Basic 风格，自定义深色面板、按钮、状态胶囊和概率条。窗口最小尺寸为 `1100 x 700`，主要区域包括：

1. 顶栏：应用名称、模型状态、类别数、设置和关于。
2. 图片区：打开、拖放、预览、清除和格式错误提示。
3. 预测区：Top-1、置信度、推理时间和 Top-3。
4. 演示图区：五个不同 GTSRB 类别，点击后自动推理。
5. 状态栏：模型路径、运行状态和可读错误。

`AppController::predict()` 使用 `QtConcurrent::run` 调用 `InferenceEngine`。工作线程持有网络共享指针、输入张量副本和标签副本，完成后由 `QFutureWatcher` 在 GUI 线程更新属性，避免直接从工作线程访问 QML。

这种拆分使资源路径规则、数值推理和界面状态可以分别测试。加载新模型时会先在临时对象中完成格式检查，只有验证成功才替换当前有效模型，避免一次错误选择破坏可用会话。

## 模型与资源发现

GUI 启动不会创建随机模型冒充训练结果。未找到模型时仍显示主界面，并提供“Select model”入口。

模型搜索：

1. `applicationDir/models/gtsrb_v2_subset10.bin`
2. `currentDir/codex/models/gtsrb_v2_subset10.bin`
3. `currentDir/models/gtsrb_v2_subset10.bin`

标签和演示图采用相同的发布目录优先、开发目录回退策略。

模型加载检查：

- 魔数 `CPPCNN1\0`
- 格式版本
- 类别数量
- 四个可训练层及其顺序
- 权重和偏置数据完整性
- 参数总数

10 类模型共有 52,202 个参数。

## Release 部署

CMake 使用 `qt_generate_deploy_qml_app_script` 调用 Qt 官方部署工具。打包脚本将 Qt DLL、QML 模块、插件、模型、标签和演示图汇总到 `Release/`，并生成根目录 `qt.conf`、`VERSION.txt`、版本化 ZIP 与 SHA-256 校验文件。

数据集与权重被 `.gitignore` 排除；Qt 部署运行库同样不进入 Git，可由脚本稳定重建。
