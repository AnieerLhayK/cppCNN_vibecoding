---
theme: seriph
transition: slide-left
title: 纯 C++ CNN 交通标志识别系统
mdc: true
lineNumbers: false
layout: cover
background: '#0A0F1D'
---

# 纯 C++ CNN 交通标志识别系统

43 类交通标志识别 · Qt Quick GUI · 95.26% 测试准确率

---
layout: center
transition: slide-left
---

# 我们做了什么

一套从数据读取、图像预处理、CNN 推理、模型加载到 GUI 展示的完整 C++ 交通标志识别系统

<div class="grid grid-cols-3 gap-6 mt-8">

<div class="card">
  <div class="card-icon">🚦</div>
  <div class="card-title">43 类</div>
  <div class="card-desc">完整 GTSRB 交通标志识别</div>
</div>

<div class="card">
  <div class="card-icon">🎯</div>
  <div class="card-title">95.26%</div>
  <div class="card-desc">测试集 Top-1 准确率</div>
</div>

<div class="card">
  <div class="card-icon">⚡</div>
  <div class="card-title">双击即用</div>
  <div class="card-desc">`run_demo.bat` 一键启动</div>
</div>

</div>

<style>
.card {
  background: linear-gradient(135deg, #1a2332 0%, #0f1628 100%);
  border: 1px solid #2a3a55;
  border-radius: 16px;
  padding: 28px 20px;
  text-align: center;
}
.card-icon {
  font-size: 2.8rem;
  margin-bottom: 12px;
}
.card-title {
  font-size: 1.6rem;
  font-weight: 700;
  color: #f3f6fd;
  margin-bottom: 6px;
}
.card-desc {
  font-size: 0.9rem;
  color: #92a1bc;
}
</style>

---
layout: center
transition: slide-left
---

# GUI 演示流程

现场展示：从启动到识别

```mermaid
graph LR
  A[run_demo.bat] --> B[Qt GUI 启动]
  B --> C[打开或拖入图片]
  C --> D[点击 Recognize]
  D --> E[Top-1 类别 + 置信度]
  E --> F[Top-3 概率条]
  style A fill:#4a3f8a,stroke:#6b5fcc,color:#fff
  style B fill:#2a3a55,stroke:#4a5a7f,color:#fff
  style C fill:#2a3a55,stroke:#4a5a7f,color:#fff
  style D fill:#6b5fcc,stroke:#8b7fec,color:#fff
  style E fill:#1a6b3a,stroke:#2a8b5a,color:#fff
  style F fill:#1a6b3a,stroke:#2a8b5a,color:#fff
```

<div class="note mt-6">
选中演示区图片会自动加载并识别，无需手动操作
</div>

<style>
.note {
  text-align: center;
  color: #92a1bc;
  font-size: 0.9rem;
}
</style>

---
layout: center
transition: slide-left
---

# 纯 C++ 技术边界

核心推理链路完全自主实现

<div class="grid grid-cols-2 gap-4 mt-6">

<div class="mod-box">
**Tensor** · Image Resize · Normalize
</div>
<div class="mod-box">
**Conv** · ReLU · MaxPooling
</div>
<div class="mod-box">
**Flatten** · **FC** · Softmax
</div>
<div class="mod-box">
**Loss** · **Trainer** · DataLoader
</div>

</div>

<div class="highlight-box mt-6">

**不依赖** PyTorch、TensorFlow 或 Keras

Qt 仅用于界面和图片解码，CNN 核心是纯 C++

</div>

<style>
.mod-box {
  background: #0f1628;
  border: 1px solid #2a3a55;
  border-radius: 12px;
  padding: 18px 16px;
  text-align: center;
  font-size: 1.05rem;
  color: #c8d1e3;
}
.highlight-box {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
  border: 1px solid #4a3f8a;
  border-radius: 14px;
  padding: 20px 24px;
  text-align: center;
  color: #bdb7f8;
  font-size: 1rem;
  line-height: 1.6;
}
</style>

---
layout: center
transition: slide-left
---

# LibTorch 训练加速

完整 43 类训练若仅靠 CPU 手写卷积耗时过长

<div class="grid grid-cols-2 gap-6 mt-6">

<div class="flow-box">
  <div class="flow-step">📦 LibTorch/CUDA</div>
  <div class="flow-arrow">⬇</div>
  <div class="flow-step">GPU 加速训练</div>
  <div class="flow-arrow">⬇</div>
  <div class="flow-step">导出 .bin 权重</div>
</div>

<div class="flow-box">
  <div class="flow-step">⚙️ Qt GUI</div>
  <div class="flow-arrow">⬇</div>
  <div class="flow-step">加载 .bin 模型</div>
  <div class="flow-arrow">⬇</div>
  <div class="flow-step">项目 C++ 推理链路</div>
</div>

</div>

<div class="highlight-box mt-6">

GPU **仅用于训练加速**，GUI 推理走的是项目自己的纯 C++ 卷积

</div>

<style>
.flow-box {
  background: #0f1628;
  border: 1px solid #2a3a55;
  border-radius: 14px;
  padding: 22px;
  text-align: center;
}
.flow-step {
  color: #c8d1e3;
  font-size: 1rem;
  padding: 6px 0;
}
.flow-arrow {
  color: #6b5fcc;
  font-size: 1.2rem;
  padding: 2px 0;
}
.highlight-box {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
  border: 1px solid #4a3f8a;
  border-radius: 14px;
  padding: 20px 24px;
  text-align: center;
  color: #bdb7f8;
  font-size: 1rem;
  line-height: 1.6;
}
</style>

---
layout: two-cols
transition: slide-left
---

::left::

## 我们负责

- 手写神经网络原型
- 架构决策与网络设计
- 代码审查与质量把关
- 实验设计与调参

::right::

## AI 辅助

- 扩展 CNN 各层模块
- 生成工程化代码
- 构建与调试自动化
- 文档与打包脚本

---
layout: center
transition: slide-left
---

> **我们是架构师和决策者，AI 是工程化执行者。**

---
layout: center
transition: slide-left
---

# 从手写原型到 CNN 工程化

<div class="grid grid-cols-1 gap-4 mt-6">

<div class="evo-row">
  <div class="evo-from">Neural::predict()</div>
  <div class="evo-arrow">→</div>
  <div class="evo-to">FCLayer::forward()</div>
  <div class="evo-desc">加权求和</div>
</div>

<div class="evo-row">
  <div class="evo-from">Neural::learn()</div>
  <div class="evo-arrow">→</div>
  <div class="evo-to">FCLayer::update()</div>
  <div class="evo-desc">梯度下降 / 增量更新</div>
</div>

<div class="evo-row">
  <div class="evo-from">隐层反传公式</div>
  <div class="evo-arrow">→</div>
  <div class="evo-to">FCLayer::backward()</div>
  <div class="evo-desc">误差反向传播</div>
</div>

</div>

<div class="highlight-box mt-6">

手写原型提供数学公式和架构蓝图，AI 负责模块化、工程化和扩展实现

</div>

<style>
.evo-row {
  display: flex;
  align-items: center;
  background: #0f1628;
  border: 1px solid #2a3a55;
  border-radius: 12px;
  padding: 14px 20px;
  gap: 16px;
}
.evo-from {
  color: #92a1bc;
  font-size: 0.95rem;
  font-family: monospace;
  min-width: 160px;
  text-align: right;
}
.evo-arrow {
  color: #6b5fcc;
  font-size: 1.3rem;
}
.evo-to {
  color: #c8d1e3;
  font-size: 0.95rem;
  font-family: monospace;
  min-width: 180px;
}
.evo-desc {
  color: #6f7f9b;
  font-size: 0.85rem;
  margin-left: auto;
}
.highlight-box {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
  border: 1px solid #4a3f8a;
  border-radius: 14px;
  padding: 20px 24px;
  text-align: center;
  color: #bdb7f8;
  font-size: 1rem;
  line-height: 1.6;
}
</style>

---
layout: center
transition: slide-left
---

# 四层质量审查

<div class="grid grid-cols-2 gap-4 mt-6">

<div class="qc-card">
  <div class="qc-icon">🧪</div>
  <div class="qc-title">ctest 持续回归</div>
  <div class="qc-desc">每次编译后自动运行单元测试</div>
</div>

<div class="qc-card">
  <div class="qc-icon">🔍</div>
  <div class="qc-title">公式逐行对照</div>
  <div class="qc-desc">核心数学推导与代码一一验证</div>
</div>

<div class="qc-card">
  <div class="qc-icon">🧮</div>
  <div class="qc-title">手算验证</div>
  <div class="qc-desc">极简输入逐层手工计算比对</div>
</div>

<div class="qc-card">
  <div class="qc-icon">📈</div>
  <div class="qc-title">训练收敛验证</div>
  <div class="qc-desc">Loss 下降与准确率上升趋势确认</div>
</div>

</div>

<div class="highlight-box mt-6">

AI 写得快，但我们审得严

</div>

<style>
.qc-card {
  background: #0f1628;
  border: 1px solid #2a3a55;
  border-radius: 14px;
  padding: 22px 20px;
  text-align: center;
}
.qc-icon {
  font-size: 2.2rem;
  margin-bottom: 8px;
}
.qc-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: #f3f6fd;
  margin-bottom: 4px;
}
.qc-desc {
  font-size: 0.85rem;
  color: #92a1bc;
}
.highlight-box {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
  border: 1px solid #4a3f8a;
  border-radius: 14px;
  padding: 20px 24px;
  text-align: center;
  color: #bdb7f8;
  font-size: 1rem;
  line-height: 1.6;
}
</style>

---
layout: center
transition: slide-left
---

# 模型迭代与最终成果

<div class="table-wrap mt-4">

| 版本 | 架构 | 增强 | 测试 Top-1 |
|:---:|:----:|:----:|:----------:|
| v1 | LeNet | — | 89.69% |
| v2 | LeNet | ✅ 数据增强 | 89.90% |
| v3 | **Enhanced** | — | **94.53%** |
| v4 | Enhanced | ✅ 全量增强 | 95.12% |
| **v5** | **Enhanced** | ✅ **全量增强** | **95.26%** |

</div>

<div class="highlight-box mt-6">

项目不是一次性生成的，而是逐步迭代、持续改进的过程

</div>

<style>
.table-wrap table {
  width: 100%;
  border-collapse: collapse;
  font-size: 1rem;
}
.table-wrap th {
  background: #1a2332;
  color: #92a1bc;
  font-weight: 600;
  padding: 10px 16px;
  border-bottom: 2px solid #2a3a55;
}
.table-wrap td {
  color: #c8d1e3;
  padding: 10px 16px;
  text-align: center;
  border-bottom: 1px solid #1a2332;
}
.table-wrap tr:last-child {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
}
.table-wrap tr:last-child td {
  color: #bdb7f8;
  font-weight: 700;
  border-bottom: none;
}
.highlight-box {
  background: linear-gradient(135deg, #1a1a3a 0%, #0f0f2a 100%);
  border: 1px solid #4a3f8a;
  border-radius: 14px;
  padding: 20px 24px;
  text-align: center;
  color: #bdb7f8;
  font-size: 1rem;
  line-height: 1.6;
}
</style>

---
layout: center
transition: slide-left
---

# 总结

<div class="grid grid-cols-1 gap-4 mt-6">

<div class="summary-row">
  <div class="summary-num">01</div>
  <div class="summary-text">
    完成了**可运行、可测试、可演示、可交付**的 C++ CNN 系统
  </div>
</div>

<div class="summary-row">
  <div class="summary-num">02</div>
  <div class="summary-text">
    理解了神经网络从**手写原型到工程项目**的演化过程
  </div>
</div>

<div class="summary-row">
  <div class="summary-num">03</div>
  <div class="summary-text">
    探索了**人做架构与把关，AI 做工程化执行**的开发模式
  </div>
</div>

</div>

<div class="final mt-8">
  谢谢大家，欢迎提问
</div>

<style>
.summary-row {
  display: flex;
  align-items: center;
  gap: 20px;
  background: #0f1628;
  border: 1px solid #2a3a55;
  border-radius: 14px;
  padding: 18px 24px;
}
.summary-num {
  font-size: 1.8rem;
  font-weight: 800;
  color: #6b5fcc;
  min-width: 44px;
  text-align: center;
}
.summary-text {
  font-size: 1.05rem;
  color: #c8d1e3;
  line-height: 1.5;
}
.final {
  text-align: center;
  font-size: 1.6rem;
  font-weight: 700;
  color: #bdb7f8;
}
</style>
