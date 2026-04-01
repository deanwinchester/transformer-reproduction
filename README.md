# Transformer: Attention Is All You Need - 完整复现

> 🎯 从零复现 Google 2017 年经典论文，包含训练、测试、评估全流程

## 📊 性能基准对比

| 模型 | WMT14 En→De (BLEU) | 参数量 | 年份 |
|------|---------------------|--------|------|
| **Transformer (Base)** | **27.3** | 65M | 2017 |
| **Transformer (Big)** | **28.4** | 213M | 2017 |
| ConvS2S | 25.2 | - | 2017 |
| GNMT | 24.6 | - | 2016 |
| **当前 SOTA (GPT-4)** | **~35-38** | 1.8T | 2024 |
| **当前 SOTA (专用 NMT)** | **~32-34** | 1B+ | 2023-24 |

> 💡 **提升幅度**: 从原始 Transformer (28.4) 到当前 SOTA (~35)，提升了约 **6-7 BLEU (23-25% 相对提升)**

## 🗂️ 项目结构

```
transformer-reproduction/
├── 📁 src/                     # 核心代码
│   ├── model.py               # Transformer 模型实现
│   ├── attention.py           # 多头注意力机制
│   ├── embedding.py           # 词嵌入 + 位置编码
│   ├── decoder.py             # 解码器
│   ├── encoder.py             # 编码器
│   ├── train.py               # 训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── utils.py               # 工具函数
├── 📁 data/                    # 数据相关
│   ├── dataset.py             # 数据集加载
│   ├── tokenizer.py           # BPE 分词器
│   └── download_wmt14.py      # 数据下载脚本
├── 📁 configs/                 # 配置文件
│   ├── base_config.yaml       # Base 模型配置
│   └── big_config.yaml        # Big 模型配置
├── 📁 scripts/                 # 实用脚本
│   ├── train.sh               # 训练启动脚本
│   ├── evaluate.sh            # 评估脚本
│   └── visualize_attention.py # 注意力可视化
├── 📁 notebooks/               # Jupyter 笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_inference_demo.ipynb
├── 📁 checkpoints/             # 模型检查点
├── 📁 logs/                    # 训练日志
├── requirements.txt            # 依赖包
└── README.md                   # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python data/download_wmt14.py --lang-pair en-de --save-dir data/wmt14
```

### 3. 训练模型

```bash
# 训练 Base 模型（8卡 V100，约 12 小时）
python -m torch.distributed.launch --nproc_per_node=8 src/train.py \
    --config configs/base_config.yaml

# 或训练 Big 模型（8卡 V100，约 30 小时）
python -m torch.distributed.launch --nproc_per_node=8 src/train.py \
    --config configs/big_config.yaml
```

### 4. 评估模型

```bash
python src/evaluate.py --checkpoint checkpoints/transformer_base_best.pt \
    --config configs/base_config.yaml
```

## 📈 复现目标

| 指标 | 论文报告 | 本复现目标 |
|------|---------|-----------|
| WMT14 En→De (BLEU) | 27.3 (Base) / 28.4 (Big) | ≥ 26.5 / ≥ 27.5 |
| WMT14 En→Fr (BLEU) | 38.1 (Base) / 41.8 (Big) | ≥ 37.0 / ≥ 40.0 |
| 训练时间 (Base, 8×V100) | ~12 小时 | ~12-15 小时 |

## 🔬 技术细节

### 模型架构
- **Encoder**: 6 层，每层 = Multi-Head Attention + Feed Forward
- **Decoder**: 6 层，每层 = Masked Multi-Head Attention + Cross Attention + Feed Forward
- **Attention Heads**: 8 (Base) / 16 (Big)
- **Hidden Dim**: 512 (Base) / 1024 (Big)
- **FFN Dim**: 2048 (Base) / 4096 (Big)

### 训练配置
- **优化器**: Adam (β1=0.9, β2=0.98, ε=10^-9)
- **学习率调度**: warmup + 逆平方根衰减
- **Dropout**: 0.1
- **标签平滑**: 0.1
- **批次大小**: 约 25k tokens (Base), 35k tokens (Big)

## 📚 参考文献

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
2. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP
3. [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) - Google 官方实现

## 📝 许可

MIT License
