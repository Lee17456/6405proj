# 6405 项目方法与底层命令参考

最初版本：2026-10-19。当前操作入口已更新为 [完整实验方案与运行步骤](FULL_PROJECT_ZH.md)（2026-10-26），使用全量数据、单种子 42 与四个长度。请按新手册的 `proj6405.full` 顺序执行；本页保留方法细节和单项命令参考，不要将旧 `main_` 矩阵混入新的 `full_` 统计。

如果再次受到六小时额度限制，可另选 [六小时预算方案](SIX_HOUR_PLAN_ZH.md)。它不是当前默认方案。

本手册描述的是实施方案与验收标准。真实准确率、显存、吞吐量及运行时间必须在服务器运行后填入。项目代码已实现下载、训练、评估、基准测试与前端流程；本地功能验证不等于通过真实 CUDA、真实模型和完整数据训练验证。详见 `VALIDATION.md`。

## 1. 研究目标与课程对齐

题目建议：**资源受限条件下的长文本分类：传统方法、Transformer 与 Mamba 的分类质量及效率比较**。

目标用户：只有一张消费级 GPU、需要选择文档分类方案的开发者。

研究问题：

1. 在新闻、影评、法律文书上，不同方法的 Macro-F1 和训练成本如何？
2. 输入增加到 1K/2K/4K 时，Mamba 与长文本 Transformer 的延迟和显存怎样变化？
3. 更多上下文带来的准确率收益是否足以抵偿额外成本？

课程契合：传统 NLP 与深度学习比较、二分类和多分类、超参数及评估指标、模型结构解释、前端展示。截图中“三个或以上数据集”是建议，baseline paper 是可选；最终评分标准应再核对 syllabus rubric。项目没有必要复现 Mamba 原论文的大规模预训练，也不要求提出新架构。

本项目不能单凭现成预训练模型的成绩推断“SSM 架构普遍优于 Transformer”：预训练目标、语料、分词器都不同。Mamba 原论文预印本为 2023 年 12 月，会议版为 COLM 2024；本实验研究原始 Mamba-130M，不声称它是当前最新模型。

## 2. 固定范围与模型

| 简称 | 下载来源 | 项目实现 |
|---|---|---|
| svm | 无预训练模型下载 | TF-IDF + LinearSVC，CPU，C=1，unigram+bigram |
| roberta | FacebookAI/roberta-base | 预训练主干 + 首位置池化 + dropout + 线性分类头 |
| mamba | state-spaces/mamba-130m-hf | 预训练主干 + 最后有效位置池化 + dropout + 线性分类头 |
| longformer | allenai/longformer-base-4096 | 首位置全局注意力 + 首位置池化 + 线性分类头 |

三个神经模型均监督微调全部主干参数，没有使用语言模型生成分类标签；Mamba 不计算整词表的语言建模输出。分类头是本项目代码实现，不是下载后天然具备任务能力。RoBERTa 默认使用 SDPA，配置会记录后端；Longformer 使用其稀疏注意力实现。不得把不同实现的速度差全部归结为数学复杂度。

先不增加 LSTM、370M、Mamba-2/3 或多标签任务。它们会增加实验量，但不能替代公平的长文本对照。

## 3. 数据、标签和划分

| 数据 | 官方来源 | 本项目划分 |
|---|---|---|
| AG News | fancyzhx/ag_news | 原 train 的 90% 训练、10% 分层验证；原 test 保留，即 108,000 / 12,000 / 7,600 |
| IMDb | stanfordnlp/imdb | 22,500 / 2,500 / 25,000；只使用有标签部分，不用 unsupervised |
| SCOTUS | coastalcph/lex_glue，config=scotus | 保持官方 5,000 / 1,400 / 1,400 时间划分 |

AG News 为四分类，IMDb 为二分类。**SCOTUS 的页面正文写 14 类，但当前发布元信息列出 13 个类别编码。代码从下载数据的 ClassLabel 动态读取标签数、保存标签映射，报告以实际 manifest 为准。** 不把类别数写死，也不自行给法律编码配未经核对的名称。

分层划验证集和小样本抽样固定使用 data seed=2026；模型训练 seed=42。不同模型种子不改变样本集合。词表和 IDF 只在训练集拟合。下载记录固定 repo commit；已有完整数据会复用，不会重新随机切分。

脚本会记录跨划分的完全相同文本数量，不会擅自删除官方测试样本。如果存在重复，应在报告中披露；需要研究去重影响时另建并命名去重实验，不覆盖原始基准。

## 4. 两种输入协议

### 4.1 native：实际应用方案比较

- SVM 读取完整原文。
- 神经模型读取原文，并按自己的 tokenizer 截断到训练配置的 `max_length`。
- 因此这组是在比较实际部署策略，不能称为“所有模型看到相同文字”。

### 4.2 shared512 / shared1024 / shared2048 / shared4096：共同内容对照

脚本寻找同一文档的共同原文前缀，使它在参与比较的每个 tokenizer 下都不超过预算，然后把完全相同的字符串保存给各模型。它不会把不同 tokenizer 的“前 512 tokens”误当作相同内容。

- 512 及以下：RoBERTa、Mamba、Longformer 的 tokenizer 一起约束前缀。
- 超过 512：仅用 Mamba、Longformer 的 tokenizer；不扩展 RoBERTa 的位置嵌入。
- SVM 可以在同一视图上训练，得到严格共享输入的传统基线。
- 保存覆盖的字符比例。这个比例描述原文覆盖率，不是语义信息比例。
- 不同长度预算的实验分别训练模型，训练和测试使用相同上限。本代码不把“512 训练后直接外推 4K”混为“4K 训练”。

主实验使用 shared512，全文 SVM 作为附加实用基线；长文本专项使用 shared2048 等。跨领域数据不能单独证明长度因果关系；长度变化主要在同一 SCOTUS 测试集内分析。

## 5. 本地代码上传到 Linux

假设远程目录为 `/home/YOUR_USER/6405proj`，SSH 端口为 22；替换为实际值。无需在 Windows 下载任何数据集或预训练模型。

本地 PowerShell 执行：

```powershell
Set-Location 'D:\11\algorism\skealran\natural_nlp\6405_study'
# 只打包源码；排除本地验证依赖、下载内容、训练产物。
tar -czf 6405proj-code.tar.gz --exclude='6405proj/local_test_deps' --exclude='6405proj/.venv' --exclude='6405proj/.conda-env' --exclude='6405proj/.conda-executable' --exclude='6405proj/.cache' --exclude='6405proj/__pycache__' --exclude='*/__pycache__' --exclude='6405proj/.pytest_cache' --exclude='6405proj/data' --exclude='6405proj/models' --exclude='6405proj/runs' --exclude='6405proj/reports' --exclude='6405proj/logs' 6405proj
scp -P 22 .\6405proj-code.tar.gz YOUR_USER@SERVER_IP:~/
ssh -p 22 YOUR_USER@SERVER_IP
```

第一次部署在 Linux 执行：

```bash
cd ~
tar -xzf 6405proj-code.tar.gz
cd ~/6405proj
nvidia-smi
conda --version
command -v tmux
command -v nvcc
```

后续更新先保留服务器上的数据、models、runs；上传包不包含它们。不要在训练正在运行时覆盖正在使用的源码。

## 6. 环境安装与 CUDA 检查

推荐 Linux x86_64、Python 3.11、RTX 3090 24GB，预留至少约 50GB 磁盘用于环境、数据和多种子检查点；完整长度矩阵可能需要更多。系统 RAM 建议 32GB 起。实际资源占用以运行记录为准。

以下安装版本是为该项目固定的兼容基线，不是“安装所有库最新版”：PyTorch 2.5.1/cu121、Transformers 4.48.3、Mamba-SSM 2.2.4、causal-conv1d 1.5.0.post8。服务器驱动必须支持所选 CUDA runtime。

```bash
cd ~/6405proj
bash scripts/setup_conda.sh
bash scripts/install_mamba.sh
source scripts/activate.sh
python -m pip check
python -m proj6405.doctor
```

环境统一使用 Conda，安装脚本创建项目目录下的 `.conda-env`，Python 固定为 3.11；无需系统预装 Python 3.11，也无需手工修改 API/UI 启动脚本。每次新终端执行 `source scripts/activate.sh`。没有 Conda、使用非默认安装位置或遇到激活问题时，参考 [Conda + RunPod 指南](CONDA_RUNPOD_ZH.md)。

对于有 sudo 权限的 Ubuntu/Debian 服务器，可以先安装基础工具（不安装或更换 GPU 驱动）：

```bash
sudo apt-get update
sudo apt-get install -y build-essential tmux
bash scripts/setup_conda.sh
```

无 sudo 权限时使用服务器已有工具或请管理员提供；CUDA toolkit 的安装位置由服务器环境决定，不能仅凭这份文档推断 `/usr/local/cuda` 一定正确。

`nvidia-smi` 显示的 CUDA 版本是驱动能力，`torch.version.cuda` 是 PyTorch runtime，`nvcc --version` 才是本机编译 toolkit。Mamba 没有匹配 wheel 而转源码编译时，需要 CUDA toolkit、g++、ninja。对于当前 cu121 基线，优先使用 CUDA 12.1 toolkit；学校环境可先 `module load` 匹配模块，再安装扩展。不要为了项目擅自升级服务器驱动。

安装成功后的 `reports/conda-explicit-linux64.txt`、`reports/environment_full.txt` 与 `reports/environment.json` 都要保留。`doctor` 中 `mamba_fast_path` 应为 true；只有导入成功而快速路径未启用，不足以开展性能比较。

## 7. 下载数据集和预训练模型（全部在 Linux）

```bash
source scripts/activate.sh
python -m proj6405.download
```

预计产生：

```text
data/ag_news/dataset/           # 固定 train/validation/test
data/ag_news/manifest.json
data/imdb/dataset/
data/scotus/dataset/
models/roberta/
models/mamba/
models/longformer/
models/*/download_manifest.json # 下载来源及精确 commit
```

只下载数据，或只下载指定模型：

```bash
python -m proj6405.download --models
python -m proj6405.download --datasets --models mamba
```

空的 `--models` 表示不下载模型；空的 `--datasets` 表示不下载数据。默认下载不需要 Hugging Face 付费服务。这些公开源通常不需要登录；如服务器网络要求代理或校方镜像，按服务器网络配置处理，不把访问令牌写入源码。

下载器优先 safetensors。Longformer 官方仓库目前提供 PyTorch `.bin`，脚本会下载对应官方权重而不下载 TensorFlow/Rust 的副本；训练后本项目统一保存为 safetensors。

第一次下载完成后，后续训练和演示只读本地文件：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

需要补下载时，先 `unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE`。下载器检查到 manifest 会复用已有文件；若你手工删坏了文件，先检查并重命名该模型目录，再重新下载，不能只保留一份错误的 manifest。

## 8. 先验证 GPU、分类头和短流程

```bash
python -m proj6405.doctor --smoke roberta --length 512
python -m proj6405.doctor --smoke mamba --length 512
python -m proj6405.doctor --smoke longformer --length 512
python -m pytest -q
```

验收：三个实际主干都完成前向、损失计算和反向传播，loss/gradient 有限。Smoke 只有一次前反向且没有优化器状态，不能拿它的耗时或显存当正式训练 benchmark。

随后做小样本 pilot：

```bash
python -m proj6405.matrix --stage pilot --phase train --prefix pilot --seeds 42
python -m proj6405.matrix --stage pilot --phase evaluate --prefix pilot --seeds 42
```

这会在 AG News 的固定 1,000 个训练样本、200 个验证样本上训练一轮三个方法，神经输入上限 256；测试只取 200 个样本。目录为：

```text
runs/pilot_ag_news_svm_native_s42/
runs/pilot_ag_news_roberta_native_s42/
runs/pilot_ag_news_mamba_native_s42/
```

得到 `metrics_test_subset200.json` 说明流程跑通，不能声称完成了整个测试集。前端实时预测可以使用这些 pilot 模型，但完整成绩页不会把子集成绩当作正式结果。

## 9. 描述数据长度并准备共同输入

```bash
# 快速查看分布；正式报告可去掉 --limit 2000，对全部训练集统计。
python -m proj6405.analyze_data --limit 2000
for dataset in ag_news imdb scotus; do
  python -m proj6405.views --dataset "$dataset" --budget 512
done
python -m proj6405.views --dataset scotus --budget 2048
```

输出长度均值、中位数、P90、最大值，以及超过 512/1K/2K/4K 的比例。先根据这些实际数字判断“短/中/长”，不要假定所有 IMDb 都长、所有法律任务都需要长程推理。

构建视图需要遍历原文并分词，尤其 SCOTUS 首次会耗时，但只需生成一次。视图保存在 `data/views/<dataset>/shared<budget>/`；检查点记录视图签名，不能换了视图后继续沿用原检查点成绩。

## 10. 超参数探索：先用验证集决定配置

默认参数仅是起点。建议神经模型分别尝试学习率 `1e-5` 与 `2e-5`，如果 Mamba 验证损失不稳定再降低学习率；每种模型使用相近搜索预算。SVM 比较 C=0.1/1/10 和 unigram/bigram，可只在一个数据集详细展示敏感性，避免扩大所有组合。

示例：

```bash
python -m proj6405.train --model mamba --dataset imdb --view shared512 --max-length 512 --name tune_imdb_mamba_lr1e5 --lr 1e-5 --seed 42 --epochs 3 --limit-train 5000 --limit-val 1000
python -m proj6405.train --model mamba --dataset imdb --view shared512 --max-length 512 --name tune_imdb_mamba_lr2e5 --lr 2e-5 --seed 42 --epochs 3 --limit-train 5000 --limit-val 1000
python -m proj6405.train --model svm --dataset imdb --view shared512 --name tune_imdb_svm_c01 --svm-c 0.1
```

查看各目录的 `validation.json` 和神经模型 `history.jsonl`。调参阶段不要运行 test。子集调参只是配置筛选，最终配置仍要完整训练。默认 matrix 对所有神经模型用 `2e-5`；若发现各模型最佳学习率不同，可使用单模型命令逐个执行，并在报告记录公平的搜索预算，不强求同一学习率。

## 11. 正式三数据集实验

固定配置后：

```bash
python -m proj6405.matrix --stage core --phase train --prefix main --dry-run
python -m proj6405.matrix --stage core --phase train --prefix main
```

默认：3 epochs、micro-batch=2、gradient accumulation=8、有效 batch 约 16、bf16、AdamW、梯度裁剪 1、10% warmup。最后一个不足量 batch 不一定满 16。训练 FP32 主权重并使用 autocast，便于 Mamba 稳定训练；不直接把整个模型转 half 后训练。

默认 core 共 12 次训练：6 次神经模型（2 模型 × 3 数据集 × seed 42）+ 6 次 SVM（3 数据集 × shared512/native）。所有配置只用一个训练种子，不估计跨种子标准差。

单独训练命令举例：

```bash
python -m proj6405.train --model roberta --dataset ag_news --view shared512 --max-length 512 --seed 42 --name custom_ag_roberta_s42 --lr 2e-5
python -m proj6405.train --model mamba --dataset scotus --view shared512 --max-length 512 --seed 42 --name custom_scotus_mamba_s42 --lr 1e-5
python -m proj6405.train --model svm --dataset scotus --view native --name custom_scotus_svm_full --svm-c 1 --ngram-max 2
```

开始长时间任务时，先开启 tmux：

```bash
tmux new -s proj6405-train
cd ~/6405proj
source scripts/activate.sh
set -o pipefail
python -m proj6405.matrix --stage core --phase train --prefix main 2>&1 | tee logs/main_train.log
```

按 Ctrl+B，松开，再按 D，离开会话；训练继续。重新查看：

```bash
tmux attach -t proj6405-train
tail -n 60 logs/main_train.log
nvidia-smi
```

同名 run 不会被覆盖；matrix 只跳过参数匹配且 completed 的 run。失败或被中断的 run 保留记录，但当前实现**不保存完整优化器状态，不支持从中断 step 恢复**。修复后用新 run 名重训该任务；不要在报告中把 partial run 当 completed。tmux 不能保证服务器重启、管理员作业超时后继续运行；有 Slurm 等调度器时须按平台方式提交 GPU 作业。

## 12. 长文本专项与消融

先做 2K，确认资源可承受：

```bash
python -m proj6405.views --dataset scotus --budget 2048
python -m proj6405.matrix --stage long --phase train --prefix main --lengths 2048 --seeds 42
```

默认长输入启用 gradient checkpointing、验证 batch=1。如果 24GB 显存不足，可改为 micro-batch=1、accumulation=16（有效 batch 仍约 16），并使用新 prefix：

```bash
python -m proj6405.matrix --stage long --phase train --prefix long_smallbatch --lengths 2048 --seeds 42 --batch-size 1 --grad-accum 16
```

有余力时完成长度曲线：

```bash
for length in 512 1024 2048 4096; do
  python -m proj6405.views --dataset scotus --budget "$length"
done
python -m proj6405.matrix --stage long --phase train --prefix long --lengths 512 1024 2048 4096 --seeds 42 --batch-size 1 --grad-accum 16
```

这是 8 次额外训练。当前默认只用 seed 42。不同 prefix 的同一预算必须核对训练样本、学习率、有效 batch 等配置后才比较。

最低限度选一个消融：Mamba last pooling 对比 mean pooling：

```bash
python -m proj6405.train --model mamba --dataset scotus --view shared2048 --max-length 2048 --name ablation_mamba_mean_s42 --pooling mean --seed 42 --batch-size 1 --grad-accum 16 --eval-batch-size 1 --gradient-checkpointing
```

对照的 last pooling run 需使用相同其余参数。报告可讨论为何较长前缀没有提高 F1，但不要仅从分类分数认定模型已具备或缺乏一般长程推理能力。

## 13. 测试集评估

在配置冻结后运行：

```bash
python -m proj6405.matrix --stage core --phase evaluate --prefix main
python -m proj6405.matrix --stage long --phase evaluate --prefix main --lengths 2048 --seeds 42
# 如果运行的是 long 前缀的完整长度曲线，用对应命令：
python -m proj6405.matrix --stage long --phase evaluate --prefix long --lengths 512 1024 2048 4096 --seeds 42
```

只执行与你实际训练计划匹配的行。单模型评估：

```bash
python -m proj6405.evaluate --run main_scotus_mamba_shared512_s42 --batch-size 1
```

每个 run 生成 `metrics_test.json`、`predictions_test.csv`。Macro-P/R/F1 固定按 manifest 全部类别计算，缺失类别的分数为 0；不是只对预测出现过的类别平均。类别不均衡时分析每类表现。不同数据集分别报告，不把二分类和多分类 F1 随意平均成一个排名。

`--limit` 用于排错，其输出另存 `metrics_test_subsetN.json`，不会覆盖完整测试文件。不要用 test 反复调参；错误分析用于解释已冻结实验结果。

## 14. 正式性能测量

先停止演示服务，确保同一 GPU 没有训练或其他模型进程竞争。若 GPU 是共享服务器，记录当时是否独占资源。

```bash
bash scripts/stop_services.sh
python -m proj6405.matrix --stage core --phase benchmark --prefix main --benchmark-batch 1
python -m proj6405.matrix --stage long --phase benchmark --prefix main --lengths 2048 --seeds 42 --benchmark-batch 1
```

批处理吞吐量另做一组：

```bash
python -m proj6405.benchmark --run main_scotus_mamba_shared2048_s42 --batch-size 4 --samples 128 --warmup 10 --repeats 3
python -m proj6405.benchmark --run main_scotus_longformer_shared2048_s42 --batch-size 4 --samples 128 --warmup 10 --repeats 3
```

若某模型 batch=4 OOM，记录该失败；可以另用双方都能运行的 batch=1/2 做匹配比较。不能静默为一个模型改 batch 后继续宣称同一条件。

测量定义：

- `model_ms_per_document`：所有 batch 前向耗时总和 / 总文档数；SVM 指稀疏特征后的决策计算。
- `pipeline_documents_per_second`：含分词或 TF-IDF、CPU→GPU、前向、argmax；不含模型加载、网络和网页。
- `batch_latency_p50_ms/p95_ms`：batch 延迟。只有 batch=1 才能解释为单文档延迟分位数。
- `input_tokens_per_second`：实际非 padding 输入 token（含特殊 token）的吞吐，非生成速度；跨 tokenizer 仅作辅助。
- `peak_allocated_mb`：PyTorch 已分配张量峰值，包含模型常驻权重；`peak_reserved_mb` 是缓存分配器保留峰值。单位实际为 MiB。
- 这些显存值不是 `nvidia-smi` 的整卡总占用，不能代表其他进程或所有驱动工作区的开销。
- SVM 的 GPU 显存为 null；另记录进程测量结束时 RSS，**不是 CPU 峰值内存**。
- 默认动态 padding；Longformer 还会按 attention window 内部补齐，实际计算成本反映在计时中。

代码先热身，再在转移与前向边界同步 CUDA；下载和模型加载不计入模型计时。默认固定、分层抽取 128 个测试样本并重复 3 次，记录样本 ID 签名。它是该样本集合的性能结果，不是对全部测试集延迟分布的保证。

共同前缀已经作为实验数据离线准备，benchmark 不计其构建成本；网页收到新文本时需实时构建共同前缀，该成本计入页面预处理列。不要把离线 benchmark 解释成包含所有在线输入处理的生产服务 SLA。

`train_seconds` 对所有方法采用同一阶段边界：从模型构建/本地加载开始，包含特征准备、训练、验证和写检查点，排除数据下载和初始读取。另保存 SVM 的 fit 秒数和神经模型的 epoch/validation/checkpoint 秒数，但这两个细分值定义不同，不直接互相比较。

## 15. 汇总、图表和存档

```bash
python -m proj6405.report --prefix main_
python -m proj6405.demo_samples --dataset ag_news
python -m proj6405.demo_samples --dataset imdb
python -m proj6405.demo_samples --dataset scotus
```

输出：

- `reports/results.csv`：每次完整测试结果。
- `reports/summary.csv`：相同配置、不同 seed 的均值和样本标准差。只有一个 seed 时 std 留空，不填虚假的 0。
- `reports/benchmarks.csv`：各 batch / precision 的效率记录。
- `reports/f1_*.png`、`confusion_*.png`、`scaling_*.png`：用于报告和录屏。
- `reports/length_f1_scotus.png`、`quality_cost_*.png`：有相应真实结果时导出长度—F1 与质量—成本散点图。
- `reports/demo_samples/`：固定前几个官方测试文档、ID 和真实标签。

report 的 CSV 使用固定名称，每次运行会刷新为当前 prefix 所选结果；需保留多个 campaign 时先复制 reports 到不同归档目录。没有训练结果时不生成假的分数。不能把试验配置、pilot 和最终模型直接合并求均值。完全重复的配置与 seed 会被 report 拒绝，要求你用 prefix 选择正确的一组。

最终备份代码、模型/数据 manifest、各 run config、Conda 清单、pip freeze、metrics、逐条预测、benchmark 原始 JSON、图表、成员贡献。训练检查点按交付空间决定是否另存；不要提交 `.conda-env`、旧 `.venv` 或整个下载缓存。

## 16. 服务器端前后端启动

```bash
cd ~/6405proj
bash scripts/start_services.sh
tmux ls
curl http://127.0.0.1:8000/health
tail -n 30 logs/api.log
tail -n 30 logs/ui.log
```

启动脚本创建 `proj6405-api` 和 `proj6405-ui` 两个 tmux 会话。API 在 8000，Streamlit 在 8501，均绑定服务器回环地址。首次启动后需等待导入完成再查 health。`status=ok` 说明服务可达，不等于每个模型都通过推理验证。

不用 tmux 时可在两个服务器终端分别运行：

```bash
# 终端 A
source scripts/activate.sh
python -m uvicorn proj6405.api:app --host 127.0.0.1 --port 8000 --workers 1
```

```bash
# 终端 B
source scripts/activate.sh
API_BASE_URL=http://127.0.0.1:8000 python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

必须一个 API worker。锁保证推理串行；服务一次缓存一个模型，切换时释放前一个，单独记录加载和预热时间。该策略优先控制 3090 显存。反复在多个模型间切换会有加载等待，正式效率图取离线 benchmark，而非页面总等待时间。

## 17. 本地访问、演示及停止

本地 PowerShell：

```powershell
ssh -N -L 8501:127.0.0.1:8501 -L 8000:127.0.0.1:8000 -p 22 YOUR_USER@SERVER_IP
```

浏览器访问 `http://127.0.0.1:8501`。隧道窗口一直保留。只需要本地浏览器，不需要本地 GPU 或 Python 环境。老师若要独立访问，需要自己的授权 SSH/VPN 或另行部署可访问地址；当前方案适合你本人答辩与录屏。

停止服务：

```bash
bash scripts/stop_services.sh
```

本地隧道按 Ctrl+C 停止。不会删除检查点或实验结果。具体录屏步骤见 `DEMO_GUIDE_ZH.md`。

## 18. 预计结果与时间规划

预计应得到的是完整产物，不是保证某个模型赢：三数据集分类指标、长度统计、至少一个长文本匹配对照、延迟/显存实测、模型错误案例和可操作界面。

可以提出但必须验证的假设：SVM 在短文本中可能有很强的性价比；Mamba 的长输入显存增长可能较温和，但 130M 分类 F1 未必优于 RoBERTa；Longformer 可能从更长上下文获益，也可能因为任务主要依赖局部线索而收益有限。Mamba 能接收 4K 输入不保证在该长度有高质量表示。若多读文本没有收益，也应如实报告。

不能预填“加速 5 倍”“节省 80% 显存”或任意 F1 数字。可以用同任务、同样本、同 batch 的实测比值计算 speedup = 对照耗时 / Mamba 耗时，并附实验条件。

建议进度：第 1 阶段完成安装、下载和 pilot；第 2 阶段完成验证集调参与 core；第 3 阶段完成长文本、消融、评估和基准；第 4 阶段完成界面、图表、录屏及交付。不要承诺四阶段各需固定 GPU 小时。先跑一个有代表性的 epoch，根据日志的 elapsed_seconds 估算完整预算，据此安排 4K 单种子实验。

截图写的截止时间是 Sunday 15 November 11:59pm，未标注年份；以当前课程通知为准。先留出录屏、组员签名和上传的时间，不把最后一天用于首次安装 CUDA。

## 19. 常见问题处理

| 现象 | 操作 |
|---|---|
| Mamba CUDA extension 编译失败 | 核对 Python/torch/CUDA toolkit，先确认 `nvcc` 和 g++；保留安装日志；不要一边测试一边更换所有库版本 |
| doctor 快速路径为 false | 先修复 mamba-ssm / causal-conv1d；`--allow-slow-mamba` 只用于功能排错，不能把慢路径速度当架构结论 |
| 显存 OOM | 确认无并行服务；batch 改 1，调整 accumulation，启用检查点，先降到 1K/2K；新配置用新 run 名 |
| loss 为 NaN/Inf | 保持 FP32 主权重，检查 bf16，降低学习率；必要时 fp32 小样本定位；不要忽略失败继续输出分数 |
| 下载超时 | 保持缓存后重试；确认远程网络；没有 manifest 的半成品 dataset 先检查并重命名后重建 |
| 同名目录存在 | completed 的匹配任务由 matrix 跳过；手工 train 不覆盖；新试验用新 name/prefix |
| 网页连接失败 | 先在服务器 curl health，检查 tmux/log，再核对本地 SSH 隧道；服务器上的 127.0.0.1 与本地不是同一台机器 |
| 第一条预测特别慢 | 看加载/预热列；这不属于 model_ms；同一模型第二次请求通常无需重新加载 |
| 图表页没有成绩 | 是否只做了 subset 评估？是否真的产生 `metrics_test.json`？ |
| 全部模型预测错误 | 先检查所选任务、标签映射、是否仅 pilot 训练、输入是否英文；不要将中文任意文本当英文任务效果测试 |

## 20. 参考来源

- Mamba 原论文：https://arxiv.org/abs/2312.00752
- COLM 2024 版本：https://openreview.net/pdf?id=tEYskw1VY2
- 固定 Mamba 版本说明：https://github.com/state-spaces/mamba/tree/v2.2.4
- Transformers Mamba 文档：https://huggingface.co/docs/transformers/v4.48.2/model_doc/mamba
- Longformer：https://huggingface.co/allenai/longformer-base-4096
- AG News：https://huggingface.co/datasets/fancyzhx/ag_news
- IMDb 原始数据：https://ai.stanford.edu/~amaas/data/sentiment/
- LexGLUE 数据与标签元信息：https://huggingface.co/datasets/coastalcph/lex_glue
- LexGLUE 论文：https://aclanthology.org/2022.acl-long.297/
- 长法律文档研究：https://aclanthology.org/2022.nllp-1.11/
- PyTorch 历史安装版本：https://pytorch.org/get-started/previous-versions/
- PyTorch CUDA 同步与显存：https://docs.pytorch.org/docs/stable/notes/cuda.html
- FastAPI 多进程内存：https://fastapi.tiangolo.com/deployment/concepts/
- Streamlit 配置：https://docs.streamlit.io/develop/api-reference/configuration/config.toml
