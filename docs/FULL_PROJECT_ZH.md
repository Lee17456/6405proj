# 完整实验方案：Mamba 与 Transformer 的文本分类质量和计算成本

版本：2026-10-26。当前主方案；取消六小时小样本限制，继续使用 **RunPod 单张 RTX 3090 + Linux + Conda + FastAPI + Streamlit**。本文件是从安装到提交的主操作顺序，旧六小时文档仅保留为备用方案。

这是待执行的实验设计与代码使用说明，不是已经完成的实验报告。真实分数、训练时长、吞吐和显存都需在服务器上测量。

## 1. 目的、研究问题和最终产物

面向只有单张 GPU、需要选择文档分类方案的开发者，回答：

1. 在新闻、影评和法律文书上，TF-IDF + SVM、RoBERTa 和 Mamba 的分类效果如何？
2. 同一批法律文档的输入预算从 512 增加到 1024、2048、4096 时，Mamba 与 Longformer 的质量、耗时和显存怎样变化？
3. 更多上下文带来的效果收益，是否值得增加的计算成本？

项目交付为：可复现 Python 源码、三个数据集的评估表、长文本长度分析、效率与显存图、错误案例、可操作网页，以及不超过课程要求时长的讲解视频。

与附件中的课程方向对齐：传统 NLP 与深度学习对照、二分类和多分类、超参数分析、Precision/Recall/F1、模型理论解释，以及 Streamlit 产品展示。三个以上数据集在截图中是建议，baseline paper 标为可选；正式评分细则仍以 syllabus rubric 为准。截图是任务背景资料，不改变本项目的执行授权。

研究对象是**现有预训练模型的监督微调和部署表现**。预训练语料、目标、参数量、分词器和内核实现都有差异，所以结果不能单独证明“状态空间架构必然优于注意力”。不复现 Mamba 原论文的大规模从零预训练。

## 2. 方法和模型

| 方法 | 实现 | 在实验中的作用 |
|---|---|---|
| TF-IDF + Linear SVM | word unigram/bigram、sublinear TF、线性 SVM | 传统方法；共享输入基线和全文应用基线 |
| RoBERTa-base | `FacebookAI/roberta-base`，首位置表示接分类头 | 标准 Transformer，最多 512 输入 tokens |
| Mamba-130M | `state-spaces/mamba-130m-hf`，最后有效位置表示接分类头 | 选择性状态空间模型；512 与长输入对照 |
| Longformer-base | `allenai/longformer-base-4096`，首位置 global attention 与分类头 | 稀疏注意力长文本 Transformer，对照到 4096 |

神经模型均微调主干和分类头。下载的是预训练模型，分类头通过本项目任务训练获得。默认不加入 LSTM、370M 或更多架构，完整程度来自控制变量、重复实验和分析，不靠无限增加模型。

实现依据见 [Transformers 4.48 Mamba 文档](https://huggingface.co/docs/transformers/v4.48.0/en/model_doc/mamba)；本项目固定安装 4.48.3，并通过 doctor 检查 CUDA 快速路径。

理论讲解应说明：TF-IDF 如何把文本转成稀疏特征；自注意力如何建立 token 关系；Mamba 如何用输入相关的状态更新处理序列；Longformer 如何用局部窗口和少量全局注意力处理长文。

Mamba 论文 arXiv 首发于 2023 年，COLM 发表版本为 2024 年。引用时区分版本：[原论文](https://arxiv.org/abs/2312.00752)、[COLM 2024](https://openreview.net/pdf?id=tEYskw1VY2)。

## 3. 三个数据集与划分

| 数据集 | 任务 | 本项目训练 / 验证 / 测试数量 | 类别 |
|---|---|---|---|
| AG News | 新闻主题分类 | 108,000 / 12,000 / 7,600 | 4 类 |
| IMDb | 电影评论情感分类 | 22,500 / 2,500 / 25,000 | 2 类 |
| LexGLUE SCOTUS | 法律文书主题分类 | 5,000 / 1,400 / 1,400 | 以下载 manifest 的标签表为准 |

AG News、IMDb 从官方训练集按数据种子 2026 分层取 10% 作验证集；官方测试集不变。IMDb 的无标签部分不参与训练。SCOTUS 保持官方时间划分，不能重新随机打乱成新的训练/测试划分。训练随机种子固定为 42；数据划分种子 2026 不变。

数据来源：[AG News](https://huggingface.co/datasets/fancyzhx/ag_news)、[IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)、[LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue)。法律类别数不要照抄论文描述或网页正文，实际标签元数据和下载版本会写入 manifest。

## 4. 公平比较协议

- 对参与某个长度比较的每个 tokenizer，寻找都能装入预算的**同一段原文前缀**，保存为 `shared512/shared1024/...`。相同预算不代表各分词器实际 token 数相同。
- 512 及以下的前缀同时满足 RoBERTa、Mamba、Longformer；更长预算同时满足 Mamba 和 Longformer。
- 每个模型都在对应长度训练并在同长度测试，不把“512 训练直接外推到 4K”当作“4K 微调”。
- 核心共享输入 SVM 也读取同一前缀。另保留 `native` 全文 SVM，独立标注输入优势。
- 核心比较使用相同全量训练/验证样本；长度实验在同一个 SCOTUS 测试集上进行，避免把领域差异误认为长度效应。
- 记录精度、实际输入 tokens、batch、GPU、CPU 和加速内核。速度比较主要用 ms/document 与 documents/s，input tokens/s 只是辅助，**不是生成 tokens/s**。

## 5. 完整实验矩阵

默认配置在 `configs/full.json`，第一次正式运行后连同数据/模型来源写入 `reports/full/plan.json` 并冻结。

当前按用户要求只用训练 seed 42。服务器若已上传旧版，仅需将该 JSON 的 `seeds` 改为 `[42]`；已有三种子正式计划冻结时，后续命令统一使用新的 `--prefix full_s42`，不要删除旧计划或在运行中修改配置。3 epochs 和推理 benchmark 的 3 次计时重复均保留。

| 实验组 | 设置 | 训练次数 |
|---|---|---:|
| 核心：SVM | 3 数据集 × shared512/native 两种输入，seed 42 | 6 |
| 核心：RoBERTa、Mamba | 3 数据集 × 2 模型 × seed 42，shared512 | 6 |
| 长文本：Mamba、Longformer | SCOTUS × 2 模型 × 4 长度 × seed 42 | 8 |
| 合计 | 6 次 SVM、14 次神经微调 | **20** |

神经模型默认 3 epochs，学习率 2e-5，AdamW、weight decay=0.01、warmup=10%、梯度裁剪=1、bf16 autocast，主权重保留 FP32。验证 Macro-F1 选择最佳 epoch，测试集不选模型。

- 核心：micro-batch=2，累积=8，有效 batch 约 16。
- 长度组所有长度：micro-batch=1，累积=16，统一开启 gradient checkpointing。512 也使用同样设置，减少长度组内部训练配置差异。
- 两组的 Mamba-512 **分别训练**，保留不同实验组标识，不合并为同一配置的重复实验。
- 所有配置只使用 seed 42 训练一次，跨种子标准差留空。
- 3 epochs 是固定起始训练预算，不保证每个模型已充分收敛；报告需检查训练/验证曲线。

默认完成后应有 20 份完整测试指标；主效率测试只用 seed 42，每个配置热身 10 次，固定 128 个文档重复 3 次，共 **20 份 batch=1 benchmark**。计时重复仍保留 3 次；这是推理测量，不会重复训练。

## 6. 环境、持久化存储与上传

使用 Linux x86_64 GPU Pod；设置 HTTP 端口 8501，确认 `/workspace` 挂载持久化存储。环境与检查点需要磁盘空间，完整方案可先按约 100 GB 可用空间规划，再用 `df -h`、`du -sh` 检查实际占用。不同存储类型在停止/删除 Pod 后的保留规则见 [RunPod 存储说明](https://docs.runpod.io/pods/storage/types)。

代码在本地编写，数据和权重全部在 Linux 下载。上传 `6405proj-code.tar.gz` 到 `/workspace` 后：

```bash
cd /workspace
tar -xzf 6405proj-code.tar.gz
cd /workspace/6405proj
nvidia-smi
df -h /workspace
conda --version
command -v nvcc
command -v tmux
```

上传可使用 RunPod JupyterLab 的文件浏览器。若用 scp，须使用控制台提供的**直接 TCP SSH** IP 和外部端口；不要假设 SSH 固定是 22 或所有代理 SSH 都支持 scp。

```powershell
scp -P EXTERNAL_SSH_PORT "D:\11\algorism\skealran\natural_nlp\6405_study\6405proj-code.tar.gz" root@POD_IP:/workspace/
```

用户和端口替换为控制台实际值。更新源码前停止使用这些代码的任务；上传包不含训练结果、环境和数据，不要删除服务器已有的模型目录。

## 7. Conda 安装、下载与验证

```bash
cd /workspace/6405proj
bash scripts/setup_conda.sh
bash scripts/install_mamba.sh
source scripts/activate.sh
python -m pip check
python -m proj6405.doctor
python -m proj6405.download
python -m proj6405.doctor --smoke roberta --length 512
python -m proj6405.doctor --smoke mamba --length 512
python -m proj6405.doctor --smoke longformer --length 512
```

没有 Conda 时按 [Conda 指南](CONDA_RUNPOD_ZH.md) 安装 Miniforge。环境默认 `/workspace/6405proj/.conda-env`；每次新终端执行 `source scripts/activate.sh`。PyTorch 2.5.1/cu121、Transformers 4.48.3、Mamba-SSM 2.2.4 等版本保持固定，不自动升级。

验收：CUDA 可用；Mamba fast path 为 True；真实主干的 loss 和梯度有限。Smoke 没有完整优化器状态，不等同于正式训练显存验收。CUDA 扩展失败时先排查 toolkit、驱动、编译器，不开启慢路径后宣称测得架构性能。

下载器保存来源 commit。下载完成后离线运行：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -m proj6405.matrix --stage pilot --phase train --prefix pilot --seeds 42
python -m proj6405.matrix --stage pilot --phase evaluate --prefix pilot --seeds 42
```

Pilot 只是 AG News 小样本流程检查；不会混进正式表格。

## 8. 输入准备、长度描述和长输入显存预检

```bash
python -m proj6405.full --phase plan
python -m proj6405.full --phase prepare
python -m proj6405.analyze_data --split train
```

prepare 生成三个数据集的 shared512，以及 SCOTUS 的 shared1024/2048/4096；不使用 h6 抽样视图。长度统计在 CPU 上执行，完整训练划分可能需要等待，结果写入 `reports/lengths_train_full.csv`。

在正式长文本矩阵前，用实际优化步骤预检最大输入。以下只使用训练与验证划分：

```bash
for model in mamba longformer; do
  python -m proj6405.train --model "$model" --dataset scotus \
    --name "preflight_${model}_4096" --view shared4096 --max-length 4096 \
    --epochs 1 --limit-train 128 --limit-val 64 \
    --batch-size 1 --grad-accum 16 --eval-batch-size 1 --gradient-checkpointing
done
```

检查实际日志和峰值显存。若 4K 仍 OOM，停止并记录该设置不可行；可以先完成 512/1K/2K，用 `--lengths 512 1024 2048` 筛选长度组，报告清楚缺少 4K。不得在不同模型上偷偷采用不同输入长度再画同条件对比。

## 9. 验证集调参：在正式配置冻结前完成

默认固定参数可直接执行。若验证曲线显示明显欠拟合或不稳定，建议只探索学习率 `{1e-5, 2e-5, 5e-5}`、SVM C `{0.1, 1, 10}`，控制搜索规模。所有候选使用同一训练子集、验证子集、轮次和 seed，不运行 test。

示例（分别改学习率和独立 run 名；其他核心数据集同理）：

```bash
python -m proj6405.train --model mamba --dataset imdb --view shared512 \
  --name tune_imdb_mamba_lr1e5 --lr 1e-5 --epochs 3 --seed 42 \
  --limit-train 5000 --limit-val 1000
python -m proj6405.train --model svm --dataset imdb --view shared512 \
  --name tune_imdb_svm_c01 --svm-c 0.1
```

为每个神经模型选定一个学习率：RoBERTa/Mamba 可按三个数据集验证 Macro-F1 的均值确定；Longformer 在 SCOTUS 验证集确定。各长度使用同一选定学习率，减少搜索预算差异。搜索若只覆盖一个数据集，报告需说明，不把它称为全域最优。

将最终选择写入 `configs/full.json`，保留候选日志和选择理由。**不要用测试结果挑参数。** 首次正式 train/evaluate/benchmark/report 后配置和来源被冻结；更改配置必须用新 prefix，例如 `--prefix full_v2`，所有后续命令使用相同 prefix。

## 10. 正式训练：分两组顺序运行

在 tmux 内执行；每条命令成功后再继续。不要同时启动两组争抢显存。

```bash
tmux new -s full-train
cd /workspace/6405proj
source scripts/activate.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
bash scripts/stop_services.sh

python -m proj6405.full --phase train --stage core
python -m proj6405.full --phase train --stage long
```

按 Ctrl+B、松开、再按 D 离开会话，任务继续；`tmux attach -t full-train` 返回。tmux 不会在 Pod 停机后保存进程。

每个任务的 stdout/stderr 保存到 `logs/full/`，父终端显示当前命令和日志位置。另开终端观察：

```bash
cd /workspace/6405proj
source scripts/activate.sh
python -m proj6405.full --phase status
cat reports/full/progress.json
tail -n 30 logs/full/full_core_ag_news_roberta_shared512_s42_train.log
```

确认任务计划而不执行：`python -m proj6405.full --phase train --stage long --dry-run`。

已经完成且配置、数据、预训练模型来源一致的训练会自动跳过。中断任务默认停止并要求检查；确认原进程已结束、日志问题已处理后：

```bash
python -m proj6405.full --phase train --stage core --retry-incomplete
```

该选项将匹配配置的中断目录移入 `runs/_interrupted/`，保留原文件，再从预训练权重重新训练；**不是恢复 optimizer step**。正常完成目录不会移走。若遇到 `active.lock`，先检查里面的 PID 与 host；确定对应进程不存在后才手动移除锁，不能边训练边清理锁。

## 11. 完整测试与效率评估

所有正式配置由验证集确定之后，运行：

```bash
python -m proj6405.full --phase evaluate --stage core
python -m proj6405.full --phase evaluate --stage long
bash scripts/stop_services.sh
python -m proj6405.full --phase benchmark --stage core --benchmark-batch 1
python -m proj6405.full --phase benchmark --stage long --benchmark-batch 1
```

evaluate 不限制测试条数。每个 seed 独立测完整测试集。benchmark 只测首个预先指定 seed（默认 42）；固定样本 ID、预热、重复测量，在 CUDA 边界同步。

有余力再测批处理吞吐；必须对各模型使用相同 batch，OOM 如实记录：

```bash
python -m proj6405.full --phase benchmark --stage core --benchmark-batch 4
python -m proj6405.full --phase benchmark --stage long --benchmark-batch 4
```

benchmark 不与训练或网页推理同时执行。推理精度保持 bf16；SVM 在 CPU 运行、精度 fp32、GPU 显存为不适用。重新执行 evaluate/benchmark 会重新测量并更新对应结果文件；应先备份需要保留的旧测量。

重点指标：Macro Precision、Macro Recall、Macro-F1、Accuracy；模型 ms/doc、本地流程 docs/s、延迟 P50/P95；GPU allocated/reserved 峰值 MiB；训练总时间。CPU SVM 的 RSS 记录不是峰值内存。

加载权重和网页网络延迟不计入模型前向时间。共同前缀的离线构建不计入离线 benchmark，在线网页则需处理新文本。不能把它们当成相同的端到端服务延迟。

## 12. 汇总图表和结果分析

```bash
python -m proj6405.full --phase report
python -m proj6405.full --phase status
for dataset in ag_news imdb scotus; do
  python -m proj6405.demo_samples --dataset "$dataset"
done
```

输出分别放在 `reports/full/core/` 和 `reports/full/long/`，防止互相覆盖或错误合并。每组有：

- `results.csv`：每次完整测试的分数、样本与配置。
- `summary.csv`：Macro-P/R/F1、Accuracy 和训练时间；默认 n_seeds=1，均值列就是单次结果，跨种子标准差留空。
- `benchmarks.csv`：效率记录，batch/precision 明确分列。
- `f1_summary_*.png`：单次分类效果；单 seed 不绘制跨种子误差条。
- `confusion_*.png`、`quality_cost_*.png`、`scaling_*.png`，以及 SCOTUS 的 `length_f1_scotus.png`。

分析顺序：先看三任务质量排序和种子波动，再看 SCOTUS 增加长度的质量变化，最后结合显存、速度和训练时间判断收益。按真实预测选 2–3 个错误/分歧案例，并标注样本 ID、真实标签、截断位置和观察到的证据。

单种子不能估计训练随机性造成的波动；在报告中说明此限制，不声称验证了跨种子稳定性。预训练差异、类别不均衡、领域差异和单卡实现都应进入限制讨论。

## 13. RunPod 前端与后台部署

```text
本地浏览器 → RunPod HTTPS 代理 :8501 → Streamlit
                                      ↓ Pod 内部请求
                               FastAPI 127.0.0.1:8000
                                      ↓
                              已微调的模型 / RTX 3090
```

API 单 worker、推理串行，每次只缓存一个模型控制显存。切换模型时单独记录加载和预热耗时。

```bash
cd /workspace/6405proj
bash scripts/stop_services.sh
RUN_PREFIX=full_ UI_HOST=0.0.0.0 bash scripts/start_services.sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/runs
tail -n 30 logs/api.log
tail -n 30 logs/ui.log
```

`RUN_PREFIX` 隐藏 pilot、tune、h6 等不匹配结果；更换 campaign 时替换前缀并重启服务。RunPod Connect → HTTP Services → 8501 打开网页。服务需监听 0.0.0.0 才能由代理访问，API 保持 Pod 内部访问；配置依据 [RunPod 端口文档](https://docs.runpod.io/pods/configuration/expose-ports)。

若用直接 TCP SSH 隧道访问，默认回环监听也可用，端口替换为控制台提供的外部 SSH 端口：

```powershell
ssh -N -L 8501:127.0.0.1:8501 -p EXTERNAL_SSH_PORT root@POD_IP
```

浏览器打开 `http://127.0.0.1:8501`。网页对外可达时只在需要演示期间开放服务；Stop Pod 后网页不再提供实时推理。

## 14. 展示操作与 10 分钟视频

1. 介绍目标用户、质量与计算成本问题、数据和固定划分。
2. 展示核心表 `reports/full/core/summary.csv` 和分类效果图。
3. 网页选择 AG News，手动选 `full_core_ag_news_svm_shared512_s42`、`full_core_ag_news_roberta_shared512_s42`、`full_core_ag_news_mamba_shared512_s42`。
4. 上传 `reports/demo_samples/ag_news/sample_0.txt`，展示真实标签、各模型结果和分阶段耗时。
5. 左侧 Dataset 选择 SCOTUS，Input view 切换到 Shared prefix · 2,048 token budget，同长度比较 `full_long_scotus_mamba_shared2048_s42` 与 `full_long_scotus_longformer_shared2048_s42`，然后展示离线长度曲线。
6. 展示一个错误案例，说明截断、领域术语或类别歧义等可能因素；区分证据和推测。
7. 总结在本实验条件下的模型选择建议、限制和成员贡献。

时间分配可用：问题 45 秒，理论 75 秒，数据协议 60 秒，关键 Python 代码 60 秒，结果 135 秒，网页 120 秒，错误/限制 60 秒，贡献/引用 30 秒，总计 9 分 45 秒。课程截图要求真人讲解、no GAI voice；具体提交格式以当前 syllabus 为准。

展示固定 seed 42，避免挑选测试分数最高的一次冒充平均效果。单条预测不能计算整体 F1，网页成绩来自离线完整测试。剪辑等待时注明，不把剪辑后时长当作模型性能。

## 15. 备份、提交和预计结果

```bash
python -m proj6405.full --phase report
tar -czf /workspace/6405-full-results.tar.gz reports logs configs docs README.md SOURCE_MANIFEST.json
tar -czf /workspace/6405-full-checkpoints.tar.gz runs
```

下载并检查结果包、检查点包和录像后再停止 Pod。不要把 `.conda-env`、下载缓存、原始数据全部塞进课程源码提交包。检查点包可能较大；若分批下载，先保证指标、配置、来源记录和图表到本地。另保留源码包及项目的精确版本。

提交：Python 源码/依赖/运行说明、报告或 slides、真人讲解视频、参考文献、带成员贡献和签名的 reflection、GAI declaration，以及每位成员的 peer evaluation。附件截图中的日期未明确年份，截止时间以当前课程通知为准。

预计得到的是**20 个已完成配置的真实结果、两组汇总图、20 份主效率记录和可操作界面**，不保证任何模型排名。可能的假设包括：SVM 短文本性价比高；Mamba 长输入资源增长较温和；更长上下文改善法律分类。但都可能被结果否定，也不提前填写 F1、加速倍数或显存节约百分比。

## 16. 时间与风险安排

按阶段推进：环境与 pilot → 验证集调参 → 核心单种子 → 长度组 → 测试与效率 → 分析和展示。20 次训练仍可能耗时较长；当前没有该 Pod 的速度实测，不承诺固定 GPU 小时。

用各数据集/模型首个正式 epoch 的日志估算剩余时间，分别计算短输入和长输入；首轮本地加载和保存也有额外成本。不要用新闻短文本的吞吐推算 4K 法律文本。训练可以夜间在 tmux 中执行，写 PPT 不需要一直占着 GPU，但演示前必须重新启动服务。

4K 无法运行就记录 OOM，明确实际完成的长度；有中断就从日志和 status 恢复流程。没有运行的配置留空，不用预估数字补齐“完整”表格。

详细底层说明见 [实验原理与命令参考](EXPERIMENT_GUIDE_ZH.md)，验证范围见 [VALIDATION.md](VALIDATION.md)。
