# 六小时 RunPod 预算方案

本方案优先适用：只有 **6 小时累计 GPU Pod 额度**，下载和安装也占用这段时间。若指的是 6 美元余额，应按控制台实际单价重新换算。这里没有运行速度实测，不保证所有可选内容都能在六小时完成。

环境现统一使用 Conda，详细说明见 [Conda + RunPod 指南](CONDA_RUNPOD_ZH.md)。脚本在项目 `.conda-env` 中建立 Python 3.11；不再使用 `.venv`。

六小时可以争取完成一套小样本课程实验和演示；原来的全量、多种子实验不再适用。建议题目改为：**有限计算预算下的文本分类：TF-IDF、RoBERTa 与 Mamba 的质量和效率比较**。只有实际完成长文本对照后，才在结论中讨论长上下文结果。

## 1. 保留与缩减

| 项目 | 六小时配置 |
|---|---|
| 数据集 | AG News、IMDb、LexGLUE SCOTUS |
| 核心模型 | TF-IDF + Linear SVM、RoBERTa-base、Mamba-130M |
| 训练 / 验证 | 每个数据集固定分层抽样 1,000 / 200 条；所有方法使用相同 ID |
| 测试 | 官方完整测试集；AG News 7,600、IMDb 25,000、SCOTUS 1,400 条 |
| 神经模型训练 | 2 epochs，seed=42，学习率 2e-5；不进行超参数搜索 |
| 输入 | 各分词器都容纳得下的相同原文前缀，预算 512 tokens |
| 核心训练总数 | 3 次 SVM + 6 次神经模型训练 = 9 次 |
| 效率评估 | 每配置固定 64 个样本，batch=1，热身 5 次，重复 3 次 |
| 可选长文本 | SCOTUS：补 Longformer-512、Mamba-2048、Longformer-2048 共 3 次训练 |
| 删除 | 370M、3 seeds、4K 全曲线、大规模调参、消融矩阵 |

SVM 的 2 epochs 参数不参与拟合。神经模型的 micro-batch=2、梯度累积=8；2048 对照改为 micro-batch=1、累积=16、gradient checkpointing。有效 batch 均约 16。

训练样本少、轮次少，神经模型可能尚未收敛。SVM 获胜是有效观察，不能解释为某个架构普遍更强。单种子没有跨种子稳定性证据，不制作虚假的标准差或显著性结论。200 条验证集上的 SCOTUS 少数类估计可能很不稳定。

抽样使用固定数据种子 2026。若某划分含仅一条样本的类别，采用每个已观测类别至少一条、剩余名额按比例分配的确定性抽样；不会为了抽样重新划分训练与测试。所有架构共用保存的样本 ID。

## 2. 六小时分配与停止条件

这是**时间盒**，不是预计每一步一定按时完成的保证。

| 从启动 Pod 起 | 时间分配 |
|---|---|
| 0:00–1:15 | 环境、下载、GPU 检查、共同输入准备 |
| 1:15–2:45 | 核心 9 次训练，优先完成 SVM 和 RoBERTa |
| 2:45–3:30 | 完整测试集评估、效率测试、结果导出 |
| 3:30–4:15 | 可选长文本对照；时间不足则直接跳过 |
| 4:15–4:45 | 启动网页、固定样例演示并录屏 |
| 4:45–5:15 | 下载指标、图表、模型、日志和录像 |
| 5:15–6:00 | 故障缓冲；尽早完成并停止 Pod |

- 第 75 分钟还未通过 Mamba 快速路径检查，就取消可选长文本，不继续反复换版本。可以先用下文的 `--models svm roberta` 完成两个方法的实验。缺少 Mamba 时必须明确报告，不能宣称完成 Mamba 对决。
- 只在核心评估完成、且距额度耗尽还有 **至少 135 分钟**时考虑可选长文本。最多再花 45 分钟，其余留给演示和备份。
- 最后 90 分钟不要开始新的训练。不要为了多一张图用掉备份和录屏时间。
- `--minutes` 是**本次调用所有子命令共享的墙钟上限**。超时会停止当前子进程，退出码为 124。它不是六小时累计计费器，也不会停止 RunPod。每次调用前都需要看控制台剩余额度。
- 超时中断的训练目录不是完整实验；虽然可能有检查点，API 不会把它当成完成模型。查看日志后，用新 `--prefix` 开始新的一组实验，或者只评估已经完成的模型组。不要手改 status 冒充完成。

## 3. 开机前完成的准备（本地，不使用 GPU）

1. 阅读本页，准备源码包、RunPod 登录和录屏工具。
2. 设置 Pod 的持久化卷，项目使用 `/workspace/6405proj`；预先配置 HTTP 端口 `8501`，避免训练期间编辑 Pod 导致重启。
3. 优先选已有 Conda 的 Linux 开发环境。Python 3.11 由脚本创建，PyTorch 使用 2.5.1/cu121。若需要源码编译 Mamba，镜像必须提供匹配的 CUDA toolkit / nvcc 和 C++ 编译器。不要只依据 `nvidia-smi` 的 CUDA 字样判断 toolkit 是否已安装。没有 Conda 时先按 Conda 指南安装 Miniforge。
4. 代码上传后，在 `/workspace` 解压。数据、模型继续在 Linux 下载，不在本地下载大权重。

## 4. Linux 上的最短执行顺序

以下步骤从项目根目录执行；安装、下载没有包含在 `budget --minutes` 计时内，但会占用 RunPod 租用额度。安装失败先读日志，不重复盲跑。

```bash
cd /workspace/6405proj
bash scripts/setup_conda.sh
bash scripts/install_mamba.sh
source scripts/activate.sh
python -m proj6405.download
python -m proj6405.doctor --smoke mamba --length 512
```

默认会下载 Longformer 权重。核心输入视图需要它的 tokenizer；可选扩展也直接使用该权重。下载完成后离线运行：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -m proj6405.budget --phase prepare --minutes 25
python -m proj6405.budget --phase train --minutes 90
```

新的 `h6_shared512` 视图会**先取训练和验证子集，再分词**，节省处理全量训练集的时间。测试集仍完整保留。`prepared_splits.json` 保存实际条数和样本 ID 签名；`manifest.json` 的 `splits` 保留来源数据集条数，确保小样本测试不会被误判成完整测试。

如需断开 SSH 后继续训练，在 tmux 中执行第二条命令：

```bash
tmux new -s h6-train
cd /workspace/6405proj
source scripts/activate.sh
python -m proj6405.budget --phase train --minutes 90
# 按 Ctrl+B，松开，再按 D 退出 tmux 界面。
# 回到会话：tmux attach -t h6-train
```

不要前台、tmux 同时启动两份训练。已经完成且配置一致的训练会跳过；未完成的目录不会被静默覆盖。

**核心训练完成后评估：**

```bash
python -m proj6405.budget --phase evaluate --minutes 30
bash scripts/stop_services.sh
python -m proj6405.budget --phase benchmark --minutes 15
python -m proj6405.report --prefix h6_
for dataset in ag_news imdb scotus; do
  python -m proj6405.demo_samples --dataset "$dataset"
done
```

先用验证集确定配置，再执行上述测试集评估。不得因为测试分数低而改变 seed、重抽训练集、追加轮次后再宣称测试集未参与调参。

查看计划而不启动任何训练：

```bash
python -m proj6405.budget --phase train --dry-run
```

如果 Mamba 安装失败，可以先完成 SVM/RoBERTa；在训练、评估、benchmark 三个命令中都加入同样的模型过滤：

```bash
python -m proj6405.budget --phase train --models svm roberta --minutes 60
python -m proj6405.budget --phase evaluate --models svm roberta --minutes 25
python -m proj6405.budget --phase benchmark --models svm roberta --minutes 10
python -m proj6405.report --prefix h6_
```

这只是保底交付，不是完成原来的三模型任务。如果某个模型组只完成部分数据集，使用已完成 run 的名字逐条调用 `proj6405.evaluate --run ...` 和 `proj6405.benchmark --run ...`，避免遇到未完成目录而停止整组任务。

## 5. 可选长文本扩展：有余量才运行

前提是核心 Mamba-512 已完成。使用相同 SCOTUS 训练/验证 ID、相同两轮训练，补齐 512 vs 2048 的配对比较。保持 `h6` prefix，报告能复用已有 Mamba-512。

下面用外层 `timeout` 将准备、训练、评估、benchmark 的**总时间限制在 45 分钟**。需要 Linux 的 GNU `timeout`。

```bash
timeout --signal=TERM --kill-after=15s 45m bash -c '
  set -e
  python -m proj6405.budget --stage long --phase prepare --minutes 10
  python -m proj6405.budget --stage long --phase train --minutes 35
  python -m proj6405.budget --stage long --phase evaluate --minutes 10
  python -m proj6405.budget --stage long --phase benchmark --minutes 5
'
python -m proj6405.report --prefix h6_
```

时间上限可能使扩展未完成，只报告实际完成的配对。不要将 Mamba 接收 2048 输入本身描述成效果改善；没有长文本配对结果时，主实验只支持 512 预算下的结论。4K 与三种子留到未来工作。

## 6. RunPod 网页与录屏

已支持通过环境变量选择监听地址，无需手改源文件。关闭既有服务，再用 RunPod 模式启动：

```bash
bash scripts/stop_services.sh
UI_HOST=0.0.0.0 bash scripts/start_services.sh
curl http://127.0.0.1:8000/health
```

RunPod Connect → HTTP Services → 8501。代理地址形式为 `https://你的PodID-8501.proxy.runpod.net`。FastAPI 保持内部 8000 端口。网页代理地址是公开入口，课堂演示后关闭服务。若出现长连接重连问题，可使用 RunPod 的直接 TCP SSH 连接做端口转发；不要把 HTTP 页面链接当作 SSH 地址。

录屏应在 Pod 仍运行时完成：

1. 说明 6 小时约束、固定 1,000 条训练样本、单种子和完整测试集。
2. 展示三数据集 Macro-P/R/F1、训练样本数、时间及显存。
3. 上传 `reports/demo_samples/` 的固定测试文档，展示三个已完成模型的预测。
4. 展示一个真实错误案例，区分模型计算耗时、加载预热与整个网页请求耗时。
5. 有长文本配对才展示对应图；没有则说明计算预算限制。

图表和 CSV 下载后可以在本地制作 PPT、写报告。后续没有 GPU 额度时仍可展示保存的结果和录屏，但不能把录像说成实时 GPU 推理。当前前端测试结果页只显示完整测试集结果；不要临时改成测试子集后期待它出现在完整成绩表里。

## 7. 先备份，再停机

在 Linux 项目根目录生成两个包：

```bash
python -m proj6405.report --prefix h6_
tar -czf /workspace/6405-results-h6.tar.gz reports logs docs README.md SOURCE_MANIFEST.json
tar -czf /workspace/6405-checkpoints-h6.tar.gz runs
```

第一个包小，优先通过 RunPod JupyterLab 文件浏览器下载；第二个包含训练检查点，体积更大。确认本地能解压结果包、图表能打开，录像也已保存在本地，再停止 Pod。训练配置和 data/model manifest 随每个 run 保存，检查点包要一并保留。不要打包整个 `.conda-env`、旧 `.venv`、缓存或原始数据集。

Volume Disk 的 `/workspace` 通常在 Stop 后保留、Terminate 后删除；Network Volume 独立持久化。Stop 后存储仍可能计费。以你的实际挂载配置为准，不依赖临时容器盘保存成果。

参考：
- [RunPod 端口配置](https://docs.runpod.io/pods/configuration/expose-ports)
- [RunPod 存储](https://docs.runpod.io/pods/storage/types)
- [RunPod 启停管理](https://docs.runpod.io/pods/manage-pods)
- [Mamba v2.2.4 安装说明](https://github.com/state-spaces/mamba/tree/v2.2.4#installation)

## 8. 报告中应使用的表述

> 本研究在单张 RTX 3090 和六小时计算资源约束下进行。各数据集从训练划分中固定分层抽取 1,000 条训练样本和 200 条验证样本，以单个随机种子微调两个 epoch，并在官方完整测试集上评估已完成配置。结果反映本次低资源设置；未进行多种子稳定性验证，也不代表充分训练后的架构上限。

若实际未完成全部配置、测试集或可选扩展，应按实际记录改写这段话。预期产物是小样本实验表、错误案例、速度/显存数据和可演示网页；不预填 F1，不保证 Mamba 获胜或六小时内完成所有项目。
