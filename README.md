# 6405proj：长文本分类的质量与计算成本

TF-IDF + Linear SVM / RoBERTa-base / Mamba-130M / Longformer-base。

**2026-10-28 前端更新：** 全英文页面、统一数据集/输入视图筛选、模型预测卡片和交互式错误阅读。服务器已有项目可只更新两个界面文件，见 [英文前端更新步骤](docs/UI_UPDATE_ZH.md)。

**从这里开始：** [完整实验方案与运行步骤](docs/FULL_PROJECT_ZH.md) · [Conda 环境](docs/CONDA_RUNPOD_ZH.md) · [展示与录屏](docs/DEMO_GUIDE_ZH.md) · [结果模板](docs/RESULTS_TEMPLATE_ZH.md) · [验证记录](docs/VALIDATION.md)

数据集和预训练模型均由你在 **Linux GPU 服务器**下载；本地只编写、上传代码及用浏览器访问。当前仓库没有真实训练成绩，不预设 Mamba 获胜。

**当前默认是完整版（2026-10-26）：** 三个数据集全量训练；所有模型固定 seed 42；SCOTUS 上 512/1024/2048/4096 长度对照，共 20 次训练。核心组和长度组分别汇总。旧 [六小时方案](docs/SIX_HOUR_PLAN_ZH.md) 仅保留作缩减预算的备选。

**环境统一使用 Conda：** [Conda + RunPod 指南](docs/CONDA_RUNPOD_ZH.md)。环境放在项目的 `.conda-env`；新终端执行 `source scripts/activate.sh`，后台 API/UI 自动激活相同环境。

## 环境和流程检查

以下命令在 Linux 服务器的 `6405proj` 目录执行。需要 Conda、可用 NVIDIA 驱动、tmux；脚本创建 Python 3.11 环境。CUDA 扩展编译还需要匹配的 CUDA toolkit 和 C++ 编译器。没有 Conda 时先按 Conda 指南安装 Miniforge。

```bash
bash scripts/setup_conda.sh
bash scripts/install_mamba.sh
source scripts/activate.sh
python -m proj6405.download
python -m proj6405.doctor --smoke roberta --length 512
python -m proj6405.doctor --smoke mamba --length 512
python -m proj6405.doctor --smoke longformer --length 512
python -m proj6405.matrix --stage pilot --phase train --prefix pilot --seeds 42
python -m proj6405.matrix --stage pilot --phase evaluate --prefix pilot --seeds 42
```

Pilot 只有小样本训练和子集评估，用于检查流程。正式长文本训练前，按完整手册进行 4096 长度的实际训练显存预检，并仅用验证集决定超参数。

## 正式实验入口

```bash
source scripts/activate.sh
python -m proj6405.full --phase plan
python -m proj6405.full --phase prepare
python -m proj6405.analyze_data --split train
bash scripts/stop_services.sh
# 在 tmux 内顺序执行；每条成功后再继续。默认配置在 configs/full.json。
python -m proj6405.full --phase train --stage core
python -m proj6405.full --phase train --stage long
python -m proj6405.full --phase evaluate
python -m proj6405.full --phase benchmark --benchmark-batch 1
python -m proj6405.full --phase report
```

完成后查看 `reports/full/core/` 与 `reports/full/long/`。`summary.csv` 给出单次结果和种子数（1）；均值列等于该次结果，所有跨种子标准差留空。`python -m proj6405.full --phase status` 查看完成数量。中断重跑、OOM、调参和备份见主手册；不支持自动恢复 optimizer step。

## RunPod 网页展示

在 Pod 的 HTTP 端口中配置 8501，服务器执行：

```bash
RUN_PREFIX=full_ UI_HOST=0.0.0.0 bash scripts/start_services.sh
curl http://127.0.0.1:8000/health
```

通过 RunPod Connect → HTTP Services → 8501 打开网页。API 留在服务器本地。也可按主手册使用直接 TCP SSH 隧道，外部端口须按控制台填写。页面包含实时预测、测试结果与效率、错误案例三个部分。正式 benchmark 前关闭服务。

## 文件结构

```text
6405proj/
  README.md
  requirements.txt / requirements-mamba.txt
  proj6405/
    download.py       数据/模型下载与版本记录
    views.py          相同原文前缀的受控实验视图
    analyze_data.py   长度分布与截断率
    models.py         三类神经主干、分类头、池化
    train.py          SVM 与神经模型训练，只用验证集选模型
    evaluate.py       测试指标、混淆矩阵、逐条预测
    benchmark.py      预热、同步、分阶段计时与显存
    full.py          完整实验调度、配置冻结、完成状态与中断处理
    matrix.py         pilot 与单独的手动实验矩阵
    budget.py         旧六小时备选方案
    report.py         汇总 CSV 和静态结果图
    api.py            FastAPI 单 worker 推理服务
    doctor.py         环境与前向/反向检查
    demo_samples.py   导出演示用测试文档
  app.py              Streamlit 前端
  .streamlit/config.toml  浅色主题
  configs/full.json   完整方案的种子、学习率、batch 和长度
  scripts/            Linux 环境、启动与停止脚本
  tests/              人工小数据和微型随机模型功能测试
  docs/               中文手册与报告模板
  data/ models/       服务器下载/预处理生成
  runs/               每次训练的独立目录
  reports/ logs/      结果与日志
```

不要同时启动训练和正式 benchmark。不要把不同 batch、文本视图或训练样本量的结果混成一个公平比较表。对 SVM，`native` 表示全文；对神经模型，`native` 仍会按 `max_length` 截断。
