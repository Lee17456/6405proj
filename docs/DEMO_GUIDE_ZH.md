# 展示、录屏和提交操作

当前对应 [完整实验方案](FULL_PROJECT_ZH.md)：全量数据、固定单种子 42、四个长度。需要实时 GPU 推理的录屏应在 Pod 运行期间完成；分析图表和写报告可离线进行。

## 1. 展示前应已有的产物

至少准备：

- 三个数据集的实际长度统计、来源和标签映射。
- 三个核心方法的完整测试指标；报告 seed 42 的单次结果，说明未测量跨种子波动。
- SCOTUS 上 Mamba 与 Longformer 的 512/1024/2048/4096 长度结果；未完成或 OOM 明确标注。
- 同一 batch、精度、样本集合的延迟与显存记录。
- 两三个错误或分歧案例，以及它们的真实标签和样本 ID。
- 可启动的 API、Streamlit、RunPod HTTP 页面或直接 TCP SSH 隧道。

演示模型固定使用预先指定的 seed 42，并保留 checkpoint 和配置；不按测试分数挑最高 seed 冒充均值。

## 2. 生成展示文件

服务器执行：

```bash
cd /workspace/6405proj
source scripts/activate.sh
python -m proj6405.full --phase report
python -m proj6405.demo_samples --dataset ag_news
python -m proj6405.demo_samples --dataset imdb
python -m proj6405.demo_samples --dataset scotus
```

本地 PowerShell 下载图表与测试文档：

```powershell
Set-Location 'D:\11\algorism\skealran\natural_nlp\6405_study'
scp -P EXTERNAL_SSH_PORT -r root@POD_IP:/workspace/6405proj/reports ./6405proj/server_reports
```

使用 RunPod 控制台提供的直接 TCP SSH IP、外部端口与用户名；也可通过 JupyterLab 下载文件。`reports/demo_samples/<dataset>/index.json` 记录真实标签。默认导出前五个测试文档，不根据正确与否筛选。可以额外选一个错误案例展示，但要明确它是错误分析样例，不能替代整体测试。

## 3. 启动并确认服务

服务器执行：

```bash
bash scripts/stop_services.sh
RUN_PREFIX=full_ UI_HOST=0.0.0.0 bash scripts/start_services.sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/runs
```

Pod 配置 HTTP 端口 8501 后，通过 RunPod Connect → HTTP Services → 8501 访问。若选择直接 TCP SSH 隧道，本地另开 PowerShell，保持开启：

```powershell
ssh -N -L 8501:127.0.0.1:8501 -p EXTERNAL_SSH_PORT root@POD_IP
```

打开 `http://127.0.0.1:8501`。如果本地 8501 已被占用，改成 `-L 18501:127.0.0.1:8501`，浏览器访问 `http://127.0.0.1:18501`；服务器仍用 8501。

## 4. 网页实际操作顺序

1. 在左侧 Dataset 选择 AG News，Input view 选择 Shared prefix · 512 token budget，打开 Classify。
2. 手动选择同一协议的 SVM、RoBERTa、Mamba，即三个 `full_core_ag_news_*_shared512_s42`；不要把同一模型的三个 seed 当成三个架构。
3. 选择 Upload file，上传 `reports/demo_samples/ag_news/sample_0.txt`，点击 Compare predictions。
4. 对照 index.json 说明真实类别；展示模型预测，不把某一条结果说成准确率。
5. 说明三种时间：模型前向、本地推理流程、API 请求总耗时。模型切换加载和预热单独显示。
6. Dataset 切换到 SCOTUS，Input view 切换到 Shared prefix · 2,048 token budget，上传一篇长文；比较 `full_long_scotus_mamba_shared2048_s42` 和 `full_long_scotus_longformer_shared2048_s42`。说明它们使用共同原文前缀。
7. 打开 Results & efficiency，展示完整测试集 Macro-P/R/F1，解释为何 Macro-F1 适合观察少数类别。
8. 打开 Error explorer，选择模型和文档，对照 Expected / Predicted 标签，说明截断、专业术语或类别歧义可能造成的问题。
9. 展示 `reports/full/core/`、`reports/full/long/` 中的分类效果、长度、耗时与显存图。页面单次延迟只用于交互观察，正式结论来自重复 benchmark。

前端一次只缓存一个模型以控制显存，选择多个模型会依次切换。为避免录屏中长时间等待，可以先录制一次操作，再保留关键片段；若剪辑或加速等待，应明确标注，不能把剪辑后的等待时间当作模型速度。

## 5. 建议的 10 分钟以内视频安排

| 时间 | 内容 | 展示材料 |
|---|---|---|
| 0:00–0:45 | 问题、目标用户和研究问题 | 1 页问题定义 |
| 0:45–2:00 | TF-IDF、注意力、选择性状态空间的差异 | 结构示意与论文引用 |
| 2:00–3:00 | 三个数据集、划分、共同输入协议 | 长度统计、类别分布 |
| 3:00–4:00 | 关键 Python 实现 | 分类头、池化、数据切分、计时同步 |
| 4:00–6:15 | 正式指标、长度与成本结果 | CSV 和图表，标明硬件及协议 |
| 6:15–8:15 | 网页预测和长文比较 | 实时操作 |
| 8:15–9:15 | 错误案例、限制和结论 | 一两例证据 |
| 9:15–9:45 | 成员贡献、引用和后续方向 | 结束页 |

留约 15 秒缓冲，确保不超过课程截图的 10 分钟。使用成员真实讲解音轨；截图明确要求 **no GAI voice**。主要实现采用 Python，代码提交为 `.py`，本项目无需 MATLAB。

## 6. 讲解时需要准确表达的内容

- “我们比较的是这些预训练模型在固定任务和资源下的表现。”
- “shared512 是相同原文内容，并不意味着不同分词器的 token 数完全相同。”
- “SVM native 使用全文，因此单独标注，不与共享前缀的结果混称完全相同输入。”
- “这项结果来自 RTX 3090、指定精度与 batch；其他硬件或更大模型可能不同。”
- “Mamba 的最终排序由实验决定，性能不及 Transformer 也是有效发现。”
- “单条输入没有标签就不能计算 F1；页面指标来自预先完成的测试集评估。”
- “SCOTUS 类别数按实际下载 manifest 填写，避免照抄页面正文造成口径错误。”

不要声称实现了完整法律理解、所有长文本任务、百万 token 推理或原论文 5 倍加速，除非确实实施并测得对应实验。

## 7. 预演与备份

录屏前一天做一次完整预演：

- 确认 SSH 密钥/账号有效，隧道能连通。
- 确认三个任务的对应 checkpoint 存在，API 能逐个加载。
- 实际点击一次短文和一次长文，确认没有 OOM。
- 确认正式图表采用正确的 prefix，没有混入 pilot。
- 检查日志没有 NaN、下载尝试或未处理异常。
- 备份离线图表、演示文档、最终视频及源代码。

如果网络临时不可用，可展示事先录制的真实运行片段和离线实验图，但应标明是录制内容。不能用编造的数值填补连接故障。

## 8. 提交材料核对

依据截图准备，具体格式以 syllabus 和课程系统为准：

1. 不超过 10 分钟的视频：含 Python 代码、理论、实验、前端和参考文献，真人配音。
2. Python 源码与 requirements、README、Linux 运行步骤。
3. 实验结果、超参数及来源记录；若教师不要求大模型文件，可提供下载及重训方法。
4. 写明每位成员贡献、成员签名的 group reflection。
5. 使用课程模板填写 GAI declaration，如实记录 AI 辅助设计、代码或文档的范围和人工验证工作。
6. 每人完成 peer evaluation。截图说明遗漏会扣分，不要只交小组材料而忘记个人步骤。
7. Week 12 咨询时带研究问题、模型/数据清单和一组 pilot 结果。

不将本手册或 AI 生成内容直接视为已经完成的实验报告；最终报告需要填入真实结果、团队自己的理解和贡献。
