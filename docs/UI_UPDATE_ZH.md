# 英文前端更新（2026-10-28）

本次只更新 `app.py` 与 `.streamlit/config.toml`。不需要重新下载数据、训练模型或修改已有训练配置；使用现有 FastAPI 接口和结果文件。界面所有固定文本、按钮、图表和 SCOTUS 主题名称为英文，原始标签编码保留。中文操作文档不受影响。

## 更新服务器

将 `6405proj-ui-update.tar.gz` 上传到服务器用户主目录。进入实际项目根目录（即同时含 app.py、scripts 和 proj6405 的目录），例如：

```bash
cd ~/6405_complete/6405proj
```

先保留旧前端文件，随后解压更新包：

```bash
backup_dir=$(mktemp -d "$PWD/ui-backup-XXXXXXXX")
cp app.py "$backup_dir/"
if [ -f .streamlit/config.toml ]; then
  cp .streamlit/config.toml "$backup_dir/streamlit-config.toml"
fi
tar -xzf ~/6405proj-ui-update.tar.gz -C .
```

更新包只含两个界面文件，里面没有模型、结果或实验配置。若你曾自行定制 `.streamlit/config.toml`，将原有非 theme 设置合并保留。

使用项目脚本管理 UI 时，只重启 UI 会话。先确认 `proj6405-ui` 确实是本项目的前端：

```bash
tmux list-sessions
tmux kill-session -t proj6405-ui
tmux new-session -d -s proj6405-ui -c "$PWD" bash scripts/run_ui.sh
tail -n 40 logs/ui.log
```

如果没有这个会话，跳过 kill-session。以上默认监听 127.0.0.1，适合通过 SSH 隧道访问；原来使用 RunPod HTTP 代理时，将启动命令替换为：

```bash
tmux new-session -d -s proj6405-ui -c "$PWD" -e UI_HOST=0.0.0.0 bash scripts/run_ui.sh
```

若原服务使用自定义 API_BASE_URL 或端口，保持原来的启动方式和设置。无需重启后端，因此原有 GPU 2 的进程配置保持不变。重新打开或刷新网页。

## 新界面操作

1. 左侧 **Dataset** 选择任务，**Input view** 选择共享前缀预算或原始输入。筛选同时作用于三个页面。
2. **Classify**：选择 **Paste text** 或 **Upload file**；也可点击 **Use example text**。右侧默认选择不同架构，具体 seed、训练样本数和 run ID 位于 **Configuration details**。
3. 点击 **Compare predictions**，逐个加载和预测。结果卡片显示英文类别、模型时间和输入 token；展开 **Timing breakdown & download** 查看详细时间并下载 CSV。改变输入后旧结果会明确提示需要重新预测，不会自动重发请求。
4. **Results & efficiency**：查看当前数据集与输入条件下的完整测试结果，切换 Macro-F1 / Precision / Recall / Accuracy。图表用百分数，下载 CSV 保留原始 0–1 指标。效率表支持 batch 和 precision 筛选，保留设备、样本签名和单位。
5. **Error explorer**：选择模型和文档，对照 Expected / Predicted 标签，阅读前 600 字符预览；可展开、下载最多 30 个导出的错误案例。

若模型尚未完成训练，界面显示英文空状态；某个结果接口失败不会阻断实时分类。SCOTUS 未知编码显示 Unknown category 和原始编码，避免错误猜测类别。所有真实成绩继续来自服务器已有的 runs。

## 验证范围

本地 Streamlit 1.42.0 下的 9 项 AppTest 检查通过，覆盖数据集/输入条件筛选、默认架构选择、示例填充、预测保留与过期提示、上传文件及 UTF-8 错误、英文法律类别、接口断连和 benchmark 筛选。浏览器布局检查使用独立本地 API 的合成测试数据，不能作为实验成绩；没有运行远程 GPU 训练。
