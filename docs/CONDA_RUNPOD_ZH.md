# Conda + RunPod 操作说明

本项目统一使用 Conda。当前执行范围以 [完整实验方案](FULL_PROJECT_ZH.md) 为准；六小时方案仅作备选。本页专门说明环境安装与激活。

## 1. 环境放在哪里

推荐把项目解压到 `/workspace/6405proj`。安装脚本会建立：

```text
/workspace/6405proj/.conda-env/       Python 3.11 和项目依赖
/workspace/6405proj/.conda-executable 服务器上 Conda 可执行文件的位置
/workspace/6405proj/.cache/           下载和 Conda 包缓存
```

这是 Conda 的路径环境（`--prefix`）。激活时使用完整路径，不是 `conda activate 6405proj`。项目文件和环境都应放在真正挂载的持久化卷内；`.conda-env` 不能直接从 Windows 复制到 Linux，也不要在建立环境后移动目录。

Conda 管理 Python；环境内的 `python -m pip` 安装固定版本的 PyTorch、Transformers 和 Mamba。所有包都装进同一 Conda 环境。脚本不向 base 环境安装项目包，也不再创建或激活 `.venv`；如果有旧 `.venv`，它会保留，新的脚本不使用它。

## 2. 服务器已安装 Conda

在 Linux 终端检查：

```bash
conda --version
conda info --base
```

即使非交互终端中没有 `conda` shell 函数，脚本也会检查常见位置及已保存的可执行文件位置。已有 Conda 安装在其他位置时，可明确指定：

```bash
export PROJ6405_CONDA_EXE=/你的实际路径/bin/conda
```

安装脚本采用 `--override-channels --channel conda-forge` 创建 Python 环境，不改全局 `.condarc`，不自动执行 `conda init`。

## 3. 服务器没有 Conda 时

已有 Conda 就跳过本节。可使用包含 Conda 的 Miniforge，安装在 `/workspace/miniforge3`；下面命令仅适用于 Linux x86_64。不要向已存在的安装目录重复安装。

```bash
cd /workspace
curl -fL --retry 2 \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
  -o Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
source /workspace/miniforge3/etc/profile.d/conda.sh
conda --version
```

这是官方 Miniforge 的安装方式。环境安装、下载都占用 Pod 时间，应计入预算。Miniforge 的包管理器 `mamba` 与本项目研究的 Mamba 神经网络是不同软件；模型依赖仍由下一节的 `install_mamba.sh` 安装。

## 4. 一次性创建环境和安装依赖

```bash
cd /workspace/6405proj
bash scripts/setup_conda.sh
bash scripts/install_mamba.sh
source scripts/activate.sh

python --version
python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip check
python -m proj6405.doctor
```

预期 Python 路径位于 `/workspace/6405proj/.conda-env/bin/python`，版本为 3.11；PyTorch 为 2.5.1 的 CUDA 12.1 构建；在正常 GPU Pod 上 `torch.cuda.is_available()` 为 True。Mamba 安装后，doctor 的 `mamba_fast_path` 应为 True。

`setup_conda.sh` 创建 Python 环境并安装基础依赖；Mamba 扩展单独安装，便于定位编译问题。重复运行会复用同一环境并检查固定依赖。若 `.conda-env` 存在但缺少完整 Conda 元数据，脚本会停止，避免覆盖未知文件。

旧的 `bash scripts/setup_linux.sh` 入口仍可使用，现在转到 Conda 安装脚本。推荐直接使用 `setup_conda.sh`，不要再使用旧文档中的 `PYTHON_BIN=...` 或 `.venv/bin/activate`。

Conda 不会提供宿主机 GPU 驱动。PyTorch CUDA wheel 提供运行库；Mamba 如果需要源码编译，仍需要镜像中的匹配 CUDA toolkit、`nvcc` 和 C++ 编译器。使用与 cu121 匹配的开发环境；不要随意升级 GPU 驱动或同时混装另一套 Conda PyTorch。

## 5. 每次重新登录或进入新 tmux 终端

```bash
cd /workspace/6405proj
source scripts/activate.sh
```

这个命令同时激活项目 Conda 环境和项目缓存、线程配置。必须用 **source**，`bash scripts/activate.sh` 无法改变父终端环境，会被脚本拒绝。不需要重新安装依赖，也不要求登录终端预先激活 Conda。

如果 Conda base 在停机后消失，先恢复 Miniforge/Conda，再指定新的 `PROJ6405_CONDA_EXE`。环境目录本身也不存在时，重新运行 setup。只保留 `.conda-env` 不等于保存宿主机 CUDA toolkit 和所有系统工具。

## 6. 完整实验命令

```bash
cd /workspace/6405proj
source scripts/activate.sh
python -m proj6405.download
python -m proj6405.doctor --smoke mamba --length 512

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -m proj6405.full --phase plan
python -m proj6405.full --phase prepare
# 按主手册先完成 pilot、4K 训练预检与验证集参数选择。
bash scripts/stop_services.sh
python -m proj6405.full --phase train --stage core
python -m proj6405.full --phase train --stage long
python -m proj6405.full --phase evaluate
python -m proj6405.full --phase benchmark
python -m proj6405.full --phase report
```

每条命令成功后再进入下一步；训练不完整时按主手册处理。完整版没有阶段计时上限，也不会自动停止 Pod；分阶段运行并查看日志。

在 tmux 内训练：

```bash
tmux new -s full-train
# 进入 tmux 后执行；不要同时在另一个终端再跑一份训练。
cd /workspace/6405proj
source scripts/activate.sh
python -m proj6405.full --phase train --stage core
```

可按 Ctrl+B、松开、再按 D 离开会话；`tmux attach -t full-train` 返回。激活脚本会重新建立正确环境，不依赖旧 tmux 服务进程保存的 PATH。

## 7. RunPod 前后端启动

确认 Pod 已配置 HTTP 端口 8501，然后执行：

```bash
cd /workspace/6405proj
bash scripts/stop_services.sh
RUN_PREFIX=full_ UI_HOST=0.0.0.0 bash scripts/start_services.sh
curl http://127.0.0.1:8000/health
```

API 和 Streamlit 会各自在自己的后台进程中激活同一个 Conda 环境。通过 RunPod Connect → HTTP Services → 8501 打开网页。服务重启时同样不需要重新安装环境。

## 8. 复现记录与故障定位

- `reports/conda-explicit-linux64.txt`：Conda 包来源与构建记录。
- `reports/environment_base.txt`、`environment_full.txt`：基础安装 / Mamba 安装后的 pip freeze。
- `reports/environment.json`：doctor 的 Python、PyTorch、CUDA、GPU 和快速路径检查。
- `.conda-executable`：本机配置，不打进源码包；换服务器重新运行 setup。

Conda 的显式记录与 pip freeze 需要一起保留，前者不能完整替代 pip 包清单。备份结果和模型即可，通常不需要压缩 `.conda-env` 上传。

| 现象 | 处理 |
|---|---|
| Conda 找不到 | 使用正确的 `PROJ6405_CONDA_EXE`，或按第 3 节安装 Miniforge |
| 环境不存在 | 在项目目录运行 `bash scripts/setup_conda.sh` |
| 当前 Python 不是 3.11 或路径不对 | 重新 `source scripts/activate.sh`；不要改用系统 pip |
| 旧服务仍用旧环境 | 先 stop_services，再 start_services |
| CUDA 不可用 | 检查 Pod GPU、驱动与 PyTorch 构建；Conda 激活不等于 GPU 可用 |
| Mamba 编译失败 | 检查 nvcc/toolkit/编译器与固定依赖的兼容性，先通过 doctor 再训练 |

官方参考：[Conda create](https://docs.conda.io/projects/conda/en/stable/commands/create.html)、[Miniforge](https://github.com/conda-forge/miniforge)、[PyTorch 2.5.1 安装命令](https://pytorch.org/get-started/previous-versions/#v251)。
