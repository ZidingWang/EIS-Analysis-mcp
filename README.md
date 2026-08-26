# EIS ECM DRT MCP

[中文](#中文) | [English](#english)

## 中文

用于 EIS 数据的 ECM 等效电路拟合和 DRT 反演。支持单个文件、多个文件或整个文件夹批量处理。

### 功能

- 自动识别频率、阻抗实部和虚部列
- ECM 多初值拟合并选择误差最低的结果
- DRT 计算及图片输出
- 支持 `.xlsx`、`.xls`、`.csv`、`.tsv`、`.txt`

### 环境要求

- Python 3.10 或更高版本（推荐 3.10–3.13）
- Git
- 第一次启动需要联网安装依赖

### 安装与启动

```powershell
git clone https://github.com/ZidingWang/EIS-Analysis-mcp.git
cd EIS-Analysis-mcp
.\run_mcp.cmd
```

macOS / Linux：

```bash
git clone https://github.com/ZidingWang/EIS-Analysis-mcp.git
cd EIS-Analysis-mcp
chmod +x run_mcp.sh eis-analysis.sh
./run_mcp.sh
```

第一次启动会自动创建 `.venv` 并安装依赖。依赖安装完成后，程序会保持运行并等待 MCP 客户端，这是 STDIO MCP 的正常状态。手动预安装时可按 `Ctrl+C` 结束，然后按下面步骤连接。

### 连接 MCP

本仓库提供的是本地 STDIO MCP，可连接任何支持 STDIO MCP 的 Agent。GitHub 仓库网址是源码地址，不能作为远程 MCP URL 直接连接。

#### 通用连接步骤

1. 在 Agent 的 MCP 设置中添加本地服务器。
2. 名称填写 `eis-ecm-drt`，传输类型选择 `STDIO`。
3. Windows 启动命令使用实际仓库中的 `run_mcp.cmd`；macOS / Linux 使用 `run_mcp.sh`。
4. 保存并重启或刷新 Agent，然后在工具列表中确认 EIS 工具已经出现。

#### Codex

可以打开 `设置 → MCP servers → Add server` 按上述方式添加，也可以把下面的配置加入 `~/.codex/config.toml`。

Windows：

```toml
[mcp_servers.eis-ecm-drt]
command = "cmd.exe"
args = ["/d", "/c", 'C:\path\to\EIS-Analysis-mcp\run_mcp.cmd']
startup_timeout_sec = 180
tool_timeout_sec = 3600
```

macOS / Linux：

```toml
[mcp_servers.eis-ecm-drt]
command = "/bin/sh"
args = ["/path/to/EIS-Analysis-mcp/run_mcp.sh"]
startup_timeout_sec = 180
tool_timeout_sec = 3600
```

把路径改成实际仓库路径，保存后重启 Codex；输入 `/mcp` 可以检查连接。Codex 模板：[Windows](examples/codex_config_windows.toml)｜[macOS / Linux](examples/codex_config_macos_linux.toml)

#### 使用 `mcpServers` JSON 的 Agent

Windows 配置模板：[examples/mcp_config_windows.json](examples/mcp_config_windows.json)

```json
{
  "mcpServers": {
    "eis-ecm-drt": {
      "command": "C:\\path\\to\\EIS-Analysis-mcp\\run_mcp.cmd",
      "args": []
    }
  }
}
```

macOS / Linux 配置模板：[examples/mcp_config_macos_linux.json](examples/mcp_config_macos_linux.json)

把模板中的路径改成实际仓库路径，然后在相应 Agent 中刷新连接。

### 分析 EIS

连接后可以直接输入：

```text
分析 D:\EIS数据
ECM 选择 L-R0-RWQ
DRT 使用推荐参数
```

正式计算前，MCP 会先展示全部 6 个 ECM 模型、全部 4 个 DRT 预设和自定义入口，再要求用户分别确认 ECM 与 DRT。推荐项不会代替用户确认。

也可以直接使用命令行：

```powershell
.\eis-analysis.cmd "D:\EIS数据" --ecm-model L-R0-RWQ
```

### ECM 模型

| 模型 ID | text 表达式 |
| --- | --- |
| `L-R0-RWQ`（推荐） | `L1-R0-((R1-W1)||CPE1)` |
| `L-R0-RQ` | `L1-R0-(R1||CPE1)` |
| `L-R0-RC` | `L1-R0-(R1||C1)` |
| `L-R0-2RC` | `L1-R0-(R1||C1)-(R2||C2)` |
| `L-R0-2RQ` | `L1-R0-(R1||CPE1)-(R2||CPE2)` |
| `L-R0-RQ-RWQ` | `L1-R0-(R1||CPE1)-((R2-W1)||CPE2)` |

所有含 `W1` 的固定模型都使用 `(R-W)` 串联支路。也可以输入解析器支持的自定义 text 表达式。

### DRT 参数

| DRT 预设 | λ 选择 | n_tau | FWHM 系数 | n_basis | 用途 |
| --- | --- | ---: | ---: | ---: | --- |
| `balanced`（推荐） | mGCV 自动选择 | 750 | 0.5 | 自动 | 常用 TR-RBF 分析 |
| `smooth` | 固定 1e-2 | 750 | 0.5 | 自动 | 噪声数据，曲线更平滑 |
| `high_resolution` | 固定 1e-4 | 1000 | 0.75 | 自动 | 保留更多细节，过拟合风险更高 |
| `fast_preview` | 固定 1e-3 | 300 | 0.5 | 60 | 快速预览 |

四组预设均使用实部/虚部联合的非负 TR-RBF、一阶 Tikhonov 正则化、Gaussian 基函数、modulus 权重和阻抗尺度归一化。推荐项使用 mGCV 为每条 EIS 自动选择 λ。`tau_min`、`tau_max` 均为 `null`，τ 范围按实际频率自动生成；用户也可以完整自定义 DRT 参数。

DRT 系数中心默认逐点对应实测频率的 `1/(2πf)`。`n_tau` 只控制 CSV 和图的采样密度，不增加实验分辨率。图上在直接支持范围两端各显示 0.5 decade 的曲线尾部，并用虚线标出 `1/(2πf_max)` 至 `1/(2πf_min)`；边界外只用于观察尾部，应谨慎解释。

### 输入与输出

输入数据至少需要频率、阻抗实部和虚部三列。程序会自动识别常见列名及虚部符号。

每个样本输出：

- `*_ecm.csv`：一组 ECM 参数和拟合指标
- `*_ecm_fit.png`：原始 EIS 与 ECM 拟合的 Nyquist 比较图
- `*_drt.csv`：DRT 数据
- `*_drt.png`：DRT 图
- `README_输出说明.txt`：结果说明

未指定输出目录时，程序会在当前用户桌面自动创建 `EIS Analysis output`，并为每次分析创建独立的时间戳子文件夹。用户明确指定 `output_dir` 时使用指定位置。

输出说明会记录实际 ECM 模型、DRT 预设和完整参数，以及曲线数量、成功/失败数量、频率范围、ECM 模型分布、relative RMSE、R² 和 DRT τ 范围等统计。

为便于和 ZView 对照，多个串联的同类型 RC/RQ 支路统一按特征频率从高到低编号。ECM CSV 同时给出 ZView 映射列：电阻为 Ω，电感为 H（另附 μH），电容为 F，`CPE_Q` 对应 ZView 的 `CPE-T`，`CPE_n` 对应 `CPE-P`。

### 许可证

[MIT License](LICENSE)

## English

MCP server for ECM fitting and DRT inversion of EIS files, file lists, and folders.

### Requirements

- Python 3.10 or newer (3.10–3.13 recommended)
- Git
- Internet access for first-run dependency installation

### Install and start

Windows:

```powershell
git clone https://github.com/ZidingWang/EIS-Analysis-mcp.git
cd EIS-Analysis-mcp
.\run_mcp.cmd
```

macOS / Linux:

```bash
git clone https://github.com/ZidingWang/EIS-Analysis-mcp.git
cd EIS-Analysis-mcp
chmod +x run_mcp.sh eis-analysis.sh
./run_mcp.sh
```

The first run creates `.venv` and installs the dependencies. The process then stays open waiting for an MCP client; this is expected for a STDIO server. If you ran it manually for setup, press `Ctrl+C`, then connect it as described below.

### Connect an MCP client or Agent

This repository provides a local STDIO MCP server for any Agent that supports STDIO MCP. The GitHub repository URL is source code, not a remote MCP server URL.

Add a local MCP server named `eis-ecm-drt`, choose `STDIO`, and use `run_mcp.cmd` on Windows or `run_mcp.sh` on macOS/Linux as the start command. Replace every example path with the actual cloned repository path, then restart or refresh the Agent.

For Codex, add the server through `Settings → MCP servers → Add server`, or use [the Windows Codex template](examples/codex_config_windows.toml) or [the macOS/Linux Codex template](examples/codex_config_macos_linux.toml) in `~/.codex/config.toml`. Restart Codex and enter `/mcp` to check the connection.

For Agents that use the `mcpServers` JSON format, use [the Windows JSON template](examples/mcp_config_windows.json) or [the macOS/Linux JSON template](examples/mcp_config_macos_linux.json).

### Analyze EIS

Submit a real `.xlsx`, `.xls`, `.csv`, `.tsv`, or `.txt` file or folder. Before analysis, the MCP displays all six ECM models, all four DRT presets, and both custom-input options. The user must explicitly confirm both ECM and DRT selections.

Recommended ECM:

```text
L-R0-RWQ = L1-R0-((R1-W1)||CPE1)
```

The six fixed model expressions are listed above. Every fixed Warburg model keeps W in series with R in the same branch, and custom parser-supported expressions are accepted.

DRT presets are `balanced` (recommended automatic mGCV), `smooth`, `high_resolution`, and `fast_preview`. The solver uses non-negative combined real/imaginary TR-RBF inversion with impedance-scale normalization. Complete custom DRT dictionaries are also accepted.

RBF centers follow the measured `1/(2πf)` points by default. `n_tau` controls only CSV/plot sampling. Plots show 0.5 decade of curve tails outside the directly supported `1/(2πf_max)` to `1/(2πf_min)` range and mark those limits with dotted lines. Custom `tau_min` and `tau_max` remain available.

Each sample produces:

- `*_ecm.csv`
- `*_ecm_fit.png`
- `*_drt.csv`
- `*_drt.png`
- `README_输出说明.txt`

When `output_dir` is omitted, the server creates `EIS Analysis output` on the current user's desktop if needed and adds a timestamped run folder. The generated README records the selected models/configuration and dataset and fit statistics.

For ZView comparison, repeated series RC/RQ branches are numbered from high to low characteristic frequency. ECM CSV files include explicit aliases and units: resistance in ohm, inductance in H (plus μH), capacitance in F, `CPE_Q` as ZView `CPE-T`, and `CPE_n` as `CPE-P`.

### License

[MIT License](LICENSE)
