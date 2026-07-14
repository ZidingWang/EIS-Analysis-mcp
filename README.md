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

- Python 3.10 或更高版本
- 第一次启动需要联网安装依赖

### 安装与启动

```powershell
git clone <你的 GitHub 仓库地址>
cd EIS-Analysis-mcp
.\run_mcp.cmd
```

macOS / Linux：

```bash
git clone <your GitHub repository URL>
cd EIS-Analysis-mcp
chmod +x run_mcp.sh eis-analysis.sh
./run_mcp.sh
```

第一次启动会自动创建 `.venv` 并安装依赖。

### 连接 MCP

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

把模板中的路径改成实际仓库路径，然后在 MCP 客户端中刷新连接。

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

| DRT 预设 | λ | n_tau | shape_factor | n_basis | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `balanced`（推荐） | 10 | 750 | 4 | 120 | 常用平衡分析 |
| `smooth` | 30 | 750 | 5 | 100 | 噪声数据，曲线更平滑 |
| `high_resolution` | 3 | 1000 | 3 | 180 | 保留更多细节，耗时和过拟合风险更高 |
| `fast_preview` | 10 | 300 | 4 | 60 | 快速预览 |

四组预设均使用一阶正则化、Gaussian 基函数、非负约束和 modulus 权重。`tau_min`、`tau_max` 均为 `null`，τ 范围按每组输入 EIS 的实际频率自动生成。用户也可以完整自定义 DRT 参数。

### 输入与输出

输入数据至少需要频率、阻抗实部和虚部三列。程序会自动识别常见列名及虚部符号。

每个样本输出：

- `*_ecm.csv`：一组 ECM 参数和拟合指标
- `*_drt.csv`：DRT 数据
- `*_drt.png`：DRT 图
- `README_输出说明.txt`：结果说明

未指定输出目录时，程序会在当前用户桌面自动创建 `EIS Analysis output`，并为每次分析创建独立的时间戳子文件夹。用户明确指定 `output_dir` 时使用指定位置。

输出说明会记录实际 ECM 模型、DRT 预设和完整参数，以及曲线数量、成功/失败数量、频率范围、ECM 模型分布、relative RMSE、R² 和 DRT τ 范围等统计。

### 许可证

[MIT License](LICENSE)

## English

MCP server for ECM fitting and DRT inversion of EIS files, file lists, and folders.

### Requirements

- Python 3.10 or newer
- Internet access for first-run dependency installation

### Install and start

Windows:

```powershell
git clone <your GitHub repository URL>
cd EIS-Analysis-mcp
.\run_mcp.cmd
```

macOS / Linux:

```bash
git clone <your GitHub repository URL>
cd EIS-Analysis-mcp
chmod +x run_mcp.sh eis-analysis.sh
./run_mcp.sh
```

Use [the Windows MCP template](examples/mcp_config_windows.json) or [the macOS/Linux template](examples/mcp_config_macos_linux.json), replace the repository path, and refresh the MCP connection.

### Analyze EIS

Submit a real `.xlsx`, `.xls`, `.csv`, `.tsv`, or `.txt` file or folder. Before analysis, the MCP displays all six ECM models, all four DRT presets, and both custom-input options. The user must explicitly confirm both ECM and DRT selections.

Recommended ECM:

```text
L-R0-RWQ = L1-R0-((R1-W1)||CPE1)
```

The six fixed model expressions are listed above. Every fixed Warburg model keeps W in series with R in the same branch, and custom parser-supported expressions are accepted.

DRT presets are `balanced` (recommended), `smooth`, `high_resolution`, and `fast_preview`. Complete custom DRT dictionaries are also accepted. All presets leave `tau_min` and `tau_max` null so the τ range is derived from each input spectrum's frequencies.

Each sample produces:

- `*_ecm.csv`
- `*_drt.csv`
- `*_drt.png`
- `README_输出说明.txt`

When `output_dir` is omitted, the server creates `EIS Analysis output` on the current user's desktop if needed and adds a timestamped run folder. The generated README records the selected models/configuration and dataset and fit statistics.

### License

[MIT License](LICENSE)
