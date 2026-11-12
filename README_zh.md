# Irodori-TTS-Desktop

[English](README.md)

基于 PySide6 和 QFluentWidgets 的日语语音生成与 LoRA 音色训练桌面应用，使用固定版本的
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)。

## 功能

- 用基础模型或训练好的音色生成语音，可组合声音描述与参考录音；每个模型记住上次用它时的参数。
- 长文本自动分段朗读；可以从面板插入模型支持的表情符号标签。
- 可以用更少显存生成，或用 4 步模型更快地生成。
- 没有文本的录音也能导入：一个模型写出文本，第二个模型交叉校对读音，只有两者不一致的录音需要校对。
- 根据录音和显卡推荐训练参数并训练 LoRA 音色；可以停止、继续、在已完成的音色上接着训练，
  训练期间可以试听样例。
- 保存生成历史，导出音频、音色或整个项目。
- 中英文界面，亮色与暗色主题。

## 运行要求

Windows x64、NVIDIA 显卡和较新的驱动。朗读的文本为日语。
请为模型（约 3 GB，每个转写模型另需 2–5 GB）和训练检查点预留磁盘空间。

## 使用打包版

解压 `Irodori-TTS-Desktop.7z`（Windows 11 可直接解压，其他系统用 7-Zip），运行其中的
`Irodori-TTS-Desktop.exe`。请保持文件夹完整；包里自带 Python，
不需要另外安装任何东西。

## 从源码运行

```powershell
git submodule update --init --recursive
uv sync --python 3.10 --extra cu128
. ./local-env.ps1
.venv/Scripts/python.exe desktop_app.py
```

首次启动会打开 **开始使用**：检查电脑，下载语音模型和推荐的转写模型；下载中断后会从中断处继续。
连不上网时，用 **从文件导入** 导入模型包（.zip 或 .7z）或装着模型文件的文件夹，例如别的电脑导出的
模型文件（对方在 **设置 → 导出已下载的模型**）。每个文件都会先校验再使用。
首次启动有一次操作引导，**设置 → 重新开始引导** 可以再看一遍。

1. 在 **生成** 页用基础模型试一下。
2. 在 **模型** 页 **新建模型** 并 **导入语音**，文本会自动转写；已有 CSV 或 TXT 文本的录音也可以导入。
3. 校对标为 **待校对** 的录音。
4. 在 **训练** 标签页采用推荐参数并开始；需要时会先自动预处理。
5. 训练期间在已保存的检查点旁点 **生成样例** 试听，完成后用该音色生成与导出。

请使用有权使用的录音。内置示例是合成语音。

**设置 → 生成模型** 可以选择生成时使用的模型，需先在 **开始使用** 中下载：

- **低显存**（约 0.9 GB）：生成时显存占用少约四分之一。
- **快速（4 步）**（约 3.1 GB）：采样步数从 40 步降到 4 步，步数和引导强度由模型决定。

两者都可以使用训练好的音色；训练仍使用语音模型。

### 转写模型

超过 12 秒的录音会在停顿处切开。

| 模型 | 下载 / 显存 | 适用场景 |
|---|---|---|
| whisper-large-v3（默认） | 3.1 GB / 约 8 GB | 读音最准确；真人录音和日常日语 |
| Qwen3-ASR-1.7B（默认交叉校对） | 4.7 GB / 约 5 GB | 汉字与假名用字规范；作为第二意见 |
| anime-whisper | 3.0 GB / 约 4 GB | 游戏、动漫语音；很多词写成假名 |
| Fun-ASR-Nano | 2.0 GB / 约 4 GB | 体积最小；游戏语音听错较多 |

## 数据

用户数据默认在 `Irodori-TTS-Desktop.exe` 旁边的 `data` 文件夹（源码运行时在 `desktop_app.py`
旁边），可以用 `--data-dir path/to/data` 指定其他位置。移动或备份项目时请保留整个项目文件夹。

## 工程结构

`upstream/Irodori-TTS` 是子模块，指向我们对原项目的 [fork](https://github.com/normalllll/Irodori-TTS)；
计算代码的修改都在 fork 里。本仓库只包含桌面应用；`irodori_cli.py infer` 与 `irodori_cli.py train`
运行 fork 中的命令行工具。开发说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

本项目采用 GPL-3.0-only（[LICENSE](LICENSE)），基于 Aratako 的
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（MIT，见 `upstream/Irodori-TTS/LICENSE`）。
模型和其他依赖遵循各自的许可。安装包附带 FFmpeg（GPLv3，见 `runtime/ffmpeg/LICENSE.txt`）和
7-Zip（见 `runtime/7zip/License.txt`）。
