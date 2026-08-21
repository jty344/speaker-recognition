# Python 依赖分类

- `runtime.txt`：NumPy、ONNX 模型检查和 ONNX Runtime CPU 推理；
- `audio-fbank.txt`：PyTorch CPU 与 TorchAudio FBank；
- `dev.txt`：测试工具；
- `gstreamer-aac.txt`：AAC 解码所需 Ubuntu 软件包的下载地址、SHA-256 和工作区内目标文件；
- `runtime-lock.txt`：本机验证通过后生成的完整版本快照。

所有 Python 依赖只安装到 `.runtime/venvs/voiceprint/`。

音频格式按文件名后缀选择 GStreamer parser，支持 `.mp3`、`.wav`、`.flac` 和 `.aac`（大小写不敏感）。`.aac` 特指裸 ADTS AAC，不包含 `.m4a` 等 MP4 容器。

系统当前没有 AAC decoder。`scripts/bootstrap.sh` 会按照 `gstreamer-aac.txt` 下载并校验固定版本的 `gstreamer1.0-plugins-bad` 和 `libfaad2` 软件包，在 `.runtime/gstreamer/` 中解包，并且只把 FAAD GStreamer 插件和 `libfaad` 放入实际加载路径；`scripts/env.sh` 再通过 `GST_PLUGIN_PATH_1_0` 和 `LD_LIBRARY_PATH` 加载这些工作区文件。此过程不调用 `apt`，也不会写入系统 GStreamer 或系统动态库目录。

上述二进制固定用于 Ubuntu 22.04 x86_64。FAAD2 以及实际加载它的插件按 GPL 许可提供，因此只做本地下载，不提交其二进制；如需重新分发，必须另行审查许可证义务。
