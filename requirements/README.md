# Python 依赖分类

- `runtime.txt`：NumPy、ONNX 模型检查和 ONNX Runtime CPU 推理；
- `audio-fbank.txt`：PyTorch CPU 与 TorchAudio FBank；
- `dev.txt`：测试工具；
- `runtime-lock.txt`：本机验证通过后生成的完整版本快照。

所有依赖只安装到 `.runtime/venvs/voiceprint/`。

