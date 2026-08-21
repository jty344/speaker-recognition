# 测试音频

`upstream/` 保存公开项目提供的短语音样本，仅用于验证处理链；`generated/` 保存由 GStreamer 在本机转码出的 MP3，随时可以重新生成。

上游样本来源于 SpeechBrain 仓库的测试资源：

- `tests/samples/ASR/spk1_snt1.wav`
- `tests/samples/ASR/spk1_snt2.wav`
- `tests/samples/ASR/spk2_snt1.wav`

来源仓库：<https://github.com/speechbrain/speechbrain>，下载固定到 commit
`e5cb1f65b940634215650aa1171e0440d0808123`，并在转码前检查每个 WAV 的
SHA-256。可运行 `scripts/make_test_mp3.sh` 重新下载和转码。
