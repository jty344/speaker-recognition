# CPU 开放集声纹识别 Demo

目标：对单说话人音频注册声纹，之后输入新的音频，返回已注册人员姓名或 `UNKNOWN`。

处理链：

```text
MP3/WAV/FLAC/AAC → 按文件后缀选择 GStreamer parser
                 → mono/16kHz/float32 PCM
                 → 80维 Kaldi FBank → WeSpeaker ResNet34-LM ONNX
                 → 256维 L2 embedding → cosine → 姓名或 UNKNOWN
```

## 工作区隔离

所有新增内容均保存在当前项目中：

```text
.runtime/                 Python bootstrap 与隔离虚拟环境
.cache/                   pip、XDG、Hugging Face、Torch 缓存
.tmp/                     临时文件
requirements/             按用途分类的依赖与完整锁定版本
models/                   ONNX、配置、来源和 SHA-256
src/voiceprint/            业务代码
data/private/              用户自己的音频
data/test_assets/          可重复生成的测试音频
datasets/                  独立的公开评测集、真值协议与结果
artifacts/registry/        NPZ 声纹中心与 JSON 元数据
```

不使用 `apt`、`pip --user` 或用户级模型缓存。系统现有 GStreamer 负责音频管线；AAC 解码所需的 FAAD 插件和动态库由初始化脚本校验后解压到 `.runtime/gstreamer/`，不新增或修改系统文件。

## 初始化

```bash
cd /mnt/code/test_env1
./scripts/bootstrap.sh
```

脚本会创建 `.runtime/venvs/voiceprint`，安装 CPU 版 PyTorch、TorchAudio、ONNX Runtime，准备工作区内的 GStreamer FAAD 插件，并下载经过 SHA-256 校验的模型。

AAC runtime 当前固定为 Ubuntu 22.04 x86_64 软件包。FAAD2 及加载它的 GStreamer 插件按 GPL 许可提供；脚本仅在本机下载到已忽略的 `.runtime/`。若以后要随产品重新分发这些二进制，需要单独完成许可证合规审查。

## 支持的音频格式

输入格式根据文件名的最后一个后缀判断，后缀大小写不敏感：`.mp3`、`.wav`、`.flac` 和 `.aac`。`.aac` 仅表示裸 ADTS AAC；MP4 容器的 `.m4a` 暂不支持。文件内容应与后缀一致，否则对应的 GStreamer parser 会解码失败。

## 基本使用

查看音频编码信息：

```bash
./scripts/voiceprint.sh inspect data/private/query.mp3
```

查看完整 256 维声纹：

```bash
./scripts/voiceprint.sh embed data/private/query.mp3
```

每名说话人建议使用 3～5 段独立、清晰、只有一人说话的录音：

```bash
./scripts/voiceprint.sh enroll \
  --speaker alice \
  data/private/alice/enroll-1.mp3 \
  data/private/alice/enroll-2.mp3 \
  data/private/alice/enroll-3.mp3
```

识别和验证：

```bash
./scripts/voiceprint.sh identify data/private/query.mp3 --top-k 3
./scripts/voiceprint.sh verify --speaker alice data/private/query.mp3
./scripts/voiceprint.sh list
./scripts/voiceprint.sh remove --speaker alice
```

已存在的姓名默认不覆盖；重新注册时加 `--replace`。

## UNKNOWN 阈值

未传 `--threshold` 时使用 `0.5` 作为未校准的演示值，输出中会标记 `demo_default_uncalibrated`。实际阈值必须由自己的独立开发录音确定：

```bash
./scripts/voiceprint.sh identify data/private/query.mp3 --threshold 0.62
```

输出的 `score` 是余弦相似度，不是概率或置信度。

## 测试

生成测试 MP3 并运行测试：

```bash
./scripts/make_test_mp3.sh
source scripts/env.sh
python -m pytest
```

测试音频只验证处理链；真实准确率与麦克风、环境、语音时长、人员规模和阈值有关。

## 带真值的开放集评测

项目内已准备 Mini LibriSpeech `dev-clean-2` 的固定开放集协议：8 名已注册人和 8 名未注册人，共 24 条注册 MP3、24 条已知查询和 24 条未知查询。每条查询都有 `expected` 真值，评测结果同时保存 `predicted` 与 `correct`。

```bash
./scripts/prepare_open_set_dataset.sh
./scripts/evaluate_open_set.sh
```

默认阈值 `0.5` 的当前结果为：已注册人 24/24 正确，未注册人 10/24 正确返回 `UNKNOWN`，总体 34/48，即 70.83%。数据来源、划分约束、校验值、逐条结果和限制见 [`datasets/mini_librispeech_open_set/README.md`](datasets/mini_librispeech_open_set/README.md)。

## 模型

- 模型：`Wespeaker/wespeaker-cnceleb-resnet34-LM`
- ONNX 输入：`[batch, time, 80]` FBank
- 输出：256 维 speaker embedding
- 模型文件约 26.5 MB
- 来源与摘要：`models/wespeaker-cnceleb-resnet34-lm/MANIFEST.json`

模型页面标记 Apache-2.0，但 WeSpeaker 说明预训练权重还应遵循训练数据许可；CN-Celeb 数据集不允许未经授权的商业使用。本项目定位为本地学习 Demo。

## 范围限制

- 每个文件只允许一名主要说话人；
- 静音裁剪后有效语音至少 2 秒，推荐 3～10 秒；
- 只能识别已注册人员，否则返回 `UNKNOWN`；
- 不包含多人分离、防重放、反合成或活体检测，不能作为安全认证系统。
