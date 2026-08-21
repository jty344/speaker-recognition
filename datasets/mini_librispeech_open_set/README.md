# Mini LibriSpeech 开放集声纹评测

本目录用于验证“注册若干人后，将单说话人 MP3 识别为姓名或 `UNKNOWN`”的完整 CPU 处理链。数据来自 [OpenSLR SLR31](https://www.openslr.org/31/) 的 Mini LibriSpeech `dev-clean-2`，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。

原始文件是单说话人 FLAC，路径中的 speaker ID 是真值标签。评测准备脚本将选中的源文件一对一转为 mono/16 kHz/128 kbit/s MP3，不拼接、不切片，也不让同一源音频出现在多个角色中。

## 目录

```text
archives/dev-clean-2.tar.gz       官方原始归档
source/LibriSpeech/dev-clean-2/   解压后的 1089 条 FLAC
mp3/enroll/                       已注册人的注册 MP3
mp3/queries/known/                已注册人的查询 MP3
mp3/queries/unknown/              未注册人的查询 MP3
protocol/open_set_manifest.json   固定划分及每条音频的 expected 真值
results/open_set_evaluation.json  每条 predicted、correct、得分及汇总指标
```

归档校验：

```text
大小     126046265 bytes
MD5      6d7ab67ac6a1d2c993d050e16d61080d
SHA-256  176ec501490eced2d6c1f89f4f0ddc7dfe799e649e5322f8ba49fe3ff50c8012
```

## 固定协议

- 固定 seed：`20260821`。
- 8 名 known speaker，每人 3 条注册、3 条查询，共 24 条注册和 24 条 known query。
- 8 名完全不相交的 unknown speaker，每人 3 条查询，共 24 条 unknown query。
- 每名 known speaker 的注册与查询来自不同 chapter。
- 72 个角色样本均来自不同源文件；源音频静音裁剪后至少 2.5 秒。
- known query 的 `expected` 是注册姓名；unknown query 的 `expected` 固定为 `UNKNOWN`。
- 基线固定使用 Demo 默认阈值 `0.5`，不使用这批测试查询反向调参。

完整名单、源路径、chapter、音频角色和逐条真值以
[`protocol/open_set_manifest.json`](protocol/open_set_manifest.json) 为准。

## 复现

下载、校验、解压并生成 MP3 与真值协议：

```bash
cd /mnt/code/test_env1
./scripts/prepare_open_set_dataset.sh
```

运行 ONNX Runtime CPU 评测：

```bash
./scripts/evaluate_open_set.sh
```

脚本会覆盖生成 `results/open_set_evaluation.json`，其中每条查询同时保存 `expected`、`predicted`、`correct`、top-1 分数和候选列表。

## 当前基线

模型为 WeSpeaker CN-Celeb ResNet34-LM ONNX，阈值为未校准的 `0.5`，CPU 线程数为 1。

| 指标 | 结果 |
| --- | ---: |
| 已注册人识别准确率 | 24/24，100.00% |
| 已注册人误拒率 | 0/24，0.00% |
| 未注册人 `UNKNOWN` 召回率 | 10/24，41.67% |
| 未注册人误接收率 FAR | 14/24，58.33% |
| 总体开放集准确率 | 34/48，70.83% |
| Balanced accuracy | 70.83% |
| 查询处理错误 | 0/48 |

该结果说明 MP3 解码、FBank、ONNX 推理、注册、姓名匹配和 `UNKNOWN` 判定已经端到端工作；同时也说明默认阈值 `0.5` 对这组开放集样本过于宽松。若要调整阈值，应另建不与本测试查询重叠的校准集，冻结阈值后再重跑本协议。

## 适用边界

Mini LibriSpeech 是干净的英文有声书语音。chapter 隔离比随机抽取 utterance 更严格，但不等同于跨麦克风、跨设备或跨真实环境。因此这里的数字适合 Demo 功能验证和回归比较，不代表中文、噪声、远场或安全认证场景中的生产准确率。
