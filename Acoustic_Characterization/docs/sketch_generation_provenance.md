# 草图生成脚本溯源

本仓库的 `scripts/batch_extract_sketches.py` 来源于原始草图处理目录中的：

```text
E:\Sketch\sketch_process\sketch\batch_extract_sketches.py
```

该脚本负责递归扫描 WAV 文件，并生成包含 `pitch_sketch` 和 `energy_sketch` 的 JSON 文件。仓库版本保留原脚本的特征提取算法、参数、Savitzky–Golay 平滑、min–max 归一化、断点跳过、多进程处理和 JSON 字段结构。

## 代码一致性核验

原始脚本的 SHA-256 为：

```text
B51C85866114FB1FD66ABCE45E2591C1DBE1D2290E8058E867F9592BF3B8BD0B
```

整理时仅将原脚本中直接写死的输入、输出目录改为从 `scripts/project_config.py` 读取 `CDSD_PROJECT_DATA` 后拼接 `wav/` 和 `processed_sketches_new/`。为实现这一配置读取而增加的导入与路径变量属于路径适配；`process_single_audio`、`worker` 和 `batch_process_parallel` 的算法函数体没有改动。

因此，仓库版本用于相同输入数据时，除输入/输出位置由配置决定外，生成过程和输出 JSON 结构与实际实验使用的原始脚本保持一致。

## 运行前提

将 `CDSD_PROJECT_DATA` 配置为包含以下两个目录的数据根目录：

```text
<CDSD_PROJECT_DATA>\\wav
<CDSD_PROJECT_DATA>\\processed_sketches_new
```

运行：

```powershell
python scripts/batch_extract_sketches.py
```

原始 WAV 和生成的草图 JSON 属于私有数据，不随代码仓库提交。
