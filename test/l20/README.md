# L20 四卡标定与验证

口径固定为 PyTorch 2.7.0+cu128、CUDA 12.8、四张 L20。标定顺序为矩阵、
向量、HBM、四卡 all-reduce、四卡 all-to-all；每一步写入独立原始结果和 manifest，
最后才生成 systems/L20_4GPU.json，避免把某次 quick smoke 数据混入正式曲线。

正式执行：

    python test/l20/calibrate_l20.py --execute

快速检查命令与 CUDA 环境：

    python test/l20/calibrate_l20.py --execute --quick

矩阵/向量曲线覆盖实际候选模型的 skinny GEMM 和小算子区间；网络必须同时测
all-reduce（DP）和 all-to-all（EP）。模型端验证使用 GPT-2 124M、Qwen3 0.6B
以及 DeepSeek-V4-2.7B-tiny MoE。MoE 配置包含 routed experts 和 shared expert。
