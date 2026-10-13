# EHPC 有界 smoke

在 manager 的共享 home 中准备本项目及 Python >= 3.10、NumPy >= 1.24。
本轮独立环境为 `/home/zhangyq/venvs/noahmp-closeout`。
脚本固定 100 个异质合成柱、48 个半小时步、float64、seed 20261013。

```bash
cd /home/zhangyq/work/noahmp-closeout-m3-20261013/repo
qsub benchmarks/ehpc/serial_pool.pbs
# 先确认上一作业 Exit_status=0，生成 outputs/ehpc/serial.npz。
qsub benchmarks/ehpc/mpi_two_nodes.pbs
qstat -u zhangyq
qstat -xf JOB_ID
```

如使用其他环境，通过 `qsub -v NOAHMP_PYTHON=/absolute/path/python` 覆盖。
可用 `NOAHMP_OUTPUT` 指定另一个输出目录，两个作业必须使用同一值。
PBS 文件需为 LF 换行；Windows 可通过 git 或文本传输显式转换。

串行与四进程在同一 PBS 作业内运行；MPI 申请两个不同节点，每节点一个
rank，按全局柱号奇偶分柱。模型柱之间无状态交换，rank 在 GPFS 写独立
结果，`mpirun` 完成后 gather 校验每柱恰好出现一次、确实执行在两个主机、
各 rank 源码 hash 相同，并对完整终态、整数控制状态与全部 48 步诊断比较。
float64 对照门固定为 `rtol=1e-6, atol=1e-6`，另报告是否逐位相同。
MPI 是 Open MPI 启动/资源分配与分柱执行检查，不是 MPI 通信性能基准。

JSON 保存源码 hash、输入 hash、完整输出 checksum、软硬件/节点、计时和
差异；NPZ 保留完整结果。计时为 smoke 实测，没有预热/三重复，不当作
正式性能基准。结果及限制汇总见 `docs/ENV_TESTS.md`。
