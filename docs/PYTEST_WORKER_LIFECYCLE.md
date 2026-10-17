# GPU测试独立工作进程（2026-10-17，完整队列已采用）

20:16UTC，完成大文件验证后，[新完整队列](v1-results/full-worker-c-launch-a/manifest.json)
在GPU1启动。逐节点核对当前1129=CPU804+GPU325：原17文件85个正常退出的
设备节点，加新工作进程19个节点，共104；余下221节点/43文件继续执行。
原483份代码/测试/基准与新增评价器快照一致，物理库174份未改，原CPU780
与新增24的实际XML并集恰为804，交叠55不重复计数。这是分布式完整清单的
续跑，不是已完成1129或一次pytest命令全过。

当前完整设备队列的灌溉文件出现解释器退出等待。19:41:15UTC对本任务
PID4024593只读采样：主线程D状态、`__folio_lock_or_retry`，Python栈为
JAX `clear_backends → clean_up`。源486份未变，17个完整文件已有进程退出记录；
该灌溉文件尚未由控制器取得退出码，不能据JUnit/日志就计完整队列通过。
[无局部变量/环境的原快照](v1-results/full-irrigation-wait-a/manifest.json)保留。
该瞬间不能说明全部耗时都在清理，也未证明内存、交换或文件系统的唯一原因。

原型对每个文件开启独立进程，仅一次调用原`pytest.main`，不改测试/选项/
同精度或守恒门。返回后核对预先收集的完整节点、全部setup/call/teardown
报告、JUnit计数、测试和源字节；对已载入JAX调用effects barrier，对CuPy调用
CUDA设备同步。实际错误、同步错误、跳过或缺覆盖均非零。记录并刷写结果后
才使用`os._exit`退出，明确省略Python解释器atexit，未验证JAX正常关闭行为。
API语义见[pytest官方说明](https://docs.pytest.org/en/stable/how-to/usage.html#calling-pytest-from-python-code)、
[CuPy设备同步](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.runtime.deviceSynchronize.html)与
[Python直接退出定义](https://docs.python.org/3/library/os.html#os._exit)。

EHPC PBS107 compute002的7个隔离控制案例在2.199848秒完成，结果按顺序为：
正常0、断言失败1、跳过70、teardown错误1、模拟同步错误70、覆盖缺失70、
测试文件变化70。仅正常案例可记通过；这七项是生命周期控制，不能加到804个
CPU物理测试或325个GPU测试覆盖中。
[每个子进程的原日志/XML/记录](v1-results/pytest-worker-controls-a/manifest.json)保留。

xuanze原单叶6项已在同一工作进程完成，真实进程退出0，全部节点/阶段/源/XML一致，
JAX与CuPy同步成功。pytest为8.809346秒，外部控制器9.114668秒；
[原型源码、全部物理输出与退出记录](v1-results/gpu-worker-first-file-a/manifest.json)保留。
随后整柱灌溉13项于20:11UTC实际退出0：pytest为1538.988687秒，外部控制器
1543.049099秒，全部原setup/call/teardown、JAX/CuPy同步、节点/XML与源哈希
核对完成。四种柱ABI各两步的CUDA32/JAX32全字段最大差0；JAX32对NumPy的
H/LE最大绝对差0.004919813/0.001287805W/m²，JAX64为
1.77636e-11/1.50919e-11W/m²。CUDA32/JAX32独立水/热/C/N残差最大分别
1.11684e-5kg/m²、18.467983J/m²、0.000606805gC/m²、2.78875e-5gN/m²；
JAX64为8.61533e-14、0.000149256、4.06682e-12、1.13687e-13（同单位顺序）。
仍逐柱使用原数值界，不以不同柱最大界或10%科学口径豁免守恒。
[全部原日志、退出记录及逐行数值](v1-results/gpu-worker-complete-a/manifest.json)保留。

原普通CLI文件也打印13 passed/1541.92秒，但其退出清理PID4024593仍存活且
没有可用的进程退出0记录，没有用它替代新执行结果。仅停止旧控制器
PID3968670，SIGINT未终止后按原身份对该PID发送SIGTERM；未向旧清理子进程
发送信号，也未将它标为终止。旧17文件输出完整保留，年度身份核对未变。
新旧测试体耗时不能据此作统计速度比较；这项绕行针对解释器清理等待。
余下完整清单、年度及科学验收仍须实际完成。

## 插桩与执行文件清单

20:28UTC，完整C队列的多层柱文件实际退出1：四项通过（含两精度参考、
独立守恒及CUDA32/JAX32全字段最大差0），第五项导入
`benchmarks.check_clm_enthalpy_column_gpu`失败。原因是脚本工作进程的
PYTHONPATH包含src/tests/benchmarks而缺少仓库根目录；只读模块查找对照
确认补入根目录可定位该namespace。该文件不计完整通过，原失败、XML、
完整节点与查找记录保留在
[独立证据](v1-results/full-c-import-failure-a/manifest.json)。
完整C其余文件继续执行，后续用修正启动环境补跑失败文件；不改物理、原测试
或已运行进程的环境，不重跑已核实的完整成功文件。1129项完整覆盖仍未达到。

20:45UTC补跑D的逐节点预检完成，等待控制器PID4114860已启动；它等待C真实
终态summary后，逐文件核对成功的原退出码/全部阶段/同步/XML/节点/源身份，
复用成功文件，再以包含仓库根目录的PYTHONPATH补跑余项。预检时仍是已完成
GPU104/待221，D本身尚未执行测试；
[等待器/执行源码和实际预检](v1-results/full-d-deferred-a/manifest.json)保留。

生产源码/物理测试修改为0。本次新增的诊断/执行原件在忽略目录outputs/v1与
远端独立任务目录；相应源码和输出随上述证据归档，可独立清理。

| 文件/位置 | 类型 | 用途 | 行号 |
| --- | --- | --- | --- |
| `probe_full_gpu_irrigation_wait_a.py` | 新建 | 有界只读本任务栈/状态/等待点 | — |
| `archive_full_irrigation_wait_a.py` | 新建 | 原采样安全归档 | — |
| `profile_output/full-irrigation-wait-a/summary.json` | 新建 | 身份、时刻、源及状态元数据 | — |
| 同目录`stack.json`、`stderr.txt`、`progress-before.json` | 新建 | 单次栈、空stderr与原进度 | — |
| 同目录`probe_full_gpu_irrigation_wait_a.py` | 新建 | 采样原件副本 | — |
| `run_pytest_worker_a.py` | 新建 | 原pytest、同步、覆盖与退出验证 | — |
| `launch_gpu_worker_lifecycle_a.py` | 新建 | 两个原GPU文件的独立原型队列 | — |
| `check_pytest_worker_controls_a.py`、`pytest_worker_controls_a.pbs` | 新建 | EHPC隔离控制与提交 | — |
| `archive_pytest_worker_controls_a.py`、`archive_gpu_worker_prefix_a.py` | 新建 | 控制与首文件结果归档 | — |
| `gpu-worker-lifecycle-a/`、`pytest-worker-controls-a/` | 新建 | 日志/XML/节点清单/退出记录；逐项见各manifest | — |
| `run_most_full_gpu_d.py`、`launch_deferred_most_full_gpu_d.py` | 新建 | 修正仓库导入根，等待C终态后精确复用/补跑 | — |
| `snapshot_full_c_most_wait_a.py`、`full-c-most-wait-a/` | 新建 | 仅自有进程身份/瞬态、原进度/日志尾与冻结源哈希 | — |

未安装新的采样依赖，继续使用已有隔离py-spy0.4.1。没有GPU重置、系统配置
修改或其他用户进程操作，原年度与完整队列继续保留。
