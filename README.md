# RAPPO 复现：第 1 天规范冻结与 Skyfield 数据层

本文件包用于完成论文《LLM-assisted Deep Reinforcement Learning for Satellite Range Scheduling》的第一天任务：

1. 冻结论文复现规范；
2. 区分“论文明确给出”“论文未给出”“标准物理常数”“复现实验补充假设”；
3. 使用 Skyfield 替代 STK 生成 24 小时卫星位置和 SGL/ISL/IDL 传输窗口；
4. 固化输入、输出、校验、日志、哈希和实验清单；
5. 为后续调度环境和 MAPPO 提供不可变的数据接口。

## 文件说明

- `PROMPT_DAY1_SKYFIELD_IMPLEMENTATION.md`
  - 可直接交给 Codex 或其他代码代理的完整实现指令。
- `configs/paper_reported.yaml`
  - 论文明确公开的全部模型、算法和实验参数；论文未给出的参数保留为 `null`。
- `configs/reproduction_choices.template.yaml`
  - 必须由复现者冻结的补充参数模板。
- `configs/reproduction_choices.recommended.yaml`
  - 一组可执行的推荐值；所有非论文参数均明确标注。
- `configs/constellation.template.yaml`
  - 15 颗卫星的轨道与域分配模板。
- `configs/ground_stations.template.yaml`
  - 4 个地面站模板。
- `configs/output_schema.yaml`
  - 位置矩阵、可用性矩阵、窗口表和报告的输出契约。
- `configs/frozen_manifest.template.yaml`
  - 科研实验清单，包括 Git 提交、依赖版本、输入文件哈希、配置哈希和运行环境。
- `docs/PARAMETER_REGISTER.md`
  - 参数总表和来源分类。
- `docs/AMBIGUITIES_AND_DECISIONS.md`
  - 论文歧义以及必须明确记录的复现决策。
- `docs/DAY1_ACCEPTANCE_CHECKLIST.md`
  - 第一天完成验收清单。
- `environment-day1.yml`
  - Conda 环境定义；本次验收使用现有的 `satellite` 环境（Python 3.9.25），按任务要求不以 Python 3.8 为阻塞条件。
- `requirements-day1.txt`
  - pip 依赖版本。

## 参数优先级

运行代码时必须遵循以下优先级：

1. `paper_reported.yaml`：论文明确设置，不得静默覆盖；
2. `reproduction_choices.yaml`：论文未公开部分，由复现者冻结；
3. `constellation.yaml` 与 `ground_stations.yaml`：具体场景输入；
4. 命令行参数：默认不得覆盖已冻结参数，除非使用显式 `--allow-override`，并写入 manifest。

## 主实验建议

论文主实验采用固定链路速率：

- ISL/IDL：80 Mbps；
- SGL：60 Mbps；
- SGL 带宽：80 MHz；
- 调度周期：24 h。

因此 Skyfield 数据层应主要生成几何可见性和传输窗口，主实验不要根据距离动态修改论文给出的固定速率。理论公式模式可另行实现，但不得与主实验结果混用。

## 当前冻结场景

仓库已包含一组可直接运行的 Day 1 冻结配置：

- `configs/reproduction_choices.yaml`
- `configs/constellation.yaml`
- `configs/ground_stations.yaml`

它们使用代表性的 GEO 通信层、705 km/98.2°观测层、约 21,500 km/55°导航层，以及 4 个公开 IGS 中国站点。论文没有公开具体卫星、轨道根数或站点身份，因此这些值全部标记为 `reproduction_choice`，不能称为论文原始设置。完整来源和差异见 `docs/DAY1_IMPLEMENTATION_REPORT.md`。

## 安装与运行

Windows PowerShell：

```powershell
conda run -n satellite python -m pip install --no-deps -e .
conda run -n satellite python -m pytest -q
conda run -n satellite python -m orbit_data.cli validate-config
conda run -n satellite python -m orbit_data.cli generate --profile development
conda run -n satellite python -m orbit_data.cli validate --data-root data/skyfield
```

`development` 是默认配置：未取得 `configs/constellation.yaml` 中研究者签署时，清单的 `data_status` 必为 `DEVELOPMENT_ONLY`。只有填写批准人、批准时间和依据并设为批准后，`--profile official` 才会生成可用于主实验的冻结数据。

新增的可用性契约为 UTC 的 `[start, end)` 区间、30 秒几何网格和 `duration_seconds × rate_Mbps` 容量计算。ISL/IDL NPZ 同时包含时间戳、卫星 ID、域 ID 和域名；天线换向采用 E/N/U 三维单位向量夹角，而非方位角/仰角直接相减。

正式输出已生成在 `data/skyfield/`。输出目录非空时默认拒绝覆盖；确需重建时必须显式使用：

```powershell
conda run -n satellite python -m orbit_data.cli generate --profile development --allow-override --overwrite
```

## 自定义场景前必须填写

以下信息论文没有公开，不能由代码自行猜测：

- 仿真起始 UTC；
- 15 颗卫星的轨道参数；
- 三个业务域各自的卫星数量与成员；
- 4 个地面站经纬度和高度；
- 最低星地仰角；
- 通信域和导航域的域内邻接规则；
- IDL 的对称化规则；
- 地球遮挡模型；
- 时间采样与窗口边界精化精度。

如需替换当前代表性场景，先从模板复制：

```bash
cp configs/reproduction_choices.template.yaml configs/reproduction_choices.yaml
cp configs/constellation.template.yaml configs/constellation.yaml
cp configs/ground_stations.template.yaml configs/ground_stations.yaml
```

然后冻结配置并生成哈希。

## 统一基线训练

当前基线实现集中在 `src/baselines/`，统一配置为
`configs/baselines.yaml`。三种可复现实验方法分别为：

- `manual_mappo`：人工奖励 MAPPO；
- `ppo_lya`：人工基础奖励加 Lyapunov 塑形；
- `llm_ppo`：读取已冻结奖励权重的 LLM-PPO，正式训练期间不调用 API。

先注册一个已经生成并人工确认的奖励规范：

```cmd
conda run -n satellite python scripts\register_llm_reward_spec.py ^
  --source "已有selected_reward_spec.json的路径"
```

训练单个方法：

```cmd
conda run -n satellite python scripts\train_baseline.py ^
  --method manual_mappo ^
  --episodes 5
```

按给定顺序训练多个方法：

```cmd
conda run -n satellite python scripts\train_baselines.py ^
  --methods manual_mappo ppo_lya llm_ppo ^
  --episodes 5 ^
  --reward-spec results\baselines\reward_specs\selected_reward_spec.json
```

训练全部方法时，固定顺序为 `manual_mappo`、`ppo_lya`、`llm_ppo`：

```cmd
conda run -n satellite python scripts\train_baselines.py ^
  --methods all ^
  --episodes 5 ^
  --reward-spec results\baselines\reward_specs\selected_reward_spec.json
```

恢复某个总体 Run 时，`--episodes`仍表示每种方法的目标总 Episode 数：

```cmd
conda run -n satellite python scripts\train_baselines.py ^
  --resume-run results\baselines\runs\<run_id> ^
  --episodes 10
```

新结果只写入 `results/baselines/`。每次训练使用独立 Run 目录，包含
`run_manifest.json`、`run_summary.json`、`comparison.json`，以及各方法独立的
日志和 Checkpoint。已存在的 Run 目录默认拒绝覆盖。

模型选择使用运行时隔离协议：`reward_search` 与 `checkpoint_selection` 由
150 个 validation 任务按 seed 2027 确定性分成两个互斥的 75 任务池，最终
`test` 协议只读取完整 test 划分。上述池大小、seed 和任务数均为复现补充设置，
不是论文公开参数。比较采用送达及时性、完成率、过期率、送达数据量、负载均衡、
拒绝动作率的容差字典序；旧 `timeliness_raw` 仍保留全部成功链路的历史含义，
新增 `delivered_timeliness_raw` 只统计成功 SGL 下传。

LLM-PPO 默认只读取冻结奖励规范；规范缺失时立即失败，不会自动调用 API。
只有显式使用 `--prepare-llm-reward` 或 `--refresh-llm-reward` 并选择
`--llm-provider deepseek` 时才可能调用真实 API。程序会完整显示预算，且只接受
交互终端输入精确的 `YES`；Mock Provider 无需确认。仓库不提供静默绕过确认的参数。

## 第五天：RAPPO知识库

知识库以 `knowledge_sources/RAPPO_第五天_知识库文献清单.xlsx` 的“文献清单”工作表为唯一元数据来源。先执行 `python scripts\organize_knowledge_sources.py` 生成 Dry Run 报告，确认全部实际 PDF 唯一匹配后才可加 `--apply`。原始 PDF 会保留，规范副本不会覆盖既有文件。

RAPPO 与 LLM-PPO 的唯一区别是：RAPPO 在每次奖励候选生成前使用本地 CodeBERT 向量索引检索 Top-5 引文；LLM-PPO 始终无 RAG。正式 5×8 搜索尚未执行。PDF 的技术提取、第二次人工审核、索引构建和 Mock RAPPO 均必须在清单完整且匹配唯一后进行。

确定性验证单个 Checkpoint 或刷新整个 Run 的比较结果：

```cmd
conda run -n satellite python scripts\evaluate_baseline.py ^
  --method ppo_lya ^
  --checkpoint results\baselines\runs\<run_id>\ppo_lya\best_checkpoint.pt

conda run -n satellite python scripts\evaluate_baseline.py ^
  --run-dir results\baselines\runs\<run_id>
```

默认使用 `checkpoint_selection` 并更新普通 `evaluation.json` / `comparison.json`。
最终测试必须显式加入 `--protocol test`，结果写入独立的
`test_evaluation.json` / `test_comparison.json`，不会覆盖普通比较文件。

LLM 权重规范原文和 `spec_id` 保持不变；训练前仅做 L1 尺度归一化，使八项
有效权重之和与人工奖励一致。日志、摘要和 Checkpoint 会记录原始/有效权重、
归一化因子及有效权重哈希。PPO-Lya 的四项势函数权重
`0.45/0.35/0.10/0.10` 以及已过期未送达债务同样属于复现补充设置。
