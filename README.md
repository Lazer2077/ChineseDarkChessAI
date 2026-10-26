# Chinese Dark Chess AI

面向**揭棋（全盘 Jieqi）**的算法科研项目，GPL-3.0。不是半盘 Banqi。
目标：研究信念状态搜索与自我博弈学习，在严格配对评测中超过固定的参考引擎。
**目前是可运行的研究起点，没有超过参考引擎的证据。** 不包含网页前端。

## 当前实现
- 复用固定版本的参考规则与 JS 搜索，来源见 docs/PROVENANCE.md。
- 真实模拟器与玩家 observation 分离。暗子被吃仅吃子方知道身份；未知离盘子保留在信念库存。
- 无放回完整隐藏世界采样、根节点 PUCT 分配、按行动方观测执行的 rollout。
- 搜索访问分布蒸馏到 15 维线性策略，终局价值回归为诊断头。
- JSON 对局记录、固定种子、布局换色、截断单列、可选参考引擎对手。

这是 root belief Monte Carlo 基线，**不是完整 POMCP、ISMCTS、ReBeL 或 AlphaZero**。
Rollout 在单一根观察者采样世界中推演，仍可能有策略融合和对手信念失配。
价值头尚未用于搜索。线性策略用于打通训练闭环，后续替换为历史/库存编码网络。

## 运行（Node.js 22+；训练另外需要 Python + PyTorch）
```sh
npm test
node src/run.js --games=4 --seed=1 --simulations=64 --horizon=8 --out=runs/gen0
python3 scripts/train.py --data=runs/gen0 --out=runs/model0.json --epochs=20 --device=cpu
node src/run.js --games=4 --seed=1001 --model=runs/model0.json --out=runs/gen1
node src/run.js --games=20 --seed=100001 --model=runs/model0.json --opponent=upstream --timeMs=1000 --out=runs/eval0
```
预算不足时搜索会保证每个根动作至少访问一次，因此 simulations 是下限。
训练仅用搜索产生的行；截断局不提供终局价值标签，仍可用于策略蒸馏。
输出目录不能覆盖。上例为功能流程，不构成等时间公平棋力实验。

## 规则版本
采用固定参考实现：红先；暗子第一步按原位角色、随后揭开；明士象可过河；马腿/象眼、炮架、九宫和飞将；不能送将；将死/困毙判负。
暗子移动合法性按移动前的原位角色判定，与参考代码一致；揭开后将军边界需在规则研究中独立核查。
第三次公开局面重复由造成重复方判负；120 半回合无吃子/揭子或400半回合判和。
此为研究规则配置，未实现赛事长将长捉裁判。更换规则必须分开报告结果。

## 科研路线
1. 独立规则 oracle、战术题集、信息泄漏与库存性质测试；固定当前基准。
2. 历史/私人暗俘/公共库存编码；残差策略价值网络；mask 合法动作。
3. 信息集树和对手信念建模，对比 root sampling、POMCP 固定对手、re-determinizing ISMCTS。
4. 公共信念表示与自我博弈 population；考虑 ReBeL/CFR 子博弈，避免把 POMDP 最优性当作均衡保证。
5. 跨种子、跨预算、换色评测和消融：去信念、去私人库存、采样数量、信息价值、风险敏感。

## 验收标准
训练与测试布局种子分离。至少100对布局换色，重复至少3个训练种子。
等墙钟时间与等模拟数分别报告；记录硬件、版本、模型哈希、耗时、超时/非法着法与截断。
报告 W/D/L、得分率和按布局配对 bootstrap 的95%区间；区间下界超过50%才支持对应条件下优于基准。
参考 JS 与 C++ 分别列出，不能把超过 JS 表述为超过整个上游项目。
不得使用真实暗子标签作为 actor 输入；任何 privileged critic 必须单列消融。

## CS01
CS01 已由用户确认为本机 csci-cscidl01。首轮实验使用独立目录和有限预算。
运行 `scripts/cs01-pilot.sh` 可复跑闭环；通过 TRAIN_PYTHON 指定已有 PyTorch 环境，TRAIN_DEVICE 指定 cpu/cuda。
自我博弈按访问分布采样探索；对外评测选择访问数最多的动作。
