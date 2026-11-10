# 实验二：搜索算法

丁禹丹　10243330407

实现 A*、Minimax 与 Alpha-Beta，并核对迷宫寻路、加权地形、博弈树和宝藏分配四个样例。报告中的路径、代价、扩展结点数和访问次数来自本目录的 `results/metrics.json`。

## 环境与依赖

- 操作系统：Windows 10 / 11
- Python：3.10 及以上（本次用 Anaconda 自带的 Python 运行）
- 依赖：仅标准库，不需要 `pip install`

`search.py` 使用 `heapq`。`experiment.py` 使用 `json`、`time`、`pathlib`。

## 目录

```text
lab02/
  README.md
  requirements.txt
  search.py            A*、Minimax、Alpha-Beta
  experiment.py        四道题的样例与对照实验
  results/
    metrics.json       运行后生成，含路径、代价、expanded、visited、cuts
```

`search.py` 提供算法，单独运行它不会做实验。`experiment.py` 调用其中的函数，把四题结果写入 `results/metrics.json`，并在终端打印同一份 JSON。

## 运行

在本目录执行：

```bash
python experiment.py
```

正常结束时，进程退出代码为 0。终端先打印完整 JSON，最后一行以 `CHECK` 开头。五项都应与样例一致：

- 题目一路径：`[(0,0), (0,1), (0,2), (1,2), (1,3), (1,4), (2,4), (3,4)]`，`q1_path_ok` 为 `true`
- 题目二代价：`7`，`q2_ok` 为 `true`
- 题目三根值：`5`，`q3_ok` 为 `true`
- 题目四 `H = 2`：`{"best_target": "T2", "best_value": -1}`，`q4_h2_ok` 为 `true`
- 题目四 `H = 4`：`{"best_target": "T3", "best_value": 3}`，`q4_h4_ok` 为 `true`

## 实验配置

- 移动：上下左右。走进一格才计 `enter_cost`，起点不计。
- 题目一：`heuristic_name` 取 `zero`、`euclidean`、`manhattan`；`tie` 取 `fifo`、`low_h`、`high_g`。样例使用 `manhattan` 与 `low_h`。另在 8×8、12×12 空地上重复这组对照。
- 题目二：终点进入代价为 0。样例使用 `heuristic_name = "adjusted"`，即 `h = max(0, d - 1)`，`tie = "low_h"`。另与 `zero`、`manhattan` 比较，并在 3×3 上枚举地形。
- 题目三：`order` 取 `left`、`right`、`perfect`、`worst`。随机树为分支 3 深度 4、分支 3 深度 5、分支 4 深度 4，种子都是 7。
- 题目四：距离用 `manhattan` 与 `low_h`。博弈的 `algo` 取 `minimax`、`ab`，`order` 取 `fixed`、`reverse`、`gain`，深度为 2 和 4。样例检查使用 `ab|fixed`。

## 结果

与报告对应的主要数字：

| 项目 | 结果 |
| --- | --- |
| 题目一样例 | 路径如上，`cost = 7`，`manhattan` + `low_h` 的 `expanded = 14` |
| 题目一空地 `low_h` | 8×8：`zero` 64，`euclidean` 51，`manhattan` 15；12×12：144、123、23 |
| 题目二样例 | `zero` / `adjusted` / `manhattan` 的 `expanded` 为 15、9、9，`cost` 都是 7 |
| 题目二 3×3 | 有路 1520 张，`wrong_cost = 0`，扩展数相同 1308 张，`manhattan` 更少 212 张 |
| 题目三样例 | 根值 5；`left` 的 Alpha-Beta 访问 14、剪枝 1；Minimax 访问 15 |
| 题目四距离 | `manhattan` 扩展 275，`zero` 扩展 680，`dist` 相同 |
| 题目四决策 | `H = 2` 为 T2、`-1`；`H = 4` 为 T3、`3` |
| 题目四访问 | `H = 2` 的 Minimax 为 17；Alpha-Beta 的 `fixed` / `reverse` / `gain` 为 13 / 16 / 11。`H = 4` 为 65，以及 47 / 46 / 37 |

`metrics.json` 里的 `sec` 是单次 `alphabeta` 或单次 `game_value` 的耗时，不是整个脚本的运行时间。每次运行会有波动，报告不使用这项。
