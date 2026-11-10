# -*- coding: utf-8 -*-
"""跑通题目一至四，并记录对照实验。结果写入 results/metrics.json。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from search import (
    TreeNode,
    alphabeta,
    astar,
    make_random_tree,
    manhattan,
    maze_astar,
    minimax,
    new_stats,
    sample_tree,
    weighted_astar,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MAZE = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
MAZE_START = (0, 0)
MAZE_GOAL = (3, 4)

TERRAIN = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
TERRAIN_START = (0, 0)
TERRAIN_GOAL = (4, 4)

# 题目四。A/B/T 格代价为 1，X 不可过，数字是进入该格的代价。
GAME_RAW = [
    ["A", "1", "1", "X", "2", "1", "T1"],
    ["1", "X", "6", "X", "1", "X", "1"],
    ["1", "1", "T2", "1", "1", "1", "1"],
    ["X", "1", "X", "8", "X", "1", "X"],
    ["1", "1", "1", "1", "T3", "1", "1"],
    ["1", "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]
LOCS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
VALUES = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
NAMES = ["A", "B", "T1", "T2", "T3", "T4"]


def game_weight(cell: str):
    if cell == "X":
        return None
    if cell in {"A", "B"} or cell.startswith("T"):
        return 1
    return int(cell)

#只用于题目四
#GAME_RAW前文已经给出例子： 是一个列表，里面又套了 7 个列表。每一小列表是地图的一行，格子里写的是字符串
def game_search(start, goal, heuristic_name: str):
    rows, cols = 7, 7

    def passable(p):
        return game_weight(GAME_RAW[p[0]][p[1]]) is not None

    def enter_cost(p):
        return game_weight(GAME_RAW[p[0]][p[1]])

    def heuristic(p):
        if heuristic_name == "zero":
            return 0
        return manhattan(p, goal)

    return astar(start, goal, passable, enter_cost, heuristic, rows, cols, tie="low_h")

#对这 6 个点两两调用 game_search，也就是 astar，得到最小代价。最后返回一个矩阵，矩阵的每个元素是两个点之间的最小代价。
# 例如 A 在 (0,0)，T2 在 (2,2)，代价是 4，记成 matrix["A"]["T2"] = 4。
def distance_matrix(heuristic_name: str):
    matrix = {}
    expanded = 0
    for a in NAMES:
        matrix[a] = {}
        for b in NAMES:
            if a == b:
                matrix[a][b] = 0
                continue
            result = game_search(LOCS[a], LOCS[b], heuristic_name)
            matrix[a][b] = result["cost"]
            expanded += result["expanded"]
    return matrix, expanded


def apply_move(pos_a, pos_b, remaining, score_a, score_b, player, target, dist):
    if player == "A":
        src = _name_of(pos_a)
        cost = dist[src][target]#dist 是距离矩阵，src 是起点名字，target 是终点名字。来自distance_matrix函数
        score_a += VALUES[target] - cost
        pos_a = LOCS[target]
    else:
        src = _name_of(pos_b)
        cost = dist[src][target]
        score_b += VALUES[target] - cost
        pos_b = LOCS[target]
    remaining = tuple(t for t in remaining if t != target)
    return pos_a, pos_b, remaining, score_a, score_b



#已经有坐标，想知道它叫什么。比如 pos 是 (2, 2)，循环到 p 也是 (2, 2) 时，返回 "T2"。
'''
LOCS.items() 是把字典里的每一对键和值拆出来。LOCS 本身是名字对应坐标：
LOCS = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}
循环 for name, p in LOCS.items() 每次拿到一对。第一次 name 是 "A"，p 是 (0, 0)；下一次 name 是 "B"，p 是 (6, 6)，一直到 T4。
'''
def _name_of(pos):
    for name, p in LOCS.items():
        if p == pos:
            return name
    raise KeyError(pos)




'''
game_value 返回两个东西：这一支最终的U，以及下一层选了哪个宝藏。
val 接住U
逗号后面的 _ 是占位符，表示第二个返回值这里不用。
'''
'''
algo 也是调用时传入的字符串，只有两种："minimax" 表示每个宝藏都试完；"ab" 表示做 Alpha-Beta 剪枝。可在main函数中Q4中看到
'''
def game_value(pos_a, pos_b, remaining, score_a, score_b, player, depth, dist, stats, algo, order, alpha, beta):
    stats["visited"] += 1
    if depth == 0 or not remaining:
        stats["leaves"] += 1
        return score_a - score_b, None
    moves = list(remaining)#moves 是列表，里面是这一回合还要尝试的宝藏名字。

    #gain：先拿眼前收益高的，MIN 先拿让自己收益高的（对 U 更不利）
    #reverse：按moves倒序，即T4321
    #fixed：按moves的顺序，即T1、T2、T3、T4
    #alpha, beta：alpha 是最大值，beta 是最小值。用于剪枝。
    if order == "gain":
        src = _name_of(pos_a if player == "A" else pos_b)
        gains = [(VALUES[t] - dist[src][t], t) for t in remaining]
        # MAX 先拿眼前收益高的，MIN 先拿让自己收益高的（对 U 更不利）
        gains.sort(reverse=True)
        moves = [t for _, t in gains]
    elif order == "reverse":
        moves = list(reversed(remaining))
    elif order == "fixed":
        moves = list(remaining)
    else:
        raise KeyError(order)

    maximizing = player == "A"
    best_move = None
    if maximizing:
        best = -10**9
        for i, target in enumerate(moves):
            nxt = apply_move(pos_a, pos_b, remaining, score_a, score_b, "A", target, dist)#nxt是 A 拿走 target 之后的新局面，一个有 5 项的元组：A 的位置、B 的位置、剩下的宝藏、A 的分数、B 的分数。
            val, _ = game_value(*nxt, "B", depth - 1, dist, stats, algo, order, alpha, beta)#*nxt 把那 5 项拆开，按顺序送进 game_value 的前 5 个参数，不用手写五个变量。
            if val > best:
                best, best_move = val, target#best=val, best_move=target的省略写法
            if algo == "ab" and best >= beta and i < len(moves) - 1:
                stats["cuts"] += 1
                stats["cut_remaining"] += len(moves) - i - 1
                break
            if algo == "ab":
                alpha = max(alpha, best)
        return best, best_move
    best = 10**9
    for i, target in enumerate(moves):
        nxt = apply_move(pos_a, pos_b, remaining, score_a, score_b, "B", target, dist)
        val, _ = game_value(*nxt, "A", depth - 1, dist, stats, algo, order, alpha, beta)
        if val < best:
            best, best_move = val, target
        if algo == "ab" and best <= alpha and i < len(moves) - 1:
            stats["cuts"] += 1
            stats["cut_remaining"] += len(moves) - i - 1
            break
        if algo == "ab":
            beta = min(beta, best)
    return best, best_move
'''
所以同一次搜索里，U先后叫这三个名字：叶子上的 score_a - score_b，上一层的 val，这一层留下的 best。它始终是一个整数。A 先拿 T2、双方走完时，这个数是 1，不是一串宝藏名字。

运行示例：
A 已经拿了 T2 之后，剩下三轮会走出 6 条完整对局，每条最后都有一个 U
B 拿 T1，然后 A 拿 T3、B 拿 T4    U = 7
B 拿 T1，然后 A 拿 T4、B 拿 T3    U = 1

B 拿 T3，然后 A 拿 T1、B 拿 T4    U = 1
B 拿 T3，然后 A 拿 T4、B 拿 T1    U = -3

B 拿 T4，然后 A 拿 T1、B 拿 T3    U = 7
B 拿 T4，然后 A 拿 T3、B 拿 T1    U = 9

B 不能在这 6 个数里直接挑最小的-3。因为中间还有一次轮到 A，A 会挑U更大的那条。所以 B 的每个选择只能拿到 A 挑完以后的结果：

B 拿 T1：A 会挑 U = 7，不会挑 1
B 拿 T3：A 会挑 U = 1，不会挑 -3
B 拿 T4：A 会挑 U = 7，不会挑 9
B 再在 7、1、7 里挑最小的，也就是 1。对应的行动是 B 拿 T3。−3 那条对局虽然存在，但 A 不会配合走到那里。
'''







def first_move_table(dist, depth):
    rows = []
    remaining0 = ("T1", "T2", "T3", "T4")
    for target in remaining0:
        pos_a, pos_b, remaining, score_a, score_b = apply_move(
            LOCS["A"], LOCS["B"], remaining0, 0, 0, "A", target, dist
        )
        stats = new_stats()
        # B 之后若 depth>1 继续完整搜索
        val, b_move = game_value(
            pos_a, pos_b, remaining, score_a, score_b, "B", depth - 1, dist, stats, "minimax", "fixed", -10**9, 10**9
        )
        rows.append(
            {
                "target": target,
                "gain_a": score_a,
                "d": dist["A"][target],
                "best_value": val,
                "b_reply": b_move,
            }
        )
    return rows


def empty_grid(n):
    return [["." for _ in range(n)] for _ in range(n)]



#题目二的自设对照：把 3×3 地图的中间 7 格换成各种代价，比较修正启发h'和原始曼哈顿距离。它不参与四道题的样例求解。
#该函数运行后得到结论：2187 张都看完后返回统计。实际结果是：有路 1520 张，扩展数相同 1308 张，曼哈顿更少 212 张，代价不同 0 张。
def enumerate_weighted():
    """3×3，S 在左上、E 在右下，其余格取 1、3 或 X。只统计有路的图。"""
    from itertools import product

    #cells 是除起点 (0,0) 和终点 (2,2) 以外的 7 个坐标。这两格后面固定写成 S 和 E，不参与替换。
    cells = [(r, c) for r in range(3) for c in range(3) if (r, c) not in {(0, 0), (2, 2)}]
    reachable = same = fewer = wrong = 0#四个计数器
    example = None

    #assignment 是长度为 7 的一组值，例如 (1, 3, "X", 1, 1, 1, 3)。
    #product 是 product(iterable1, iterable2, ...)，返回一个生成器，生成所有iterable1和iterable2的笛卡尔积。
    for assignment in product((1, 3, "X"), repeat=len(cells)):
        grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        grid[0][0] = "S"
        grid[2][2] = "E"
        for p, w in zip(cells, assignment):
            grid[p[0]][p[1]] = w
        base = weighted_astar(grid, (0, 0), (2, 2), "adjusted", "low_h")
        if base["cost"] < 0:
            continue
        raw = weighted_astar(grid, (0, 0), (2, 2), "manhattan", "low_h")
        zero = weighted_astar(grid, (0, 0), (2, 2), "zero", "low_h")
        reachable += 1
        if raw["cost"] != base["cost"]:
            wrong += 1
        elif raw["expanded"] < base["expanded"]:
            fewer += 1
            if example is None:
                example = {
                    "grid": grid,
                    "cost": base["cost"],
                    "expanded_adjusted": base["expanded"],
                    "expanded_manhattan": raw["expanded"],
                    "expanded_zero": zero["expanded"],
                }
        elif raw["expanded"] == base["expanded"]:
            same += 1
    return {
        "reachable": reachable,
        "same_nodes": same,
        "manhattan_fewer": fewer,
        "wrong_cost": wrong,
        "example": example,
    }



'''
1.enumerate 是给一串东西编号。for i, p in enumerate(path) 里，i 从 0 开始，是第几个；p 是这一项本身。路径 [(0,0), (1,0), (2,0)] 会依次得到：
i = 0, p = (0,0)
i = 1, p = (1,0)
i = 2, p = (2,0)
'''
def path_steps(grid_result, heuristic_fn):
    path = grid_result["path"]
    gmap = grid_result["g"]
    rows = []
    for i, p in enumerate(path):
        g = gmap[p]#走到当前格 p 的总代价
        h = heuristic_fn(p)
        prev = path[i - 1] if i else None
        step = None if prev is None else g - gmap[prev]
        rows.append({"pos": list(p), "enter": step, "g": g, "h": h, "f": g + h})
    return rows


def main():
    report = {}

    # 题目一
    q1 = {}
    for name in ("zero", "euclidean", "manhattan"):
        for tie in ("fifo", "low_h", "high_g"):
            result = maze_astar(MAZE, MAZE_START, MAZE_GOAL, name, tie)
            q1[f"{name}|{tie}"] = {
                "cost": result["cost"],
                "expanded": result["expanded"],
                "path": result["path"],
            }
    best = maze_astar(MAZE, MAZE_START, MAZE_GOAL, "manhattan", "low_h")
    q1["official_path"] = best["path"]
    q1["official_expanded"] = best["expanded"]
    q1["steps"] = path_steps(best, lambda p: manhattan(p, MAZE_GOAL))

    empty = {}
    for n in (8, 12):
        grid = empty_grid(n)
        empty[str(n)] = {}
        for name in ("zero", "euclidean", "manhattan"):
            empty[str(n)][name] = {}
            for tie in ("fifo", "low_h", "high_g"):
                result = maze_astar(grid, (0, 0), (n - 1, n - 1), name, tie)
                empty[str(n)][name][tie] = {"cost": result["cost"], "expanded": result["expanded"]}
    q1["empty"] = empty
    report["q1"] = q1

    # 题目二
    q2 = {}
    for name in ("zero", "adjusted", "manhattan"):
        result = weighted_astar(TERRAIN, TERRAIN_START, TERRAIN_GOAL, name, "low_h", goal_cost=0)
        q2[name] = {
            "cost": result["cost"],
            "expanded": result["expanded"],
            "path": result["path"],
        }
    adj = weighted_astar(TERRAIN, TERRAIN_START, TERRAIN_GOAL, "adjusted", "low_h")
    q2["steps"] = path_steps(adj, lambda p: max(0, manhattan(p, TERRAIN_GOAL) - 1))
    q2["steps_manhattan_h"] = path_steps(
        weighted_astar(TERRAIN, TERRAIN_START, TERRAIN_GOAL, "manhattan", "low_h"),
        lambda p: manhattan(p, TERRAIN_GOAL),
    )
    q2["enumeration_3x3"] = enumerate_weighted()
    report["q2"] = q2

    # 题目三
    q3 = {}
    root = sample_tree()
    for order in ("left", "right", "perfect", "worst"):
        st_m = new_stats()
        st_a = new_stats()
        # 每轮重建树，避免没有必要，树无副作用
        tree = sample_tree()
        vm = minimax(tree, True, st_m, order)
        tree = sample_tree()
        va = alphabeta(tree, True, -10**9, 10**9, st_a, order)
        q3[order] = {"minimax": vm, "minimax_stats": st_m, "alphabeta": va, "alphabeta_stats": st_a}
    wider = {}
    for branch, depth, seed in ((3, 4, 7), (3, 5, 7), (4, 4, 7)):
        key = f"b{branch}_d{depth}"
        wider[key] = {}
        for order in ("left", "right", "perfect", "worst"):
            tree = make_random_tree(branch, depth, seed)
            st = new_stats()
            t0 = time.perf_counter()
            val = alphabeta(tree, True, -10**9, 10**9, st, order)
            elapsed = time.perf_counter() - t0
            tree = make_random_tree(branch, depth, seed)
            st_m = new_stats()
            base = minimax(tree, True, st_m, "left")
            wider[key][order] = {
                "value": val,
                "matches_minimax": val == base,
                "visited": st["visited"],
                "leaves": st["leaves"],
                "cuts": st["cuts"],
                "minimax_visited": st_m["visited"],
                "sec": round(elapsed, 6),
            }
    q3["wider"] = wider
    report["q3"] = q3

    # 题目四
    dist_h, exp_h = distance_matrix("manhattan")
    dist_0, exp_0 = distance_matrix("zero")
    assert dist_h == dist_0
    q4 = {
        "distance": dist_h,
        "expanded_manhattan": exp_h,
        "expanded_zero": exp_0,
    }
    searches = {}
    for depth in (2, 4):
        searches[str(depth)] = {}
        for algo in ("minimax", "ab"):
            for order in ("fixed", "reverse", "gain"):
                stats = new_stats()
                t0 = time.perf_counter()
                val, move = game_value(
                    LOCS["A"],
                    LOCS["B"],
                    ("T1", "T2", "T3", "T4"),
                    0,
                    0,
                    "A",
                    depth,
                    dist_h,
                    stats,
                    algo,
                    order,
                    -10**9,
                    10**9,
                )
                searches[str(depth)][f"{algo}|{order}"] = {
                    "best_target": move,
                    "best_value": val,
                    "visited": stats["visited"],
                    "leaves": stats["leaves"],
                    "cuts": stats["cuts"],
                    "sec": round(time.perf_counter() - t0, 6),
                }
    q4["search"] = searches
    q4["h2_first_moves"] = first_move_table(dist_h, 2)
    q4["h4_first_moves"] = first_move_table(dist_h, 4)
    report["q4"] = q4

    # 检查官方期望
    official = {
        "q1_path": [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)],
        "q2_cost": 7,
        "q3_value": 5,
        "q4_h2": {"best_target": "T2", "best_value": -1},
        "q4_h4": {"best_target": "T3", "best_value": 3},
    }
    got = {
        "q1_path": [list(p) for p in best["path"]],
        "q2_cost": q2["adjusted"]["cost"],
        "q3_value": q3["left"]["alphabeta"],
        "q4_h2": {k: searches["2"]["ab|fixed"][k] for k in ("best_target", "best_value")},
        "q4_h4": {k: searches["4"]["ab|fixed"][k] for k in ("best_target", "best_value")},
    }
    report["check"] = {
        "official": official,
        "got": got,
        "q1_path_ok": got["q1_path"] == [list(p) for p in official["q1_path"]],
        "q2_ok": got["q2_cost"] == 7,
        "q3_ok": got["q3_value"] == 5,
        "q4_h2_ok": got["q4_h2"] == official["q4_h2"],
        "q4_h4_ok": got["q4_h4"] == official["q4_h4"],
    }

    def convert(obj):
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    text = json.dumps(convert(report), ensure_ascii=False, indent=2)
    (OUT / "metrics.json").write_text(text, encoding="utf-8")
    print(text)
    print("CHECK", json.dumps(report["check"], ensure_ascii=False))


if __name__ == "__main__":
    main()
