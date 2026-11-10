# -*- coding: utf-8 -*-
"""实验二：A*、Minimax 与 Alpha-Beta。"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Pos = Tuple[int, int]
Grid = Sequence[Sequence[object]]

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(
    start: Pos,
    goal: Pos,
    passable: Callable[[Pos], bool],
    enter_cost: Callable[[Pos], int],
    heuristic: Callable[[Pos], int],
    rows: int,
    cols: int,
    tie: str = "fifo",
) -> dict:
    """图搜索 A*。目标出队后停止。tie: fifo | low_h | high_g。"""
    if not passable(start) or not passable(goal):
        return {"path": [], "cost": -1, "expanded": 0, "generated": 0}

    tie_id = {"fifo": 0, "low_h": 1, "high_g": 2}[tie]
    counter = 0
    g_score = {start: 0}
    parent: Dict[Pos, Optional[Pos]] = {start: None}
    h0 = heuristic(start)

    def priority(g: int, h: int, seq: int):
        f = g + h
        if tie_id == 1:
            return (f, h, seq)
        if tie_id == 2:
            return (f, -g, seq)
        return (f, seq)

    heap = [(priority(0, h0, counter), start)]
    closed = set()
    expanded = 0
    generated = 1
    order: List[Pos] = []

    while heap:
        _, current = heapq.heappop(heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1
        order.append(current)
        if current == goal:
            path = []
            node: Optional[Pos] = current
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return {
                "path": path,
                "cost": g_score[current],
                "expanded": expanded,
                "generated": generated,
                "order": order,
                "g": dict(g_score),
            }
        r, c = current#当前格子的行和列
        #dr, dc 是这一步行、列要加多少。
        #nxt 是加完得到的邻居，是一个有两项的元组。nxt[0]表示行，nxt[1]表示列
        #DIRS 是上、下、左、右，对应 (-1, 0)、(1, 0)、(0, -1)、(0, 1)。
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if not passable(nxt) or nxt in closed:
                continue
            ng = g_score[current] + enter_cost(nxt)
            if ng < g_score.get(nxt, 10**9):
                g_score[nxt] = ng
                parent[nxt] = current
                generated += 1
                counter += 1
                h = heuristic(nxt)
                heapq.heappush(heap, (priority(ng, h, counter), nxt))

    return {"path": [], "cost": -1, "expanded": expanded, "generated": generated, "order": order}


def cell_kind(cell) -> str:
    if isinstance(cell, str) and cell in {"S", "E", "A", "B"}:
        return cell
    if isinstance(cell, str) and cell.startswith("T"):
        return cell
    return "num"


def maze_astar(grid: Grid, start: Pos, goal: Pos, heuristic_name: str = "manhattan", tie: str = "fifo") -> dict:
    rows, cols = len(grid), len(grid[0])

    def passable(p: Pos) -> bool:
        v = grid[p[0]][p[1]]#Pos是（行，列）。p[0]是行，p[1]是列
        return v != "X"

    def enter_cost(_: Pos) -> int:
        return 1

    def heuristic(p: Pos) -> int:
        if heuristic_name == "zero":
            return 0
        if heuristic_name == "euclidean":
            return int((abs(p[0] - goal[0]) ** 2 + abs(p[1] - goal[1]) ** 2) ** 0.5)
        if heuristic_name == "manhattan":
            return manhattan(p, goal)
        raise KeyError(heuristic_name)

    return astar(start, goal, passable, enter_cost, heuristic, rows, cols, tie)


def weighted_astar(
    grid: Grid,
    start: Pos,
    goal: Pos,
    heuristic_name: str = "adjusted",
    tie: str = "fifo",
    goal_cost: int = 0,
) -> dict:
    """进入格子支付地形代价。终点默认代价 0，与题目二样例一致。"""
    rows, cols = len(grid), len(grid[0])

    def weight(p: Pos) -> Optional[int]:
        v = grid[p[0]][p[1]]
        if v == "X":
            return None
        if v in {"S", "E"} or (isinstance(v, str) and str(v).startswith("T")) or v in {"A", "B"}:
            return 0 if (p == goal and v in {"E", "S"}) else 1
        if p == goal:
            return goal_cost
        return int(v)

    def passable(p: Pos) -> bool:
        return weight(p) is not None

    def enter_cost(p: Pos) -> int:
        w = weight(p)
        assert w is not None
        return w


    #adjusted是题目二中用的修正启发 h'=max(0,d-1).d是当前格到终点的曼哈顿距离

    def heuristic(p: Pos) -> int:
        d = manhattan(p, goal)
        if heuristic_name == "zero":
            return 0
        if heuristic_name == "manhattan":
            return d
        if heuristic_name == "adjusted":
            # 最后一步进入终点代价为 0，其余格子代价至少为 1
            return max(0, d - 1)
        raise KeyError(heuristic_name)#三个名字都对不上时，程序在这里主动停下来，并带上那个不认识的名字。例如输出：KeyError(mahattannnn)

    return astar(start, goal, passable, enter_cost, heuristic, rows, cols, tie)


class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else []

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


def minimax(node: TreeNode, maximizing: bool, stats: dict, order: str = "left") -> int:
    stats["visited"] += 1
    if node.is_leaf:
        stats["leaves"] += 1
        return node.value
    kids = _ordered(node.children, order, maximizing, stats)
    #maximizing是True，说明是max节点，否则是min节点
    if maximizing:
        best = -10**9
        for child in kids:
            best = max(best, minimax(child, False, stats, order))
        return best
    #maximizing是False，说明是min节点
    best = 10**9
    for child in kids:
        best = min(best, minimax(child, True, stats, order))
    return best

#stats = {"visited": 15, "leaves": 8, "cuts": 0, "cut_remaining": 0}
#visited: 访问的节点数
#leaves: 叶子节点数
#cuts: 剪枝数
#cut_remaining: 剩余剪枝数

def alphabeta(
    node: TreeNode,
    maximizing: bool,
    alpha: int,
    beta: int,
    stats: dict,
    order: str = "left",
) -> int:
    stats["visited"] += 1
    if node.is_leaf:
        stats["leaves"] += 1
        return node.value
    kids = _ordered(node.children, order, maximizing, stats)
    if maximizing:
        v = -10**9
        for i, child in enumerate(kids):
            v = max(v, alphabeta(child, False, alpha, beta, stats, order))
            if v >= beta and i < len(kids) - 1:
                stats["cuts"] += 1
                stats["cut_remaining"] += len(kids) - i - 1
                break
            alpha = max(alpha, v)
        return v
    v = 10**9
    for i, child in enumerate(kids):
        v = min(v, alphabeta(child, True, alpha, beta, stats, order))
        if v <= alpha and i < len(kids) - 1:
            stats["cuts"] += 1
            stats["cut_remaining"] += len(kids) - i - 1
            break
        beta = min(beta, v)
    return v



# experiment里对同一棵样例树连跑四次，每次只改这个字符串："left" / "right" / "perfect" / "worst"
#这次比的是搜索做了多少工作，也就是快慢，不是占用多少空间。
#stats["visited"] 记的是 alphabeta 被调用了几次。少调用一次，就少看一个结点，计算量就少一点。
def _ordered(children: List[TreeNode], order: str, maximizing: bool, stats: dict) -> List[TreeNode]:
    if order == "left":
        return list(children)
    if order == "right":
        return list(reversed(children))
    # 子结点与当前层玩家相反。perfect / worst 用子树极小极大值排序，只作为排序上界。
    child_is_max = not maximizing
    if order == "perfect":
        return sorted(children, key=lambda n: _minimax_value(n, child_is_max), reverse=maximizing)
    if order == "worst":
        return sorted(children, key=lambda n: _minimax_value(n, child_is_max), reverse=not maximizing)
    raise KeyError(order)


def _minimax_value(node: TreeNode, maximizing: bool) -> int:
    if node.is_leaf:
        return node.value
    vals = [_minimax_value(c, not maximizing) for c in node.children]
    return max(vals) if maximizing else min(vals)


def new_stats() -> dict:
    return {"visited": 0, "leaves": 0, "cuts": 0, "cut_remaining": 0}
'''
每跑一次搜索就要一个新的，因为同一次搜索里所有递归调用必须共用同一个字典，下一次搜索却不能接着上次的数加。
题目三的四次循环就是这样
for order in ("left", "right", "perfect", "worst"):
    st_m = new_stats()
    vm = minimax(tree, True, st_m, order)
'''

def sample_tree() -> TreeNode:
    """题目三 JSON 输入样例。右子树是独立的 (6,9) 与 (1,2)，不是共享结点。"""
    max1 = TreeNode(children=[TreeNode(3), TreeNode(5)])
    max2 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max3 = TreeNode(children=[TreeNode(6), TreeNode(9)])
    max4 = TreeNode(children=[TreeNode(1), TreeNode(2)])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max3, max4])
    return TreeNode(children=[min1, min2])

#这个函数是另外造一棵更大的树，用来看顺序对访问数的影响。具体用法见experiment文件
def make_random_tree(branch: int, depth: int, seed: int) -> TreeNode:
    import random

    rng = random.Random(seed)

    def build(d: int) -> TreeNode:
        if d == 0:
            return TreeNode(value=rng.randint(0, 99))
        return TreeNode(children=[build(d - 1) for _ in range(branch)])

    return build(depth)
