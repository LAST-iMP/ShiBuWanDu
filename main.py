import time

import numpy as np
import taichi as ti

H = 6
W = 5
TOTAL = H*W
MOVE = 15
SET_SIZE = 1000000

ti.init(arch=ti.cuda, debug=False)
# ti.init(arch=ti.cpu, debug=False)
res = ti.types.struct(idx=ti.uint64, score=ti.uint32, index=ti.types.vector(TOTAL, ti.uint32))

max_score = ti.field(ti.uint32, shape=())
factorio = ti.field(ti.uint64, shape=(MOVE + 1, TOTAL + MOVE))
res_set = res.field(shape=(SET_SIZE,))
board_set = ti.field(ti.uint32, shape=(W, H, SET_SIZE))
mask = ti.field(ti.uint32, shape=(W, H))


def init():
    factorio.fill(1)
    for i in range(MOVE + 1):
        for j in range(TOTAL + MOVE):
            if (i <= 1 and j <= 1) or i > j or i == 0: continue
            if i * 2 <= j:
                factorio[i, j] = factorio[i - 1, j - 1] + factorio[i, j - 1]
            else:
                factorio[i, j] = factorio[j - i, j]
    print("total possibility:", factorio[MOVE, TOTAL+MOVE-1])
    # 添加掩码
    mask[0, 0] = 1
    mask[W-1, H-1] = 1


@ti.func
def play(i: ti.uint64, opt: ti.uint32) -> ti.uint32:
    finish = False
    move_num = 0
    set_num = i % SET_SIZE
    next_opt = opt
    while not finish:
        move_num += 1
        y = ti.int32(next_opt / W)
        x = next_opt - y * W
        board_set[x, y, set_num] = (board_set[x, y, set_num] + 1) & 3
        if board_set[x, y, set_num] == 0:
            if y == 0: finish = True
            y -= 1
        elif board_set[x, y, set_num] == 1:
            if x == W - 1: finish = True
            x += 1
        elif board_set[x, y, set_num] == 2:
            if y == H - 1: finish = True
            y += 1
        else:
            if x == 0: finish = True
            x -= 1
        next_opt = x + y * W
        if not finish and mask[x, y] == 1:
            finish = True
    return move_num


@ti.func
def score(i: ti.uint64, index: ti.types.vector(TOTAL, ti.uint32)) -> ti.uint32:
    total_score = 0
    for j in range(TOTAL):
        while index[j] > 0:
            total_score += play(i, j)
            index[j] -= 1
    return total_score


@ti.func
def get_index(idx: ti.uint64) -> ti.types.vector(TOTAL, ti.uint32):
    index = ti.Vector(np.zeros(TOTAL), dt=ti.int32)
    index[0] = MOVE
    cur_idx = idx
    for i in range(TOTAL - 1):
        if cur_idx <= 0: break
        for j in range(index[i] + 1):
            threshold = factorio[j, (TOTAL - i - 1) + j - 1]
            if cur_idx < threshold or j == index[i] + 1:
                index[i] -= j
                index[i + 1] = j
                break
            cur_idx -= threshold
    return index


@ti.kernel
def exhaustive_one_set(start: ti.uint64) -> res:
    board_set.fill(0)
    result = res()
    for i in range(start, ti.min(start + SET_SIZE, factorio[MOVE, TOTAL + MOVE - 1])):
        set_num = ti.int32(i - start)
        res_set[set_num].idx = i
        res_set[set_num].index = get_index(i)
        res_set[set_num].score = score(i, res_set[set_num].index)
    for i in range(SET_SIZE):
        ti.atomic_max(max_score[None], res_set[i].score)
    for i in range(SET_SIZE):
        if res_set[i].score == max_score[None]:
            result = res_set[i]
    return result


def begin():
    i = 0
    result = res()
    start_time = time.time()
    while i < factorio[MOVE, TOTAL + MOVE - 1]:
        cur_result = exhaustive_one_set(i)
        if cur_result.score > result.score:
            result = cur_result
        process = float(i) / factorio[MOVE, TOTAL + MOVE - 1]
        i += SET_SIZE
        time_usage = time.time() - start_time
        print(f"\rprocessing: {'{:.3f}'.format(process * 100)}%, time usage: {'{:.2f}'.format(time_usage)}s"
              f" ({'{:.2f}'.format(time_usage/max(0.001,process)*(1.0-process))}s remaining),"
              f" score:{result.score * 90}({result.score}), opt:{result.index}", end='')
    print(f"\nfinal result: score:{result.score * 90}({result.score}), id:{result.idx}, opt:{result.index}")


init()
begin()
