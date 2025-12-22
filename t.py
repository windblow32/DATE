import sys
from os import name


def solve():
    data = sys.stdin.read().strip().split()
    n = int(data[0])
    s = data[1].strip()
    m = n // 2
    # 第1对
    a = ord(s[0]) - 48
    b = ord(s[1]) - 48
    dp = [int(a != x) + int(b != x) for x in range(10)]
    INF = 10 ** 9
    idx = 2
    for _ in range(1, m):
        a = ord(s[idx]) - 48
        b = ord(s[idx + 1]) - 48
        idx += 2
        cost = [int(a != x) + int(b != x) for x in range(10)]

        # 上一层最小值与次小值
        best1_val, best1_idx = INF, -1
        best2_val = INF
        for x in range(10):
            v = dp[x]
            if v < best1_val:
                best2_val = best1_val
                best1_val, best1_idx = v, x
            elif v < best2_val:
                best2_val = v

        new_dp = [0] * 10
        for x in range(10):
            prev = best1_val if x != best1_idx else best2_val
            new_dp[x] = cost[x] + prev
        dp = new_dp

    print(min(dp))


if name == "main":
    solve()
