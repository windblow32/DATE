import numpy as np
import pandas as pd
import matplotlib

from various import wilcoxon_holm

matplotlib.use('agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon, friedmanchisquare
import networkx
from scipy.stats import studentized_range, norm

def critical_difference(k, N, alpha=0.05, method='nemenyi'):
    """
    k: 方法个数
    N: 数据集数量（用于计算平均rank的样本数）
    method: 'nemenyi' 或 'bonferroni-dunn'
    返回值单位为“rank”，可直接用于 CD 图
    """
    if method.lower().startswith('nemenyi'):
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
        return float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * N)))
    elif method.lower() in ('bonferroni-dunn', 'bd', 'dunn'):
        z = norm.ppf(1 - alpha / (2.0 * (k - 1)))
        return float(z * np.sqrt(k * (k + 1) / (6.0 * N)))
    else:
        raise ValueError('Unknown CD method: {}'.format(method))


def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        if not len(lr):
            yield ()
        else:
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant


    fig = plt.figure(figsize=(width, 1.3 * height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    # ----- draw CD bracket and label -----
    if cd is not None:
        # 选择一个不会挡住刻度的位置（靠左/靠右都可以）
        left_rank = lowv
        # 若放不下，就往里挪
        if left_rank + cd > highv - 0.2:
            left_rank = max(lowv, highv - cd - 0.5)

        y = cline - 0.6   # 括号位于刻度线之上（注意本图y轴是倒置的）
        cap = 0.06         # 括号两端的“立杆”高度

        x1 = rankpos(left_rank)
        x2 = rankpos(left_rank + cd)

        # 横线
        line([(x1, y), (x2, y)], linewidth=2)
        # 两端立杆
        line([(x1, y - cap), (x1, y + cap)], linewidth=2)
        line([(x2, y - cap), (x2, y + cap)], linewidth=2)
        # 文字：CD=xxx
        text((x1 + x2) / 2.0, y - cap - 0.02, f'CD={cd:.3f}',
             ha='center', va='bottom', size=20)
    # ----- end CD -----

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=20)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    max_rank_idx = np.argmin(ssums)  # 获取排名最高方法的索引

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        color = 'red' if i == max_rank_idx else 'black'
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=5 if color == 'red' else linewidth, color=color)  # 设置红色线条更粗
        if labels:
            text(textspace + 0.53, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=20)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=20, color=color)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        color = 'red' if i == max_rank_idx else 'black'
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=5 if color == 'red' else linewidth, color=color)
        if labels:
            text(textspace + scalewidth - 0.45, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=20)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=20, color=color)

    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    start = cline + 0.2
    side = -0.02
    height = 0.1

    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height


def form_cliques(p_values, nnames):
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False):
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:
        print(p)

    k = len(average_ranks)
    N = df_perf.shape[0]   # 行数即数据集个数
    cd_value = critical_difference(k=k, N=N, alpha=alpha, method='nemenyi')  # 或 'bonferroni-dunn'

    graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=cd_value, cdmethod='nemenyi', reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
            'color': 'black',
            'weight': 'normal',
            'size': 20,
            }
    if title:
        plt.title(title, fontdict=font, y=0.9, x=0.5)
    plt.savefig('/Users/ovoniko/Documents/GitHub/work_crr_test/draw/cd_small.eps',
                bbox_inches='tight')
    plt.savefig('/Users/ovoniko/Documents/GitHub/work_crr_test/draw/cd_small.png',
                bbox_inches='tight')


df_perf = pd.read_csv(r'/Users/ovoniko/Documents/GitHub/work_crr_test/draw/cd_DATE.csv', index_col=False)  # 在这里更改要读取的csv的文件位置

draw_cd_diagram(df_perf=df_perf, title='Accuracy', labels=True)
