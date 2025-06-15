import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bisect
from collections import defaultdict
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt

config = json.load(open(os.path.join(os.getcwd(), 'config.json'), 'r'))
mona_dir = config["json_dir"]
ass_value_dic = {
    'attackPercentage': 0.0583,
    'attackStatic': 19.45,
    'critical': 0.0389,
    'criticalDamage': 0.0777,
    'defendPercentage': 0.0729,
    'defendStatic': 23.15,
    'elementalMastery': 23.31,
    'lifePercentage': 0.0583,
    'lifeStatic': 298.75,
    'recharge': 0.0648
}
ass_reward_dic = config["ass_reward_dic"]

# 从mona的数据中载入并转化为dataframe
def json_to_artifact_df(path):
    # 1. 载入 JSON（自动跳过 version 字段）
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    # 2. 遍历每个 position（flower, feather, sand, cup, head）
    for position, artifacts in data.items():
        if position == 'version':
            continue
        for art in artifacts:
            rec = art.copy()

            records.append(rec)

    df = pd.DataFrame(records)
    return df

df = json_to_artifact_df(mona_dir)
print(f"总共有 {len(df)} 条记录，每行代表一个圣遗物。")

# 筛选符合条件的圣遗物
filtered_df = df[(df["level"]==20) &( 
                ((df["position"]=="sand") & (df["mainTag"].apply(lambda x: True if x["name"] in config["sand"] else False))) |
                ((df["position"]=="cup") & (df["mainTag"].apply(lambda x: True if x["name"] in config["cup"] else False))) |
                ((df["position"]=="head") & (df["mainTag"].apply(lambda x: True if x["name"] in config["head"] else False))) |
                (df["position"]=="flower") | (df["position"]=="feather"))]

# 区分初始5词条或者4词条的圣遗物
def filter_normalTags(ls):
    count = 0
    for tmp_dic in ls:
        tmp_name = tmp_dic['name']
        tmp_value = tmp_dic['value']
        tmp_ratio = tmp_value / ass_value_dic[tmp_name]
        if tmp_ratio <= 1.2:
            count += 1
        elif tmp_ratio <= 2.05:
            count += 2
        elif tmp_ratio <= 2.9:
            count += 3
        elif tmp_ratio <= 3.8:
            count += 4
    if count == 9:
        return 5
    return 4


# 获得每一件圣遗物的分数
def get_reward(ls):
    reward = 0
    for tmp_dic in ls:
        tmp_name = tmp_dic['name']
        tmp_value = tmp_dic['value']
        reward += tmp_value * config["ass_reward_dic"][tmp_name] / ass_value_dic[tmp_name]
    return reward

filtered_df = filtered_df.copy()
filtered_df["init_num"] = filtered_df["normalTags"].apply(filter_normalTags)
filtered_df['reward'] = filtered_df['normalTags'].apply(get_reward)

# 计算分组最大的reward值
# （可选）将索引从字符串转成整数
filtered_df.index = filtered_df.index.astype(int)

# 确保 reward 是数值型
filtered_df['reward'] = pd.to_numeric(filtered_df['reward'])


# 按 setName 是否为 目标的圣遗物 划分两类
for pos in ['sand', 'cup', 'head', 'flower', 'feather']:
    # “ObsidianCodex” 类（目标圣遗物）
    mask_codex = (filtered_df['position'] == pos) & (filtered_df['setName'] == config["artifact_name"])
    if mask_codex.any():
        max_codex = filtered_df.loc[mask_codex, 'reward'].max()
        filtered_df.loc[mask_codex, 'groupMaxReward'] = max_codex

    # “非 ObsidianCodex” 类（目标圣遗物）
    mask_other = (filtered_df['position'] == pos) & (filtered_df['setName'] != config["artifact_name"])
    if mask_other.any():
        max_other = filtered_df.loc[mask_other, 'reward'].max()
        filtered_df.loc[mask_other, 'groupMaxReward'] = max_other
        
# 计算trajectory的概率，进行洗练的结果分布
class FourOptionProcess:
    """
    Exhaustive-enumeration engine for the “4-option / 5-step” increment game.

    Parameters
    ----------
    init_vals : Tuple[float, float, float, float]
        Initial (A, B, C, D).  **Must be one-decimal numbers** so that
        `v*10` is an integer (e.g. 0.8, 1.5 …).  They are stored internally
        as integers in “tenths”.
    n : int
        Hyper-parameter in {2, 3, 4}:  A + B must be incremented at least `n`
        times in total.

    Attributes
    ----------
    final_dist : Dict[Tuple[int, int, int, int], float]
        Mapping “scaled (A,B,C,D) → probability”.  Keys are integers that
        already include the ×10 scaling.
    scale : int
        The scaling factor (fixed to 10).
    """

    # fixed per-step increment choices (already ×10 → integers)
    INC = (10, 9, 8, 7)          # 1, 0.9, 0.8, 0.7

    def __init__(self, init_vals: Tuple[float, float, float, float], n: int, L: int):
        if n not in (2, 3, 4):
            raise ValueError("n must be 2, 3, or 4")
        self.n = n
        self.L = L
        self.scale = 10
        # store all values as integers in “tenths”
        self._init_int = [int(round(v * self.scale)) for v in init_vals]
        if any(abs(v * self.scale - r) > 1e-9 for v, r in zip(init_vals, self._init_int)):
            raise ValueError("All initial values must have exactly one decimal place")
        # container for the enumeration result
        self.final_dist: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
        self._enumerate()
        self._check_total_prob()

    # --------------------------------------------------------------------- #
    # public helpers
    # --------------------------------------------------------------------- #
    def combo_dist(self,
                   coeffs: Tuple[float, float, float, float],
                   *,
                   M: Optional[float] = None,
                   plot: bool = False,
                   smooth_bandwidth=0.05) -> Dict[float, float]:
        """
        Return the distribution of  Σ coeff_i * X_i   (or  max(M, ·)  if M given).

        Parameters
        ----------
        coeffs : (a, b, c, d)
            Linear-combination coefficients.
        M : float, optional
            If supplied, the random variable becomes  max(M, Σ …).
        plot : bool
            If True, draw a stem plot via matplotlib.

        Returns
        -------
        Dict[float, float]
            Mapping “value → probability”.  Values are real numbers (not scaled).
        """
        dist: Dict[float, float] = defaultdict(float)
        a, b, c, d = coeffs

        for (Ai, Bi, Ci, Di), p in self.final_dist.items():
            val = round((a * Ai + b * Bi + c * Ci + d * Di) / self.scale, 4)
            if M is not None:
                val = max(M, val)
            dist[val] += p

        # ----------- 可视化（PMF + CDF） ---------------------------------
        if plot:

            xs = np.array(sorted(dist))
            ps = np.array([dist[x] for x in xs])

            # ---- (1) 光滑 PMF -----------------------------------------
            grid = np.linspace(xs.min() - 4*smooth_bandwidth,
                               xs.max() + 4*smooth_bandwidth,
                               1200)
            gauss_norm = 1.0 / (np.sqrt(2*np.pi) * smooth_bandwidth)
            smooth = np.zeros_like(grid)
            for xi, pi in zip(xs, ps):
                smooth += pi * gauss_norm * np.exp(-0.5 * ((grid - xi) /
                                                           smooth_bandwidth) ** 2)

            # ---- (2) 原始 CDF -----------------------------------------
            cdf_vals = np.cumsum(ps)

            # ---- 画图布局 --------------------------------------------
            fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4))

            # 左：平滑 PMF
            ax_pmf.plot(grid, smooth)
            ax_pmf.set_title(f"Smoothed PMF (Gaussian h={smooth_bandwidth})")
            ax_pmf.set_xlabel("value")
            ax_pmf.set_ylabel("probability density")

            # 右：CDF 阶梯图
            ax_cdf.step(xs, cdf_vals, where="post")
            ax_cdf.set_ylim(0, 1.02)
            ax_cdf.set_title("Raw CDF")
            ax_cdf.set_xlabel("value")
            ax_cdf.set_ylabel("cumulative probability")

            fig.suptitle("Distribution of " +
                         ("max(M, Σ)" if M is not None else "Σ coeff·X"))
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

        return dist

    # --------------------------------------------------------------------- #
    # internal: depth-first enumeration
    # --------------------------------------------------------------------- #
    def _enumerate(self):
        """Depth-first walk of the ≤16⁵ trajectory tree."""
        def dfs(step: int,
                vals: List[int],
                b_count: int,
                prob: float,):
            if step == self.L:                                  # reached leaf
                self.final_dist[tuple(vals)] += prob
                return

            remaining = self.L - step
            # decide the allowed option set
            if b_count + remaining == self.n and b_count < self.n:
                options = (0, 1)                           # A, B only
                option_prob = 0.5
            else:
                options = (0, 1, 2, 3)                     # A, B, C, D
                option_prob = 0.25

            # branch over every (option, increment)
            step_weight = option_prob * 0.25               # also 1/4 for INC
            for opt in options:
                for inc in self.INC:
                    new_vals = vals.copy()
                    new_vals[opt] += inc
                    dfs(step + 1,
                        new_vals,
                        b_count + (opt in (0, 1)),
                        prob * step_weight)

        dfs(0, self._init_int.copy(), 0, 1.0)

    def _check_total_prob(self):
        total = sum(self.final_dist.values())
        if abs(total - 1.0) > 1e-12:
            raise RuntimeError(f"probabilities sum to {total}, not 1")
        

def _cdf_from_pmf(pmf: Dict[float, float]) -> Tuple[List[float], List[float]]:
    """
    helper – 把 {x:p} 转成 (xs, cdf_vals) 便于二分查表
    """
    xs = sorted(pmf)
    cdf = []
    s = 0.0
    for x in xs:
        s += pmf[x]
        cdf.append(s)
    return xs, cdf


def _cdf_query(xs: List[float], cdf: List[float], x: float) -> float:
    """
    二分返回 F(x) – P{X ≤ x}
    """
    idx = bisect.bisect_right(xs, x) - 1
    return 0.0 if idx < 0 else cdf[idx]


def max_of_runs(init_vals: Tuple[float, float, float, float],
                coeffs: Tuple[float, float, float, float],
                n_list: List[int],
                M: float,
                eps: float = 1e-3,
                L: int = 5
                ) -> Tuple[Dict[float, float], float, float]:
    """
    枚举 n_list 指定的若干独立过程，返回
    (1) Z = max(V₁,…,V_k,M) 的 pmf
    (2) E[Z]
    (3) P{Z > M + eps}

    ----------
    Parameters
    ----------
    init_vals : (A,B,C,D) 的初值
    coeffs    : (a,b,c,d) 线性组合系数
    n_list    : 诸如 [2,2,2,3]，每个元素都是一次独立实验的 n
    M         : baseline 参加 max
    eps       : “显著高于 M” 的阈值 (默认 1e-3)

    Returns
    -------
    pmf_Z : dict   value → probability
    expectation_Z : float
    prob_gt_M_eps : float
    """
    # --------- 1. 单次实验的分布 -----------------------------
    single_dists = []
    for n in n_list:
        proc = FourOptionProcess(init_vals, n, L)
        pmf = proc.combo_dist(coeffs, plot=False)      # dict
        single_dists.append(pmf)

    # --------- 2. 把每个 pmf → (xs, cdf) --------------------
    cdf_tables = []
    overall_support = {M}
    for pmf in single_dists:
        cdf_tables.append(_cdf_from_pmf(pmf))
        overall_support.update(pmf.keys())

    support_sorted = sorted(overall_support)

    # --------- 3. 计算 Z 的 CDF & PMF ------------------------
    F_prev = 0.0
    pmf_Z: Dict[float, float] = {}
    for x in support_sorted:
        if x < M:
            F_x = 0.0
        else:
            prod = 1.0
            for xs, cdf in cdf_tables:
                prod *= _cdf_query(xs, cdf, x)
            F_x = prod   # CDF_Z(x)
        pmf_Z[x] = F_x - F_prev
        F_prev = F_x

    # 清理极小概率噪声
    pmf_Z = {x: p for x, p in pmf_Z.items() if p > 1e-15}

    # --------- 4. 期望 & 超过概率 ----------------------------
    exp_Z = sum(x * p for x, p in pmf_Z.items())
    prob_exceed = sum(p for x, p in pmf_Z.items() if x > M + eps)

    return pmf_Z, exp_Z, prob_exceed

# 获得每一个圣遗物副词条对应的权重
def get_coeffs(normalTags):
    ass_ls = []
    for tmp_dic in normalTags:
        ass_ls.append(tmp_dic["name"])
    coeffs = []
    for name in ass_ls:
        coeffs.append(ass_reward_dic[name])
    coeffs = sorted(coeffs, reverse=True)
    return tuple(coeffs)

def compute_run_metrics(row):
    # 拿到这一行所需的输入
    coeffs = get_coeffs(row['normalTags'])
    init_vals = config["init_vals"]
    n_list = [2, 2] if row["position"] in ["flower", "feather"] else [2]
    M = row["groupMaxReward"]
    L = row["init_num"]

    # 计算主方案（花费两个羽毛的方案）
    _, E_Z, p_big = max_of_runs(init_vals, coeffs, n_list, M, L=L)
    # 计算 3 次和 4 次的备选方案
    _, E_Z_3, p_big_3 = max_of_runs(init_vals, coeffs, [3], M, L=L)
    _, E_Z_4, p_big_4 = max_of_runs(init_vals, coeffs, [4], M, L=L)

    # 返回一个 Series，会自动按名字对齐到 DataFrame 的列
    return pd.Series({
        'p_big':    p_big,
        'Aug':      E_Z   - M,
        'p_big_3':  p_big_3,
        'Aug_3':    E_Z_3 - M,
        'p_big_4':  p_big_4,
        'Aug_4':    E_Z_4 - M,
    })

# 短期收益计算
filtered_df[['p_big','Aug','p_big_3','Aug_3','p_big_4','Aug_4']] = filtered_df.apply(compute_run_metrics, axis=1)

# 按照指定的列名排序，获得最优的Top5
top5 = filtered_df.nlargest(5, config["short_target"])

# 输出最终结果
print("短期最优选Top5:\n")
for i in range(len(top5)):
    row = top5.iloc[i]
    print(f"第{i+1}条选择，其部位为{row['position']}，圣遗物名称为{row['setName']}")
    print(f"主词条为{row['mainTag']['name']}， 副词条为{row['normalTags']}")
    print(f"在花费两根启圣之尘的情况下，有{row['p_big']:.2%}的概率有提升，其提升的期望为{row['Aug']:.4f}")
    print(f"在触发高阶重塑的情况下，有{row['p_big_3']:.2%}的概率有提升，其提升的期望为{row['Aug_3']:.4f}")
    print(f"在触发谕告重塑的情况下，有{row['p_big_4']:.2%}的概率有提升，其提升的期望为{row['Aug_4']:.4f}\n")
    
# 对Top5的选项进行长期预测
def plot_long_term_fig(row):
    # 拿到这一行所需的输入
    coeffs = get_coeffs(row['normalTags'])
    init_vals = config["init_vals"]
    n_list = [2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 4] if row["position"] in ["flower", "feather"] else [2, 2, 3, 2, 2, 3, 2, 2, 4]
    M = row["groupMaxReward"]
    L = row["init_num"]

    # 计算主方案（花费两个羽毛的方案）
    pmf_Z, E_Z, p_big = max_of_runs(init_vals, coeffs, n_list, M, L=L)

    smooth_bandwidth = config["smooth_bandwidth"]
    xs = np.array(sorted(pmf_Z))
    ps = np.array([pmf_Z[x] for x in xs])

    # ---- (1) 光滑 PMF -----------------------------------------
    grid = np.linspace(xs.min() - 4*smooth_bandwidth,
                        xs.max() + 4*smooth_bandwidth,
                        1200)
    gauss_norm = 1.0 / (np.sqrt(2*np.pi) * smooth_bandwidth)
    smooth = np.zeros_like(grid)
    for xi, pi in zip(xs, ps):
        smooth += pi * gauss_norm * np.exp(-0.5 * ((grid - xi) /
                                                    smooth_bandwidth) ** 2)

    # ---- (2) 原始 CDF -----------------------------------------
    cdf_vals = np.cumsum(ps)

    # ---- 画图布局 --------------------------------------------
    fig, (ax_pmf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4))

    # 左：平滑 PMF
    ax_pmf.plot(grid, smooth)
    ax_pmf.set_title(f"Smoothed PMF (Gaussian h={smooth_bandwidth})")
    ax_pmf.set_xlabel("value")
    ax_pmf.set_ylabel("probability density")

    # 右：CDF 阶梯图
    ax_cdf.step(xs, cdf_vals, where="post")
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.set_title("Raw CDF")
    ax_cdf.set_xlabel("value")
    ax_cdf.set_ylabel("cumulative probability")

    fig.suptitle(f"Distribution with position {row['position']} and name {row['setName']}, while the expectation of augment is {E_Z-M:.4f}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{row['setName']}_{row['position']}_{E_Z-M:.4f}.png")
    # plt.show()
    plt.close(fig)
    
for i in range(len(top5)):
    plot_long_term_fig(top5.iloc[i])