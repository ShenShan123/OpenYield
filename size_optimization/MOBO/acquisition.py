import numpy as np
from individual import Individual


def extract_pareto_front(Y_train):
    n = len(Y_train)
    pareto_front_indices = []
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i == j:
                continue
            if dominates(Y_train[j], Y_train[i]):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front_indices.append(i)
    return pareto_front_indices


def dominates(y1, y2):
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    return np.all(y1 <= y2) and np.any(y1 < y2)


def compute_hypervolume_2d(pareto_front, ref_point):
    """
    计算2D超体积
    pareto_front: list of [MRED, PDP]
    ref_point: [max_MRED, max_PDP]
    """
    if len(pareto_front) == 0:
        return 0.0
    
    # 按第一个目标排序
    sorted_front = sorted(pareto_front, key=lambda x: x[0])
    
    hv = 0.0
    prev_y = ref_point[1]
    
    for point in sorted_front:
        if point[0] < ref_point[0] and point[1] < ref_point[1]:
            width = ref_point[0] - point[0]
            height = prev_y - point[1]
            hv += width * height
            prev_y = point[1]
    
    return hv


def compute_hypervolume(pareto_front, ref_point, n_samples=2000, seed=None):
    if len(pareto_front) == 0:
        return 0.0
    pf = np.asarray(pareto_front, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    if pf.ndim == 1:
        pf = pf.reshape(1, -1)
    if pf.shape[1] == 2:
        return compute_hypervolume_2d(pf.tolist(), ref.tolist())
    pf = pf[np.all(pf < ref, axis=1)]
    if pf.size == 0:
        return 0.0
    lower = np.min(pf, axis=0)
    if np.any(lower >= ref):
        return 0.0
    rng = np.random.default_rng(seed)
    samples = rng.uniform(low=lower, high=ref, size=(n_samples, ref.shape[0]))
    dominated = np.zeros(samples.shape[0], dtype=bool)
    for point in pf:
        dominated |= np.all(samples >= point, axis=1)
    volume = float(np.prod(ref - lower))
    return volume * float(np.mean(dominated))


def compute_hypervolume_improvement(new_point, pareto_front, ref_point, n_samples=2000, seed=None):
    new_point = np.asarray(new_point, dtype=float)
    ref_point = np.asarray(ref_point, dtype=float)
    if np.any(new_point >= ref_point):
        return 0.0
    for pf_point in pareto_front:
        if dominates(pf_point, new_point):
            return 0.0
    hv_before = compute_hypervolume(pareto_front, ref_point, n_samples=n_samples, seed=seed)
    hv_after = compute_hypervolume(list(pareto_front) + [new_point], ref_point, n_samples=n_samples, seed=seed)
    return float(max(hv_after - hv_before, 0.0))


def monte_carlo_ehvi(mu, sigma, pareto_front, ref_point, n_samples=200, hv_samples=500):
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    rng = np.random.default_rng()
    samples = rng.normal(loc=mu, scale=sigma, size=(n_samples, mu.shape[0]))
    improvements = []
    for idx, sample in enumerate(samples):
        improvement = compute_hypervolume_improvement(
            sample,
            pareto_front,
            ref_point,
            n_samples=hv_samples,
            seed=idx
        )
        improvements.append(improvement)
    return float(np.mean(improvements))


def create_individuals_from_front(X_front, Y_front):
    """
    将帕累托前沿转换为Individual对象列表
    
    参数:
        X_front: array of shape (n, 9), 设计变量
        Y_front: array of shape (n, m), 目标值
    
    返回:
        Individual对象列表
    """
    individuals = []
    for i in range(len(X_front)):
        ind = Individual()
        ind.features = list(X_front[i])
        ind.objectives = list(Y_front[i])
        individuals.append(ind)
    return individuals
