import cvxpy as cp
import numpy as np

def FairnessLambda(
    sg: np.ndarray,
    trg: np.ndarray,
    a: float,
    b: float,
    tau: float,
    d: int,
    B: float,
) -> tuple[np.ndarray, float]:
    """Solve fairness-aware min-max program over eigenvalues."""
    sg = np.asarray(sg, dtype=float)
    trg = np.asarray(trg, dtype=float)

    if sg.ndim != 2:
        raise ValueError("sg must have shape (numgroups, k).")
    if trg.ndim != 1 or trg.shape[0] != sg.shape[0]:
        raise ValueError("trg must have shape (numgroups,) matching sg.")

    groups, k = sg.shape
    if d < k:
        raise ValueError("d must be >= k.")

    budget = B - tau * d
    if budget <= 0:
        lamzero = np.zeros(k, dtype=float)
        lamplus = lamzero + tau
        constinv = (d - k) / tau if d > k else 0.0
        vals = []
        for g in range(groups):
            trpg = a * (
                sg[g].dot(lamplus) + tau * (float(trg[g]) - sg[g].sum())
            )
            dist = trpg + b * (np.sum(1.0 / lamplus) + constinv)
            vals.append(dist)
        return lamzero, float(max(vals))

    lam = cp.Variable(k, nonneg=True)
    tval = cp.Variable()

    lamplus = lam + tau
    constinv = (d - k) / tau if d > k else 0.0

    cons = [cp.sum(lam) <= budget]
    for g in range(groups):
        trpg = a * (
            sg[g] @ lamplus + tau * (float(trg[g]) - float(np.sum(sg[g])))
        )
        invterm = b * (cp.sum(1.0 / lamplus) + constinv)
        cons.append(trpg + invterm <= tval)

    obj = cp.Minimize(tval)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        lamuni = np.full(k, budget / k, dtype=float) if k > 0 else np.array([])
        return lamuni, float("inf")

    return np.array(lam.value, dtype=float), float(tval.value)
