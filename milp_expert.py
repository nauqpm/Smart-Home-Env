# milp_expert.py
"""
Generate expert schedules by solving full-day MILP (Problem 13).
Produces output: output/example_expert_schedule.json
Dependencies: numpy, pandas, pulp (if available)
Usage: python milp_expert.py
"""
import json
import os
import numpy as np

try:
    import pulp
except Exception:
    pulp = None
    print("WARNING: pulp not installed. This script will return a heuristic schedule (not MILP).")

def build_and_solve_milp(price: np.ndarray, pv: np.ndarray, config: dict):
    T = len(price)
    critical = config.get("critical", 0.0)
    if hasattr(critical, "__len__") and len(critical) == T:
        P_cr = np.array(critical)
    else:
        P_cr = np.array([float(critical)] * T)
    adjustable = config.get("adjustable", [])
    su = config.get("shiftable_su", [])
    si = config.get("shiftable_si", [])
    beta = config.get("beta", 0.5)
    rho_b = price
    rho_s = beta * rho_b

    # If pulp not present -> heuristic fallback
    if pulp is None:
        print("pulp not available -> heuristic schedule created.")
        schedule = {
            "P_ad": np.array([[ad['P_com'] if rho_b[t] < rho_b.mean() else ad['P_min'] for ad in adjustable] for t in range(T)]),
            "z_su": np.zeros((len(su), T), dtype=int),
            "z_si": np.zeros((len(si), T), dtype=int),
            "P_b": np.zeros(T),
            "P_s": np.zeros(T)
        }
        for i, item in enumerate(su):
            start = item['t_s']
            L = item['L']
            schedule['z_su'][i, start:start+L] = 1
        for i, item in enumerate(si):
            start = item['t_s']
            rate = item['rate']
            L = int(round(item['E'] / rate))
            schedule['z_si'][i, start:start+L] = 1
        return schedule

    # Build MILP with pulp
    prob = pulp.LpProblem("FullDay_HEM", pulp.LpMinimize)
    # Variables
    P_ad = {}
    for t in range(T):
        for i in range(len(adjustable)):
            ad = adjustable[i]
            P_ad[(t,i)] = pulp.LpVariable(f"P_ad_{t}_{i}", lowBound=ad['P_min'], upBound=ad['P_max'], cat='Continuous')
    # SU start decision variables (one start slot)
    z_su = {}
    for i in range(len(su)):
        kmin = su[i]['t_s']
        kmax = su[i]['t_f'] - su[i]['L'] + 1
        for k in range(kmin, kmax+1):
            z_su[(i,k)] = pulp.LpVariable(f"z_su_{i}_{k}", cat='Binary')
    # SI on/off per timestep
    z_si = {}
    for i in range(len(si)):
        for t in range(si[i]['t_s'], si[i]['t_f']+1):
            z_si[(i,t)] = pulp.LpVariable(f"z_si_{i}_{t}", cat='Binary')
    # Transactions
    P_b = {t: pulp.LpVariable(f"P_b_{t}", lowBound=0.0) for t in range(T)}
    P_s = {t: pulp.LpVariable(f"P_s_{t}", lowBound=0.0) for t in range(T)}
    z_b = {t: pulp.LpVariable(f"z_b_{t}", cat='Binary') for t in range(T)}
    z_s = {t: pulp.LpVariable(f"z_s_{t}", cat='Binary') for t in range(T)}
    bigM = 1e5

    objective = 0
    for t in range(T):
        # compute P_su_t from start decisions
        P_su_t = 0
        for i in range(len(su)):
            rate = su[i]['rate']
            L = su[i]['L']
            # if start k implies ON at t when k <= t <= k+L-1
            terms = []
            kmin = su[i]['t_s']
            kmax = su[i]['t_f'] - L + 1
            for k in range(kmin, kmax+1):
                if k <= t <= k + L - 1:
                    terms.append(z_su[(i,k)])
            if terms:
                P_su_t += rate * pulp.lpSum(terms)
        # compute P_si_t
        P_si_t = 0
        for i in range(len(si)):
            rate = si[i]['rate']
            if (i,t) in z_si:
                P_si_t += rate * z_si[(i,t)]
        # adjustable sum
        P_ad_sum = pulp.lpSum([P_ad[(t,i)] for i in range(len(adjustable))]) if adjustable else 0
        P_net_expr = P_cr[t] + P_ad_sum + P_su_t + P_si_t - pv[t]
        prob += (P_b[t] - P_s[t] == P_net_expr)
        prob += P_b[t] <= z_b[t] * bigM
        prob += P_s[t] <= z_s[t] * bigM
        prob += z_b[t] + z_s[t] <= 1
        # cost
        objective += rho_b[t] * P_b[t] - rho_s[t] * P_s[t]
        # linearized discomfort surrogate
        for i in range(len(adjustable)):
            ad = adjustable[i]
            u = pulp.LpVariable(f"u_{t}_{i}", lowBound=0.0)
            prob += u >= ad['P_com'] - P_ad[(t,i)]
            prob += u >= P_ad[(t,i)] - ad['P_com']
            objective += ad['alpha'] * u

    prob += objective
    # SU constraint: exactly one start
    for i in range(len(su)):
        kmin = su[i]['t_s']
        kmax = su[i]['t_f'] - su[i]['L'] + 1
        prob += pulp.lpSum([z_su[(i,k)] for k in range(kmin, kmax+1)]) == 1
    # SI constraint: exact number of ON slots
    for i in range(len(si)):
        tmin = si[i]['t_s']
        tmax = si[i]['t_f']
        Lsi = int(round(si[i]['E'] / si[i]['rate']))
        prob += pulp.lpSum([z_si[(i,t)] for t in range(tmin, tmax+1)]) == Lsi

    solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=60)
    res = prob.solve(solver)
    print("MILP status:", pulp.LpStatus[res])

    # extract
    P_ad_res = np.zeros((len(adjustable), T))
    for t in range(T):
        for i in range(len(adjustable)):
            P_ad_res[i,t] = pulp.value(P_ad[(t,i)])
    z_su_res = np.zeros((len(su), T), dtype=int)
    for i in range(len(su)):
        kmin = su[i]['t_s']
        kmax = su[i]['t_f'] - su[i]['L'] + 1
        for k in range(kmin, kmax+1):
            val = pulp.value(z_su[(i,k)])
            if val is not None and val > 0.5:
                for tt in range(k, k + su[i]['L']):
                    z_su_res[i, tt] = 1
    z_si_res = np.zeros((len(si), T), dtype=int)
    for i in range(len(si)):
        for t in range(si[i]['t_s'], si[i]['t_f']+1):
            val = pulp.value(z_si[(i,t)])
            if val is not None and val > 0.5:
                z_si_res[i,t] = 1
    P_b_res = np.array([pulp.value(P_b[t]) for t in range(T)])
    P_s_res = np.array([pulp.value(P_s[t]) for t in range(T)])
    schedule = {'P_ad': P_ad_res.T, 'z_su': z_su_res, 'z_si': z_si_res, 'P_b': P_b_res, 'P_s': P_s_res}
    return schedule

if __name__ == "__main__":
    T = 24
    np.random.seed(0)
    price = 0.1 + 0.2 * np.random.rand(T)
    pv = np.clip(1.5 * np.sin(np.linspace(0, 3.14, T)) + 0.2*np.random.randn(T), 0, None)
    config = {
        'critical': [0.3]*T,
        'adjustable': [
            {'P_min':0.1, 'P_max':1.5, 'P_com':1.2, 'alpha':0.06},
            {'P_min':0.0, 'P_max':1.2, 'P_com':1.0, 'alpha':0.12}
        ],
        'shiftable_su': [ {'rate':0.5, 'L':2, 't_s':6, 't_f':20}, {'rate':0.6, 'L':1, 't_s':8, 't_f':22} ],
        'shiftable_si': [ {'rate':1.0, 'E':4.0, 't_s':0, 't_f':23} ],
        'beta': 0.5
    }
    schedule = build_and_solve_milp(price, pv, config)
    out = {'price': price.tolist(), 'pv': pv.tolist(), 'schedule': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k,v in schedule.items()}}
    os.makedirs('output', exist_ok=True)
    with open('output/example_expert_schedule.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("Wrote output/example_expert_schedule.json")
