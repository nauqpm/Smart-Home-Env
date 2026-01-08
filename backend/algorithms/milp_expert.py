# milp_expert.py
"""
MILP Expert for SmartHomeEnv
- Generates full-day (T x 7) continuous action schedule
- Compatible with SmartHomeEnv action space
- Used for: IL / PPO warm-start / baseline comparison
"""

import numpy as np

try:
    import pulp
except Exception:
    pulp = None
    print("WARNING: pulp not installed. Heuristic expert will be used.")


# =====================================================
# Main MILP Solver
# =====================================================
def solve_milp_expert(price, pv, config):
    """
    Return expert action schedule: shape (T, 7)
    Action order:
    [battery, ac_living, ac_master, ac_bed2, ev, wm, dw]
    """

    T = len(price)
    battery = config["battery"]
    ev_cfg = config["ev"]
    ac_cfg = config["ac"]
    wm_cfg = config["wm"]
    dw_cfg = config["dw"]

    # -------------------------------------------------
    # Heuristic fallback (if pulp not available)
    # -------------------------------------------------
    if pulp is None:
        actions = np.zeros((T, 7))
        for t in range(T):
            if pv[t] > np.mean(pv):
                actions[t, 0] = 0.6      # charge battery
                actions[t, 4] = 0.5      # charge EV
                actions[t, 1:4] = 0.4
            else:
                actions[t, 0] = -0.4     # discharge
                actions[t, 1:4] = 0.2
        return actions

    # -------------------------------------------------
    # Build MILP
    # -------------------------------------------------
    prob = pulp.LpProblem("SmartHome_MILP_Expert", pulp.LpMinimize)

    # ---------- Variables ----------
    P_bat = {t: pulp.LpVariable(f"P_bat_{t}",
                                lowBound=-battery["p_discharge_max_kw"],
                                upBound=battery["p_charge_max_kw"])
             for t in range(T)}

    SOC = {t: pulp.LpVariable(f"SOC_{t}",
                              lowBound=battery["soc_min"],
                              upBound=battery["soc_max"])
           for t in range(T)}

    P_ac = {(t, r): pulp.LpVariable(f"P_ac_{r}_{t}",
                                    lowBound=0,
                                    upBound=ac_cfg["power_max"])
            for t in range(T) for r in range(3)}

    P_ev = {t: pulp.LpVariable(f"P_ev_{t}",
                               lowBound=0,
                               upBound=ev_cfg["power_max"])
            for t in range(T)}

    z_wm = {t: pulp.LpVariable(f"z_wm_{t}", cat="Binary") for t in range(T)}
    z_dw = {t: pulp.LpVariable(f"z_dw_{t}", cat="Binary") for t in range(T)}

    P_grid = {t: pulp.LpVariable(f"P_grid_{t}", lowBound=0) for t in range(T)}

    # ---------- Constraints ----------
    # Battery SOC
    prob += SOC[0] == battery["soc_init"]
    for t in range(1, T):
        prob += SOC[t] == SOC[t-1] + (
            P_bat[t-1] * battery["eta"] / battery["capacity_kwh"]
        )

    # WM & DW duration
    prob += pulp.lpSum(z_wm[t] for t in range(T)) == wm_cfg["duration"]
    prob += pulp.lpSum(z_dw[t] for t in range(T)) == dw_cfg["duration"]

    # EV energy
    prob += pulp.lpSum(P_ev[t] for t in range(T)) >= ev_cfg["target_energy"]

    # Power balance
    for t in range(T):
        total_load = (
            pulp.lpSum(P_ac[(t, r)] for r in range(3)) +
            P_ev[t] +
            z_wm[t] * wm_cfg["power"] +
            z_dw[t] * dw_cfg["power"]
        )
        prob += P_grid[t] >= total_load + P_bat[t] - pv[t]

    # ---------- Objective ----------
    cost = pulp.lpSum(price[t] * P_grid[t] for t in range(T))
    comfort = pulp.lpSum((ac_cfg["comfort"] - P_ac[(t, r)])**2
                          for t in range(T) for r in range(3))
    prob += cost + 0.05 * comfort

    # ---------- Solve ----------
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    res = prob.solve(solver)
    print("MILP status:", pulp.LpStatus[res])

    # -------------------------------------------------
    # Extract & normalize actions
    # -------------------------------------------------
    actions = np.zeros((T, 7))

    for t in range(T):
        # battery
        actions[t, 0] = P_bat[t].value() / battery["p_charge_max_kw"]

        # AC
        for r in range(3):
            actions[t, 1+r] = P_ac[(t, r)].value() / ac_cfg["power_max"]

        # EV
        actions[t, 4] = P_ev[t].value() / ev_cfg["power_max"]

        # WM / DW
        actions[t, 5] = 1.0 if z_wm[t].value() > 0.5 else -1.0
        actions[t, 6] = 1.0 if z_dw[t].value() > 0.5 else -1.0

    return np.clip(actions, -1.0, 1.0)


# =====================================================
# Example usage
# =====================================================
if __name__ == "__main__":
    T = 24
    price = 0.2 + 0.1 * np.sin(np.linspace(0, 2*np.pi, T))
    pv = np.clip(2.0 * np.sin(np.linspace(0, np.pi, T)), 0, None)

    config = {
        "battery": {
            "capacity_kwh": 10,
            "soc_init": 0.5,
            "soc_min": 0.1,
            "soc_max": 0.9,
            "p_charge_max_kw": 3.0,
            "p_discharge_max_kw": 3.0,
            "eta": 0.95,
        },
        "ac": {
            "power_max": 2.0,
            "comfort": 1.2,
        },
        "ev": {
            "power_max": 3.3,
            "target_energy": 10.0,
        },
        "wm": {
            "power": 0.5,
            "duration": 2,
        },
        "dw": {
            "power": 0.7,
            "duration": 1,
        },
    }

    expert_actions = solve_milp_expert(price, pv, config)
    print("Expert action shape:", expert_actions.shape)
