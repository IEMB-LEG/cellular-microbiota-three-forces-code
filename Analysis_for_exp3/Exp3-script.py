#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 25 19:36:48 2025

@author: zhan1530
"""

import pandas as pd
import numpy as np
import itertools
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# ----------------------- Config -----------------------
INFILE = "Exp3-data.csv"  # put your csv in the same folder or change to absolute path
ENABLE_AITCHISON = False               # set True to also compute Aitchison outputs
SEED = 2025
B_PERM = 20000                         # permutations for pairwise tests
N_PERM_PERMANOVA = 999                 # permutations for PERMANOVA/dispersion

# ----------------------- Helpers ----------------------
def closure(df: pd.DataFrame) -> pd.DataFrame:
    s = df.sum(axis=1).replace(0, np.nan)
    return df.div(s, axis=0).fillna(0.0)

def bray_curtis_dm(X: pd.DataFrame):
    dv = pdist(X.values, metric="braycurtis")
    return squareform(dv)

def aitchison_dm(X: pd.DataFrame, pseudocount=1e-6):
    Xp = np.asarray(X.values, dtype=float) + pseudocount
    gm = np.exp(np.mean(np.log(Xp), axis=1, keepdims=True))
    Z = np.log(Xp / gm)
    dv = pdist(Z, metric="euclidean")
    return squareform(dv)

def phase_from_day(d: int) -> str:
    d = int(d)
    if 1 <= d <= 4:  return "Growth"
    if 5 <= d <= 8:  return "Plateau"
    if 9 <= d <= 12: return "Decline"
    return "Other"

def perm_test_mean_diff(a, b, B=B_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    obs = float(a.mean() - b.mean())
    y = np.concatenate([a, b]).copy()
    n_a = len(a)
    count = 0
    for _ in range(B):
        rng.shuffle(y)
        diff = y[:n_a].mean() - y[n_a:].mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (B + 1)
    return obs, float(p)

def cohens_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    s_pooled = np.sqrt(((a.size - 1)*a.var(ddof=1) + (b.size - 1)*b.var(ddof=1)) / (a.size + b.size - 2))
    if s_pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s_pooled)

def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    # O(n log n) via ranks
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    i = j = greater = less = 0
    na, nb = len(a_sorted), len(b_sorted)
    while i < na and j < nb:
        if a_sorted[i] > b_sorted[j]:
            greater += (na - i); j += 1
        elif a_sorted[i] < b_sorted[j]:
            less += (nb - j); i += 1
        else:
            i += 1; j += 1
    return float((greater - less) / (na * nb))

def one_way_eta_squared(values_by_group: dict):
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in values_by_group.values() if len(v)>0], axis=0)
    if all_vals.size == 0:
        return np.nan
    grand_mean = all_vals.mean()
    ss_between = 0.0
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    for g, vals in values_by_group.items():
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0: 
            continue
        ss_between += len(vals) * (vals.mean() - grand_mean) ** 2
    if ss_total == 0:
        return np.nan
    return float(ss_between / ss_total)

def pcoa_from_D(D):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D**2).dot(H)
    evals, evecs = np.linalg.eigh(B)
    pos = evals > 1e-10
    if not np.any(pos):
        return None, None, None
    Y = evecs[:, pos] * np.sqrt(evals[pos])
    Yc = Y - Y.mean(axis=0, keepdims=True)
    SS_total = float((Yc**2).sum())
    return Y, Yc, SS_total

def proj(X):
    Q, R = np.linalg.qr(X)
    return Q @ Q.T, np.linalg.matrix_rank(X)

def adjusted_R2(SS_model, SS_total, N, rankX):
    if SS_total <= 0 or N - rankX <= 0:
        return float("nan")
    R2 = SS_model / SS_total
    return 1.0 - (1.0 - R2) * ((N - 1.0) / (N - rankX))

def permanova_two_factor(D, factor_A, factor_B, n_perm=N_PERM_PERMANOVA, seed=SEED):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    Y, Yc, SS_tot = pcoa_from_D(D)
    if Y is None:
        return None

    A = pd.Categorical(factor_A)
    B = pd.Categorical(factor_B)

    X_int = np.ones((n, 1))
    X_A = pd.get_dummies(A, drop_first=True).values
    X_B = pd.get_dummies(B, drop_first=True).values

    XA  = np.hstack([X_int, X_A]) if X_A.size>0 else X_int
    XAB = np.hstack([XA, X_B])    if X_B.size>0 else XA

    HA,  rankA  = proj(XA)
    HAB, rankAB = proj(XAB)

    SS_A  = float(np.trace(Y.T @ HA  @ Y))
    SS_AB = float(np.trace(Y.T @ HAB @ Y))
    SS_BA = SS_AB - SS_A

    R2_A  = SS_A  / SS_tot if SS_tot>0 else np.nan
    R2_BA = SS_BA / SS_tot if SS_tot>0 else np.nan

    R2_adj_A  = adjusted_R2(SS_A,  SS_tot, n, rankA)
    R2_adj_AB = adjusted_R2(SS_AB, SS_tot, n, rankAB)
    RSS_A  = SS_tot - SS_A
    RSS_AB = SS_tot - SS_AB
    MS_res_A  = RSS_A  / (n - rankA)
    MS_res_AB = RSS_AB / (n - rankAB)
    R2_adj_BA = 1.0 - (MS_res_AB / MS_res_A) if (n-rankA)>0 and (n-rankAB)>0 else np.nan

    # Permutations for A
    count_A = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        XpA = np.hstack([X_int, pd.get_dummies(A[perm], drop_first=True).values]) if X_A.size>0 else X_int
        HpA, _ = proj(XpA)
        SS_A_p = float(np.trace(Y.T @ HpA @ Y))
        if SS_A_p >= SS_A - 1e-12:
            count_A += 1
    p_A = (count_A + 1) / (n_perm + 1)

    # Freedman–Lane for B|A
    Yhat_A = HA @ Y
    E_A = Y - Yhat_A
    count_BA = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Y_perm = Yhat_A + E_A[perm, :]
        SS_A_perm  = float(np.trace(Y_perm.T @ HA  @ Y_perm))
        SS_AB_perm = float(np.trace(Y_perm.T @ HAB @ Y_perm))
        SS_BA_perm = SS_AB_perm - SS_A_perm
        if SS_BA_perm >= SS_BA - 1e-12:
            count_BA += 1
    p_BA = (count_BA + 1) / (n_perm + 1)

    return {
        "R2_A": float(R2_A), "R2_adj_A": float(R2_adj_A), "p_A": float(p_A),
        "R2_B|A": float(R2_BA), "R2_adj_B|A": float(R2_adj_BA), "p_B|A": float(p_BA),
        "SS_total": float(SS_tot), "rank_A": int(rankA), "rank_AB": int(rankAB),
        "df_res_A": int(n - rankA), "df_res_AB": int(n - rankAB)
    }
# -------- One-factor PERMANOVA with optional strata (block permutations) --------
def permanova_one_factor_with_strata(D, factor, strata=None, n_perm=N_PERM_PERMANOVA, seed=SEED):
    """
    One-factor distance-based PERMANOVA:
        distance ~ factor
    If 'strata' is provided, labels of 'factor' are permuted *within* each stratum (block permutations).

    Returns: dict with R2, R2_adj, p_value, SS_total, rankX, df_res.
    """
    rng = np.random.default_rng(seed)
    n = D.shape[0]

    # PCoA (same pipeline as elsewhere to obtain Euclidean Y and total inertia)
    Y, Yc, SS_total = pcoa_from_D(D)
    if Y is None:
        raise RuntimeError("PCoA failed in permanova_one_factor_with_strata")

    # Design matrix: intercept + factor dummies
    F = pd.Categorical(factor)
    X_int = np.ones((n, 1))
    X_F   = pd.get_dummies(F, drop_first=True).values
    X     = np.hstack([X_int, X_F]) if X_F.size > 0 else X_int

    # Projection & sums of squares
    H, rankX = proj(X)
    SS_model = float(np.trace(Y.T @ H @ Y))
    R2       = SS_model / SS_total if SS_total > 0 else np.nan
    R2_adj   = adjusted_R2(SS_model, SS_total, n, rankX)
    df_res   = int(n - rankX)

    # --- Permutations (block-wise if strata is given) ---
    # 构造 strata -> index 列表
    if strata is not None:
        strata = pd.Categorical(strata)
        blocks = [np.where(strata.codes == k)[0] for k in range(strata.categories.size)]
    else:
        blocks = [np.arange(n)]

    count = 0
    for _ in range(n_perm):
        # 仅在各 block 内打乱 factor 标签
        perm_codes = F.codes.copy()
        for idx in blocks:
            rng.shuffle(perm_codes[idx])

        Xp = np.hstack([X_int, pd.get_dummies(pd.Categorical.from_codes(perm_codes, F.categories),
                                             drop_first=True).values]) if X_F.size > 0 else X_int
        Hp, _ = proj(Xp)
        SS_p = float(np.trace(Y.T @ Hp @ Y))
        if SS_p >= SS_model - 1e-12:
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    return {
        "R2": float(R2),
        "R2_adj": float(R2_adj),
        "p_value": float(p_value),
        "SS_total": float(SS_total),
        "rankX": int(rankX),
        "df_res": int(df_res)
    }

# --- Interaction PERMANOVA (Bottle + Phase + Bottle:Phase) ---
def pcoa_for_perm(D):
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H.dot(D**2).dot(H)
    evals, evecs = np.linalg.eigh(B)
    pos = evals > 1e-10
    if not np.any(pos):
        raise RuntimeError("PCoA failed")
    return evecs[:, pos] * np.sqrt(evals[pos])

def proj_mat(X):
    Q, _ = np.linalg.qr(X)
    return Q @ Q.T, np.linalg.matrix_rank(X)

def permanova_with_interaction(D, bottle, phase, n_perm=N_PERM_PERMANOVA, seed=SEED):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    Y = pcoa_for_perm(D)
    I = np.eye(n)

    X0 = np.ones((n,1))
    B = pd.get_dummies(pd.Categorical(bottle), drop_first=True).values
    P = pd.get_dummies(pd.Categorical(phase),  drop_first=True).values
    BP = np.hstack([B[:,[i]] * P[:,[j]] for i in range(B.shape[1]) for j in range(P.shape[1])]) if (B.size and P.size) else np.zeros((n,0))

    H0,_ = proj_mat(X0)
    HB,_ = proj_mat(np.hstack([X0,B]) if B.size else X0)
    HBP,_= proj_mat(np.hstack([X0,B,P]) if (B.size or P.size) else X0)
    HBPi,_=proj_mat(np.hstack([X0,B,P,BP]) if (B.size or P.size or BP.size) else X0)

    SS_tot = float(np.trace(Y.T @ (I - H0)  @ Y))
    SS_B   = float(np.trace(Y.T @ (HB - H0)  @ Y))
    SS_PB  = float(np.trace(Y.T @ (HBP - HB) @ Y))     # Phase | Bottle
    SS_Int = float(np.trace(Y.T @ (HBPi - HBP) @ Y))   # Interaction | Bottle+Phase

    R2_B   = SS_B   / SS_tot
    R2_PB  = SS_PB  / SS_tot
    R2_Int = SS_Int / SS_tot

    # permutations
    cnt_B = cnt_PB = cnt_Int = 0
    # Bottle: permute labels
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Bp = pd.get_dummies(pd.Categorical(np.asarray(bottle)[perm]), drop_first=True).values if B.size else np.zeros((n,0))
        HBp,_ = proj_mat(np.hstack([X0,Bp]) if B.size else X0)
        SS_B_p = float(np.trace(Y.T @ (HBp - H0) @ Y))
        cnt_B += (SS_B_p >= SS_B - 1e-12)
    p_B = (cnt_B + 1) / (n_perm + 1)

    # Phase | Bottle: Freedman–Lane
    Yhat_B = HB @ Y
    E_B = Y - Yhat_B
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Yp = Yhat_B + E_B[perm,:]
        SS_PB_p = float(np.trace(Yp.T @ (HBP - HB) @ Yp))
        cnt_PB += (SS_PB_p >= SS_PB - 1e-12)
    p_PB = (cnt_PB + 1) / (n_perm + 1)

    # Interaction | Bottle+Phase: Freedman–Lane
    Yhat_BP = HBP @ Y
    E_BP = Y - Yhat_BP
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Yp2 = Yhat_BP + E_BP[perm,:]
        SS_Int_p = float(np.trace(Yp2.T @ (HBPi - HBP) @ Yp2))
        cnt_Int += (SS_Int_p >= SS_Int - 1e-12)
    p_Int = (cnt_Int + 1) / (n_perm + 1)

    return {"R2_Bottle":R2_B, "p_Bottle":p_B,
            "R2_Phase|Bottle":R2_PB, "p_Phase|Bottle":p_PB,
            "R2_Interaction|BP":R2_Int, "p_Interaction|BP":p_Int}

# ----------------------- Load data --------------------
def load_and_prepare(infile=INFILE):
    long_df = pd.read_csv(infile)
    long_c = long_df[long_df["Sample"].astype(str).str.startswith("C")].copy()
    wide = long_c.pivot(index="Sample", columns="Taxonomy", values="Proportion").fillna(0.0)
    wide = closure(wide)
    meta = long_c[["Sample","Day","Bottle"]].drop_duplicates().rename(columns={"Sample":"SampleID"})
    meta["Phase"] = meta["Day"].apply(phase_from_day)
    wide = wide.loc[meta["SampleID"]]  # align
    return wide, meta

# ---------------------- Main analysis -----------------
def run_all(infile=INFILE, enable_aitchison=ENABLE_AITCHISON):
    wide, meta = load_and_prepare(infile)
    dm_bc = bray_curtis_dm(wide)
    dm_dict = {"braycurtis": dm_bc}
    if enable_aitchison:
        dm_dict["aitchison"] = aitchison_dm(wide)

    summary_all = {}; tests_all = {}; permanova_all = {}; effects_all = {}

    for metric, D in dm_dict.items():
        phases = ["Growth", "Plateau", "Decline"]
        sample_idx = {sid:i for i, sid in enumerate(wide.index)}
        within = {ph: [] for ph in phases}
        between = {ph: [] for ph in phases}

        # within & between
        for ph in phases:
            meta_ph = meta[meta["Phase"] == ph]
            # within
            for b in sorted(meta_ph["Bottle"].unique()):
                ids = meta_ph.loc[meta_ph["Bottle"] == b, "SampleID"].tolist()
                idx = [sample_idx[i] for i in ids if i in sample_idx]
                for i, j in itertools.combinations(idx, 2):
                    within[ph].append(D[i, j])
            # between
            bottles = sorted(meta_ph["Bottle"].unique())
            for b1, b2 in itertools.combinations(bottles, 2):
                ids1 = meta_ph.loc[meta_ph["Bottle"] == b1, "SampleID"].tolist()
                ids2 = meta_ph.loc[meta_ph["Bottle"] == b2, "SampleID"].tolist()
                idx1 = [sample_idx[i] for i in ids1 if i in sample_idx]
                idx2 = [sample_idx[i] for i in ids2 if i in sample_idx]
                for i in idx1:
                    for j in idx2:
                        between[ph].append(D[i, j])

        # summaries
        summary_rows = []
        for ph in phases:
            wvals = np.array(within[ph], dtype=float)
            bvals = np.array(between[ph], dtype=float)
            summary_rows.append({
                "Phase": ph,
                "Within_n_pairs": int(wvals.size),
                "Within_mean": float(wvals.mean()) if wvals.size else np.nan,
                "Within_sd": float(wvals.std(ddof=1)) if wvals.size>1 else np.nan,
                "Between_n_pairs": int(bvals.size),
                "Between_mean": float(bvals.mean()) if bvals.size else np.nan,
                "Between_sd": float(bvals.std(ddof=1)) if bvals.size>1 else np.nan,
            })
        summary_df = pd.DataFrame(summary_rows); summary_all[metric] = summary_df

        # pairwise permutation tests + effect sizes
        comps = [("Plateau", "Growth"), ("Decline", "Plateau"), ("Decline", "Growth")]
        test_rows = []
        for a, b in comps:
            obs_w, p_w = perm_test_mean_diff(within[a], within[b], B=B_PERM, seed=SEED)
            d_w = cohens_d(np.array(within[a]), np.array(within[b]))
            delta_w = cliffs_delta(np.array(within[a]), np.array(within[b]))
            obs_b, p_b = perm_test_mean_diff(between[a], between[b], B=B_PERM, seed=SEED)
            d_b = cohens_d(np.array(between[a]), np.array(between[b]))
            delta_b = cliffs_delta(np.array(between[a]), np.array(between[b]))
            test_rows.append({
                "Comparison": f"{a} - {b}",
                "Within_diff_mean": obs_w, "Within_perm_p_two_sided": p_w,
                "Within_Cohens_d": d_w, "Within_Cliffs_delta": delta_w,
                "Between_diff_mean": obs_b, "Between_perm_p_two_sided": p_b,
                "Between_Cohens_d": d_b, "Between_Cliffs_delta": delta_b,
                "Within_n_a": len(within[a]), "Within_n_b": len(within[b]),
                "Between_n_a": len(between[a]), "Between_n_b": len(between[b]),
            })
        tests_df = pd.DataFrame(test_rows); tests_all[metric] = tests_df

        # one-way eta-squared (Phase)
        eff_df = pd.DataFrame([{
            "Metric": metric,
            "Eta_squared_within": one_way_eta_squared(within),
            "Eta_squared_between": one_way_eta_squared(between)
        }])
        effects_all[metric] = eff_df

        # PERMANOVA: Phase/Bottle sequential
        res_phase_first = permanova_two_factor(D, factor_A=meta["Phase"], factor_B=meta["Bottle"], n_perm=N_PERM_PERMANOVA, seed=SEED)
        res_bottle_first = permanova_two_factor(D, factor_A=meta["Bottle"], factor_B=meta["Phase"], n_perm=N_PERM_PERMANOVA, seed=SEED)
        rows = []
        rows.append({"Model":"Phase (main)", "R2":res_phase_first["R2_A"], "R2_adj":res_phase_first["R2_adj_A"], "p_value":res_phase_first["p_A"], "Entered_first":"Phase"})
        rows.append({"Model":"Bottle | Phase", "R2":res_phase_first["R2_B|A"], "R2_adj":res_phase_first["R2_adj_B|A"], "p_value":res_phase_first["p_B|A"], "Entered_first":"Phase"})
        rows.append({"Model":"Bottle (main)", "R2":res_bottle_first["R2_A"], "R2_adj":res_bottle_first["R2_adj_A"], "p_value":res_bottle_first["p_A"], "Entered_first":"Bottle"})
        rows.append({"Model":"Phase | Bottle", "R2":res_bottle_first["R2_B|A"], "R2_adj":res_bottle_first["R2_adj_B|A"], "p_value":res_bottle_first["p_B|A"], "Entered_first":"Bottle"})
        permanova_df = pd.DataFrame(rows); 
        permanova_all[metric] = permanova_df
# -------- PERMANOVA (one-factor) with strata ----------
   
    # (A) Phase main effect with Bottle as strata（重复测量：同瓶内打乱 Phase）
    res_phase_strata = permanova_one_factor_with_strata(
        dm_bc,
        factor=meta["Phase"],
        strata=meta["Bottle"],
        n_perm=N_PERM_PERMANOVA,
        seed=SEED
    )

    pd.DataFrame([
        {"Model":"Phase (stratified by Bottle)", **res_phase_strata},
    ]).to_csv("permanova_strata.csv", index=False)
    # --- Interaction PERMANOVA and figure ---
    res_inter = permanova_with_interaction(dm_bc, meta["Bottle"].values, meta["Phase"].values, n_perm=N_PERM_PERMANOVA, seed=SEED)
    inter_df = pd.DataFrame([res_inter])
    inter_df.to_csv("permanova_with_interaction.csv", index=False)

    # Save core outputs
    summary_all["braycurtis"].to_csv("phase_distance_summary.csv", index=False)
    tests_all["braycurtis"].to_csv("phase_permutation_tests.csv", index=False)
    permanova_all["braycurtis"].to_csv("permanova_phase_bottle.csv", index=False)
    effects_all["braycurtis"].to_csv("effect_sizes_phase.csv", index=False)

    # R2 bar figure (interaction)
    plt.figure(figsize=(6,4))
    labels = ["Bottle", "Phase|Bottle", "Bottle:Phase|BP"]
    vals = [res_inter["R2_Bottle"], res_inter["R2_Phase|Bottle"], res_inter["R2_Interaction|BP"]]
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=10)
    plt.ylabel("R²")
    plt.title("PERMANOVA with interaction")
    plt.tight_layout()
    plt.savefig("fig_permanova_interaction_R2.png", dpi=300)
    plt.close()

    print("Saved outputs:")
    for f in ["phase_distance_summary.csv","phase_permutation_tests.csv",
              "permanova_phase_bottle.csv","effect_sizes_phase.csv",
              "permanova_with_interaction.csv","fig_permanova_interaction_R2.png"]:
        print(" -", f)

if __name__ == "__main__":
    run_all(INFILE, ENABLE_AITCHISON)
