#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

def closure(df):
    s = df.sum(axis=1).replace(0, np.nan)
    return df.div(s, axis=0).fillna(0.0)

def clr_transform(M, pseudocount=1e-6):
    M = np.asarray(M, dtype=float) + pseudocount
    gm = np.exp(np.mean(np.log(M), axis=1, keepdims=True))
    return np.log(M / gm)


def filter_wide_for_metric(wide_df, metric):
    """For braycurtis, drop samples with zero total abundance (all zeros after pivot).
       For aitchison, return as-is (clr will add pseudocount)."""
    if metric == "braycurtis":
        rs = wide_df.sum(axis=1)
        keep = rs > 0
        return wide_df.loc[keep]
    return wide_df

def distance_matrix(wide_df, metric):
    if metric == "aitchison":
        X = clr_transform(wide_df.values)
        dv = pdist(X, metric="euclidean")
    elif metric == "braycurtis":
        dv = pdist(wide_df.values, metric="braycurtis")
    else:
        raise ValueError("metric must be 'aitchison' or 'braycurtis'")
    return dv, squareform(dv)

def build_pair_map(df_exp):
    meta = df_exp.drop_duplicates(subset=["SampleID","PairID"])[["SampleID","PairID"]]
    pair_map = []
    for pid, sub in meta.groupby("PairID"):
        ids = sub["SampleID"].tolist()
        if len(ids)==2:
            pair_map.append((pid, ids[0], ids[1]))
    return pair_map

def compute_icc_from_D(D, pair_map, sample_ids):
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    V_total = float(np.mean((D[iu]**2)) / 2.0)
    within2 = []
    for pid, a, b in pair_map:
        i = sample_ids.index(a); j = sample_ids.index(b)
        within2.append(D[i, j]**2)
    V_within = float(np.mean(within2)/2.0) if within2 else float('nan')
    V_between = V_total - V_within if not np.isnan(V_within) else float('nan')
    ICC = (V_between / V_total) if V_total>0 and not np.isnan(V_between) else float('nan')
    return ICC, V_total, V_within, V_between

def within_pair_distances(D, pair_map, sample_ids):
    rows = []
    for pid, a, b in pair_map:
        i = sample_ids.index(a); j = sample_ids.index(b)
        d = float(D[i, j])
        rows.append({"PairID": pid, "Sample_A": a, "Sample_B": b, "WithinPairDistance": d})
    return rows

def _pcoa_from_D(D):
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

def _proj(X):
    Q, R = np.linalg.qr(X)
    return Q @ Q.T, np.linalg.matrix_rank(X)

def _adjusted_R2_from_SS(SS_model, SS_total, N, rankX):
    if SS_total <= 0 or N - rankX <= 0:
        return float('nan')
    R2 = SS_model / SS_total
    return 1.0 - (1.0 - R2) * ((N - 1.0) / (N - rankX))

def permanova_pair_only(D, groups, n_perm=999, seed=2025):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    Y, Yc, TSS = _pcoa_from_D(D)
    if Y is None:
        return dict(R2=np.nan, p=np.nan, R2_adj=np.nan, SS_total=np.nan,
                    rank=np.nan, df_res=np.nan)
    pairs = pd.Categorical(groups)
    X_int = np.ones((n, 1))
    X_pair = pd.get_dummies(pairs, drop_first=True).values
    XA = np.hstack([X_int, X_pair]) if X_pair.size>0 else X_int
    HA, rankA = _proj(XA)
    SS_A = float(np.trace(Y.T @ HA @ Y))
    R2_obs = SS_A / TSS if TSS>0 else np.nan
    R2_adj = _adjusted_R2_from_SS(SS_A, TSS, n, rankA)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        pairs_p = pairs[perm]
        Xp = np.hstack([X_int, pd.get_dummies(pairs_p, drop_first=True).values]) if X_pair.size>0 else X_int
        Hp, _ = _proj(Xp)
        SS_p = float(np.trace(Y.T @ Hp @ Y))
        if SS_p >= SS_A - 1e-12:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return dict(R2=R2_obs, p=pval, R2_adj=R2_adj, SS_total=TSS,
                rank=int(rankA), df_res=int(n-rankA))

def permanova_pair_env(D, pair_labels, env_labels, n_perm=999, seed=2025):
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    Y, Yc, SS_tot = _pcoa_from_D(D)
    if Y is None:
        return dict(R2_Pair=np.nan, p_Pair=np.nan, R2_adj_Pair=np.nan,
                    R2_Env_given_Pair=np.nan, p_Env_given_Pair=np.nan,
                    R2_adj_Env_given_Pair=np.nan, SS_total=np.nan,
                    rank_A=np.nan, rank_AB=np.nan, df_res_A=np.nan, df_res_AB=np.nan)
    pair = pd.Categorical(pair_labels)
    env  = pd.Categorical(env_labels)
    X_int = np.ones((n, 1))
    X_pair = pd.get_dummies(pair, drop_first=True).values
    X_env  = pd.get_dummies(env,  drop_first=True).values
    XA  = np.hstack([X_int, X_pair]) if X_pair.size>0 else X_int
    XAB = np.hstack([XA, X_env])    if X_env.size>0  else XA
    HA,  rankA  = _proj(XA)
    HAB, rankAB = _proj(XAB)
    SS_A  = float(np.trace(Y.T @ HA  @ Y))
    SS_AB = float(np.trace(Y.T @ HAB @ Y))
    SS_BA = SS_AB - SS_A
    R2_A  = SS_A  / SS_tot if SS_tot>0 else np.nan
    R2_BA = SS_BA / SS_tot if SS_tot>0 else np.nan
    R2_adj_A  = _adjusted_R2_from_SS(SS_A,  SS_tot, n, rankA)
    R2_adj_AB = _adjusted_R2_from_SS(SS_AB, SS_tot, n, rankAB)
    RSS_A  = SS_tot - SS_A
    RSS_AB = SS_tot - SS_AB
    MS_res_A  = RSS_A  / (n - rankA)
    MS_res_AB = RSS_AB / (n - rankAB)
    R2_adj_BA = 1.0 - (MS_res_AB / MS_res_A) if (n-rankA)>0 and (n-rankAB)>0 else np.nan
    count_A = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        HpA, _ = _proj(np.hstack([X_int, pd.get_dummies(pair[perm], drop_first=True).values]) if X_pair.size>0 else X_int)
        SS_A_p = float(np.trace(Y.T @ HpA @ Y))
        if SS_A_p >= SS_A - 1e-12:
            count_A += 1
    p_A = (count_A + 1) / (n_perm + 1)
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
    return dict(R2_Pair=float(R2_A), p_Pair=float(p_A), R2_adj_Pair=float(R2_adj_A),
                R2_Env_given_Pair=float(R2_BA), p_Env_given_Pair=float(p_BA),
                R2_adj_Env_given_Pair=float(R2_adj_BA), SS_total=float(SS_tot),
                rank_A=int(rankA), rank_AB=int(rankAB),
                df_res_A=int(n - rankA), df_res_AB=int(n - rankAB))

def perm_test_mean_diff(vals_exp2, vals_exp1, B=20000, seed=2025):
    rng = np.random.default_rng(seed)
    y = np.r_[vals_exp1, vals_exp2]
    labels = np.r_[np.zeros(len(vals_exp1)), np.ones(len(vals_exp2))]
    obs = y[labels==1].mean() - y[labels==0].mean()
    count = 0
    for _ in range(B):
        rng.shuffle(labels)
        diff = y[labels==1].mean() - y[labels==0].mean()
        if abs(diff) >= abs(obs):
            count += 1
    p = (count + 1) / (B + 1)
    return float(obs), float(p)

def main():
    infile = "combined_long_format.csv"
    if not os.path.exists(infile):
        raise FileNotFoundError("combined_long_format.csv not found in /mnt/data")
    df = pd.read_csv(infile)

    summary_rows, within_rows, perexp_rows, combined_rows = [], [], [], []

    for metric in ["braycurtis", "aitchison"]:
        exp_within = {}
        for exp_id in [1, 2]:
            df_exp = df[df["Experiment"]==exp_id]
            wide = df_exp.pivot(index="SampleID", columns="Taxonomy", values="Proportion").fillna(0.0)
            wide = filter_wide_for_metric(wide, metric)
            wide = closure(wide)
            sample_ids = list(wide.index)
            pair_map = build_pair_map(df_exp)
            sample_ids = list(wide.index)
            pair_map = [t for t in pair_map if (t[1] in sample_ids and t[2] in sample_ids)]
            dv, D = distance_matrix(wide, metric)

            ICC, Vt, Vw, Vb = compute_icc_from_D(D, pair_map, sample_ids)
            rows = within_pair_distances(D, pair_map, sample_ids)
            for r in rows:
                r.update({"Experiment": exp_id, "Distance": metric})
            within_rows.extend(rows)
            wdist = np.array([r["WithinPairDistance"] for r in rows])
            exp_within[exp_id] = wdist

            meta_pairs = df_exp.drop_duplicates(subset=["SampleID","PairID"]).set_index("SampleID")["PairID"].loc[sample_ids].values
            out = permanova_pair_only(D, meta_pairs, n_perm=999, seed=2025)
            perexp_rows.append({"Experiment": exp_id, "Distance": metric,
                                "R2_Pair": out["R2"], "R2_adj_Pair": out["R2_adj"],
                                "p_Pair": out["p"], "SS_total": out["SS_total"],
                                "rank": out["rank"], "df_res": out["df_res"]})

            summary_rows.append({"Experiment": exp_id, "Distance": metric, "n_pairs": len(pair_map),
                                 "ICC": ICC, "V_total": Vt, "V_within": Vw, "V_between": Vb,
                                 "within_mean": float(np.mean(wdist)) if wdist.size else np.nan,
                                 "within_sd": float(np.std(wdist, ddof=1)) if wdist.size>1 else np.nan,
                                 "within_median": float(np.median(wdist)) if wdist.size else np.nan,
                                 "within_min": float(np.min(wdist)) if wdist.size else np.nan,
                                 "within_max": float(np.max(wdist)) if wdist.size else np.nan,
                                 "PERM_R2_Pair": out["R2"], "PERM_R2_adj_Pair": out["R2_adj"], "PERM_p_Pair": out["p"]})

        if 1 in exp_within and 2 in exp_within and exp_within[1].size and exp_within[2].size:
            diff, p = perm_test_mean_diff(exp_within[2], exp_within[1], B=20000, seed=2025)
            summary_rows.append({"Experiment": 0, "Distance": metric, "Contrast": "Exp2-Exp1_mean_within_pair",
                                 "Diff": diff, "perm_p_two_sided": p,
                                 "n_pairs_Exp1": int(exp_within[1].size), "n_pairs_Exp2": int(exp_within[2].size)})

    wide_all = df.pivot(index="SampleID", columns="Taxonomy", values="Proportion").fillna(0.0)
    # For braycurtis in combined, drop zero-sum samples
    wide_all_bc = filter_wide_for_metric(wide_all.copy(), "braycurtis")
    wide_all = closure(wide_all)
    meta_all = df.drop_duplicates(subset=["SampleID"])[["SampleID","PairID","Environment"]].set_index("SampleID").loc[wide_all.index]

    for metric in ["braycurtis", "aitchison"]:
        if metric == "braycurtis":
            _, D_all = distance_matrix(closure(wide_all_bc), metric)
            meta_index = wide_all_bc.index
        else:
            _, D_all = distance_matrix(wide_all, metric)
            meta_index = wide_all.index
        meta_sub = meta_all.loc[meta_index]
        out = permanova_pair_env(D_all, meta_sub["PairID"].values, meta_sub["Environment"].values,
                                 n_perm=999, seed=2025)
        r = {"Distance": metric}; r.update(out); combined_rows.append(r)

    pd.DataFrame(summary_rows).to_csv("results_summary.csv", index=False)
    pd.DataFrame(within_rows).to_csv("within_pair_distances.csv", index=False)
    pd.DataFrame(perexp_rows).to_csv("er_experiment_permanova.csv", index=False)
    pd.DataFrame(combined_rows).to_csv("combined_permanova_pair_env.csv", index=False)

if __name__ == "__main__":
    main()
