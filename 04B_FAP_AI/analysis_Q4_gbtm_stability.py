"""
Q4 (revision): stability of the K=3 joint-GBTM class assignment across random seeds.

Replicates the exact joint-GBTM construction of 02_mdap_gbtm.py:
  - trajectory-eligible subset (>=4/5 core markers complete in both 6-24/24-48 h windows),
  - sparse within-eligible imputation (other window -> baseline -> cohort median),
  - per-marker StandardScaler on the 5 core markers x 2 windows (10 features),
  - GaussianMixture(n_components=3, n_init=50, max_iter=1000, covariance_type='full',
    reg_covar=1e-5), the same estimator the main analysis uses.

Then refits K=3 across many random seeds and quantifies assignment stability with the
Adjusted Rand Index (ARI, invariant to label switching) against the reference seed (42),
plus per-patient modal-class agreement. Outputs outputs/revision_Q4_gbtm_stability.csv.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
df = pd.read_csv(os.path.join(OUT, "canonical_mdap_cohort.csv"))
TRAJ = ["wbc", "creatinine", "bun", "platelet", "glucose"]
WIN = ["w6_24", "w24_48"]
MIN_CORE_COMPLETE = 4

def complete_count(row):
    return sum(all(pd.notna(row[f"{m}_{w}"]) for w in WIN) for m in TRAJ)

df["traj_complete_count"] = df.apply(complete_count, axis=1)
df_traj = df[df["traj_complete_count"] >= MIN_CORE_COMPLETE].copy().reset_index(drop=True)
print(f"Trajectory-eligible: {len(df_traj)}/{len(df)}")

# sparse within-eligible imputation: other window -> baseline -> cohort median
for m in TRAJ:
    for w in WIN:
        col = f"{m}_{w}"; mask = df_traj[col].isna()
        for idx in df_traj[mask].index:
            ow = [x for x in WIN if x != w][0]
            ov = df_traj.loc[idx, f"{m}_{ow}"]
            bc = f"baseline_{m}"
            if pd.notna(ov):
                df_traj.loc[idx, col] = ov
            elif bc in df_traj.columns and pd.notna(df_traj.loc[idx, bc]):
                df_traj.loc[idx, col] = df_traj.loc[idx, bc]
            else:
                df_traj.loc[idx, col] = df_traj[col].median()

# per-marker standardization -> 10-feature joint matrix
Y = np.zeros((len(df_traj), len(TRAJ) * len(WIN)))
for i, m in enumerate(TRAJ):
    raw = df_traj[[f"{m}_{w}" for w in WIN]].values.astype(float)
    Y[:, i * 2:(i + 1) * 2] = StandardScaler().fit_transform(raw)
mask = np.all(np.isfinite(Y), axis=1)
Yj = Y[mask]
print(f"Joint matrix: {Yj.shape[0]} complete cases x {Yj.shape[1]} features")

def fit_labels(seed):
    gmm = GaussianMixture(n_components=3, n_init=50, max_iter=1000, random_state=seed,
                          covariance_type="full", reg_covar=1e-5)
    return gmm.fit_predict(Yj)

ref = fit_labels(42)
seeds = [42, 0, 1, 2, 7, 13, 21, 99, 123, 2024, 31415]
labels = {s: fit_labels(s) for s in seeds}

# pairwise ARI across all seed pairs
aris = []
for i, a in enumerate(seeds):
    for b in seeds[i + 1:]:
        aris.append(adjusted_rand_score(labels[a], labels[b]))
aris = np.array(aris)

# per-patient modal-class agreement: align every seed to the reference by best label map,
# then fraction of patients whose modal assignment matches the reference
def align_to_ref(lab, ref):
    # map each cluster id in lab to ref id by majority overlap
    mapping = {}
    for c in np.unique(lab):
        ref_ids, counts = np.unique(ref[lab == c], return_counts=True)
        mapping[c] = ref_ids[np.argmax(counts)]
    return np.array([mapping[x] for x in lab])

aligned = np.vstack([align_to_ref(labels[s], ref) for s in seeds])
modal = np.array([np.bincount(aligned[:, j]).argmax() for j in range(aligned.shape[1])])
agree = (aligned == modal[np.newaxis, :]).mean()  # mean over seeds x patients
ref_match = (ref == modal).mean()

print(f"\nPairwise ARI across {len(seeds)} seeds ({len(aris)} pairs): "
      f"mean {aris.mean():.3f}  min {aris.min():.3f}  max {aris.max():.3f}")
print(f"Per-patient modal-class agreement: {agree*100:.1f}%  "
      f"(reference seed matches modal in {ref_match*100:.1f}% of patients)")
# class size stability
sizes = np.vstack([np.bincount(align_to_ref(labels[s], ref), minlength=3) for s in seeds])
print("Class sizes per seed (aligned):"); print(sizes)

pd.DataFrame([dict(n_seeds=len(seeds), n_pairs=len(aris),
                   ari_mean=round(aris.mean(), 3), ari_min=round(aris.min(), 3),
                   ari_max=round(aris.max(), 3),
                   modal_agreement_pct=round(agree * 100, 1),
                   class_size_mean=str(sizes.mean(0).round(0).astype(int).tolist()),
                   class_size_range=str([f"{sizes[:,k].min()}-{sizes[:,k].max()}" for k in range(3)]))]
             ).to_csv(os.path.join(OUT, "revision_Q4_gbtm_stability.csv"), index=False)
print("\nSaved -> revision_Q4_gbtm_stability.csv")
