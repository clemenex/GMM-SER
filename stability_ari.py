import numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
df = pd.read_csv("data/daic_prosodic_summary_60s.csv")
X = df[["pitch_sd","energy_sd"]].dropna().to_numpy()
Xz = StandardScaler().fit_transform(X)

labels = []
seeds = [0,1,2,3,4]
for rs in seeds:
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=rs, n_init=5, reg_covar=1e-6).fit(Xz)
    labels.append(gmm.predict(Xz))

for i in range(len(seeds)):
    for j in range(i+1, len(seeds)):
        ari = adjusted_rand_score(labels[i], labels[j])
        print(f"seed {seeds[i]} vs {seeds[j]}: ARI={ari:.4f}")
