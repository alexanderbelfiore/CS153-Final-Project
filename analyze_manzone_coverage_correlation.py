"""
Analyze the correlation between defense_man_zone_type and defense_coverage_type.

Loads the persisted cleaned dataset (data/pbp_3rd4th_2024_2025.csv) so the
notebook does not need to be re-run.  Outputs:

  1. Cross-tabulation (raw counts and row-normalised %)
  2. Cramér's V  — standard categorical association measure in [0, 1]
  3. Mutual information and conditional entropy H(man_zone | coverage)
  4. Coverage types that map to MORE than one man/zone value (the only
     rows where man_zone_type adds information)
  5. Candidate-space impact: how the 26-tuple action set collapses when
     defense_man_zone_type is dropped
  6. Verdict: is it worth removing defense_man_zone_type from action features?
"""

import pathlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ── load data ─────────────────────────────────────────────────────────────────
DATA = pathlib.Path(__file__).parent / "data" / "pbp_3rd4th_2024_2025.csv"
df = pd.read_csv(DATA, low_memory=False)

MZ = "defense_man_zone_type"
CT = "defense_coverage_type"

# Keep only rows with both labels present (same filter as rich_set in notebook)
df = df[df[MZ].isin(["MAN_COVERAGE", "ZONE_COVERAGE"]) & df[CT].notna()].copy()
print(f"Rows analysed: {len(df):,}\n")

# ── 1. Cross-tabulation ───────────────────────────────────────────────────────
print("=" * 70)
print("1. Cross-tabulation: coverage_type (rows) × man_zone_type (cols)")
print("=" * 70)
ct_raw = pd.crosstab(df[CT], df[MZ])
ct_pct = pd.crosstab(df[CT], df[MZ], normalize="index").mul(100).round(1)

print("\nRaw counts:")
print(ct_raw.to_string())
print("\nRow-normalised % (how often each coverage type is man vs zone):")
print(ct_pct.to_string())

# ── 2. Cramér's V ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. Cramer's V  (categorical correlation, range 0 = none to 1 = perfect)")
print("=" * 70)
chi2, p, dof, _ = chi2_contingency(ct_raw)
n = ct_raw.values.sum()
k = min(ct_raw.shape)  # min(n_rows, n_cols)
cramers_v = np.sqrt(chi2 / (n * (k - 1)))
print(f"\n  chi2={chi2:.2f}  dof={dof}  p={p:.2e}")
print(f"  Cramér's V = {cramers_v:.4f}")
thresholds = [(0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"),
              (0.7, "strong"), (1.01, "very strong / deterministic")]
label = next(lbl for upper, lbl in thresholds if cramers_v < upper)
print(f"  Interpretation: {label}")

# ── 3. Mutual information and conditional entropy ────────────────────────────
print("\n" + "=" * 70)
print("3. Mutual information and conditional entropy")
print("=" * 70)

joint = ct_raw.div(n)                           # joint P(coverage, man_zone)
p_mz  = joint.sum(axis=0)                       # marginal P(man_zone)
p_ct  = joint.sum(axis=1)                       # marginal P(coverage)

H_ct  = -float((p_ct[p_ct > 0] * np.log2(p_ct[p_ct > 0])).sum())
H_mz  = -float((p_mz[p_mz > 0] * np.log2(p_mz[p_mz > 0])).sum())

# H(man_zone | coverage)
H_mz_given_ct = 0.0
for cov in p_ct.index:
    p_row = joint.loc[cov] if cov in joint.index else pd.Series(dtype=float)
    p_cov = float(p_ct.get(cov, 0))
    if p_cov == 0:
        continue
    cond = p_row / p_cov
    cond = cond[cond > 0]
    H_mz_given_ct += p_cov * float(-(cond * np.log2(cond)).sum())

MI = H_mz - H_mz_given_ct

print(f"\n  H(man_zone_type)               = {H_mz:.4f} bits")
print(f"  H(man_zone_type | coverage)    = {H_mz_given_ct:.4f} bits")
print(f"  Mutual information             = {MI:.4f} bits")
print(f"  Fraction of H(mz) explained by coverage: {(1 - H_mz_given_ct / H_mz) * 100:.1f}%")

# ── 4. Coverage types where man_zone is ambiguous ────────────────────────────
print("\n" + "=" * 70)
print("4. Coverage types where defense_man_zone_type is NOT fully determined")
print("   (i.e., the same coverage_type appears under BOTH man and zone)")
print("=" * 70)

ambiguous = ct_raw[(ct_raw > 0).all(axis=1)]
if ambiguous.empty:
    print("\n  None — coverage_type perfectly determines man_zone_type.")
else:
    print(f"\n  {len(ambiguous)} coverage type(s) appear under BOTH MAN and ZONE:\n")
    print(ambiguous.to_string())
    print("\n  Row %:")
    print(ct_pct.loc[ambiguous.index].to_string())

# ── 5. Candidate-space impact ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. Candidate-space impact of dropping defense_man_zone_type")
print("=" * 70)

# Reconstruct candidate set the same way the notebook does (MIN_COUNT=20)
CANDIDATE_KEYS = ["def_personnel", MZ, CT]

# Need def_personnel; it was added in the notebook. Approximate it from
# defense_personnel if available, otherwise skip that dimension.
if "def_personnel" not in df.columns:
    def _classify_defense(s):
        import re
        counts = {m[1]: int(m[0]) for m in re.findall(r"(\d+)\s+([A-Z]+)", str(s))}
        dbs = counts.get("CB", 0) + counts.get("FS", 0) + counts.get("SS", 0) + counts.get("S", 0)
        if dbs <= 3:  return "Heavy"
        if dbs == 4:  return "Base"
        if dbs == 5:  return "Nickel"
        if dbs == 6:  return "Dime"
        return "Quarter"
    df["def_personnel"] = df["defense_personnel"].apply(_classify_defense)

cands = (
    df.groupby(CANDIDATE_KEYS, observed=True)
    .size().reset_index(name="count")
)
cands_full = cands[cands["count"] >= 20].copy()
print(f"\n  Full candidate set (with man_zone_type): {len(cands_full)} tuples")

# Collapse by dropping man_zone_type
cands_collapsed = (
    df.groupby(["def_personnel", CT], observed=True)
    .size().reset_index(name="count")
)
cands_collapsed = cands_collapsed[cands_collapsed["count"] >= 20].copy()
print(f"  Collapsed (without man_zone_type):       {len(cands_collapsed)} tuples")
print(f"  Reduction: {len(cands_full) - len(cands_collapsed)} fewer candidates "
      f"({(1 - len(cands_collapsed)/len(cands_full))*100:.0f}%)")

# Show which full candidates map to the same collapsed key
dup_mask = cands_full.duplicated(subset=["def_personnel", CT], keep=False)
merged_pairs = cands_full[dup_mask].sort_values(["def_personnel", CT])
if merged_pairs.empty:
    print("\n  No full-candidates share the same (def_personnel, coverage) key —")
    print("  dropping man_zone_type would NOT merge any candidates.")
else:
    print(f"\n  Full-candidate tuples that WOULD MERGE when man_zone_type is dropped:")
    print(merged_pairs.to_string(index=False))

# ── 6. Verdict ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. Verdict")
print("=" * 70)
print(f"""
  Cramer's V = {cramers_v:.3f}  ({label} association).
  {(1 - H_mz_given_ct / H_mz) * 100:.0f}% of man_zone_type's entropy is already explained by coverage_type.

  Practical question for the action space:
  - If ZERO full-candidate tuples share the same (personnel, coverage) pair,
    dropping man_zone_type merges NO candidates and removes a redundant feature.
  - If some pairs do share a key, dropping it collapses those candidates,
    narrowing the action space and potentially losing a real distinction.

  Check the output of section 5 above to make the final call.
""")
