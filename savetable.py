import pandas as pd

# File list
title = "limited_scan_metrics"
dfs = [
    "dump_[limited scan][4]#2025-04-14#22-36",
    "dump_[limited scan][4]#2025-04-14#23-41",
    "dump_[limited scan][4]#2025-04-14#23-32",
]
dfs = [f"results/out/{folder}/metrics_4_limited_scan.csv" for folder in dfs]


title = "sparse_scan_metrics"
dfs = [
    "dump_[sparse scan][13]#2025-04-14#22-02",
    "dump_[sparse scan][13]#2025-04-14#22-11",
]
dfs = [f"results/out/{folder}/metrics_13_sparse_scan.csv" for folder in dfs]

# Method rename map
method_map = {
    "Full Scan Reconstruction": "Full Scan",
    "Train Set Reconstruction": "Train Set",
    "Biharmonic Reconstruction": "Biharmonic",
    "TV Reconstruction": "TV",
    "FMM Reconstruction": "FMM",
    "NS Reconstruction": "Navier Stokes",
    "NVS Reconstruction": "NVS",
}

# Step 1: Parse and reshape
all_data = []
for phantom_id, path in enumerate(dfs):
    df = pd.read_csv(path)
    df = df[df["Method"] != "GT Slice"]
    df["Method"] = df["Method"].map(method_map)
    df = df[df["Method"] != "Full Scan"]  # Remove "Full Scan"
    df["PhantomID"] = phantom_id
    all_data.append(df)

df_all = pd.concat(all_data)

# Step 2: Pivot
pivot_psnr = df_all.pivot(index="PhantomID", columns="Method", values="PSNR")
pivot_ssim = df_all.pivot(index="PhantomID", columns="Method", values="SSIM")


# Step 3: Format for LaTeX
def fmt(x):
    return f"{x:.3f}"


latex_df = pd.DataFrame(index=pivot_psnr.index)
for method in method_map.values():
    if method != "Full Scan":  # Skip "Full Scan"
        latex_df[f"{method} PSNR"] = pivot_psnr[method].map(fmt)
        latex_df[f"{method} SSIM"] = pivot_ssim[method].map(fmt)


# Reorganize columns as MultiIndex: [(method, PSNR), (method, SSIM), ...]
tuples = []
data = {}
for method in method_map.values():
    if method != "Full Scan":
        tuples.extend([(method, "PSNR"), (method, "SSIM")])
        data[(method, "PSNR")] = pivot_psnr[method].map(fmt)
        data[(method, "SSIM")] = pivot_ssim[method].map(fmt)

# Create MultiIndex DataFrame
multi_columns = pd.MultiIndex.from_tuples(tuples, names=["Method", "Metric"])
latex_df = pd.DataFrame(data, index=pivot_psnr.index)
avg_row = latex_df.astype(float).mean().map(fmt)
avg_row.name = "Avg"
latex_df = pd.concat([latex_df, avg_row.to_frame().T])
latex_df.index.name = "PhantomID"

# Export to LaTeX with multirow columns
latex_out = latex_df.reset_index().to_latex(
    index=False,
    multicolumn=True,
    multicolumn_format="c",
    multirow=True,
    column_format="|l" + "cc|" * len(method_map.keys() - {"Full Scan"}),
    caption="Reconstruction quality metrics per phantom.",
    label="tab:phantom_metrics",
    escape=False,
)

# Save to file
with open(f"tables/{title}.tex", "w") as f:
    f.write(latex_out)
