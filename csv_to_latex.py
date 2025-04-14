def get_metrics_all_phantoms(
    phantoms: list[np.ndarray],
    all_methods: list[list[Method]],  # List of methods per phantom
):
    all_metrics = []
    for i, (phantom, methods) in enumerate(zip(phantoms, all_methods)):
        print(f"Phantom {i + 1}/{len(phantoms)}")
        for method in methods:
            print(f"  Evaluating {method.title}")
            ssim_val = utils.ssim_3d(method.reconstructed_images, phantom)
            psnr_val = utils.psnr(method.reconstructed_images, phantom)
            all_metrics.append((i, method.title, ssim_val, psnr_val))
    return all_metrics


all_gt_phantoms = [
    gt_phantom,
]
all_sliced_methods = [
    sliced_methods,
]
metrics_all = get_metrics_all_phantoms(all_gt_phantoms, all_sliced_methods)

df_metrics = pd.DataFrame(metrics_all, columns=["Phantom ID", "Method", "SSIM", "PSNR"])


# %%
df_metrics_all = df_metrics.copy()

# for each method splot on ' ' and use first word
df_metrics_all["Method"] = df_metrics_all["Method"].apply(lambda x: x.split(" ")[0])
df_metrics_all["Method"] = df_metrics_all["Method"].replace(["GT", "GT Scan"])

# rename NS to Navier Stokes
df_metrics_all["Method"] = df_metrics_all["Method"].replace("NS", "Navier Stokes")
df_metrics_all["Method"] = df_metrics_all["Method"].replace("Full", "Full Scan")
df_metrics_all["Method"] = df_metrics_all["Method"].replace("Train", "Train Set")

# Drop GT
df_metrics_all = df_metrics_all[df_metrics_all["Method"] != "GT"]


df_pivot = df_metrics_all.pivot_table(
    index="PhantomID", columns=["Method"], values=["SSIM", "PSNR"]
)

# Swap levels: make Method the top-level, Metric the sub-level
df_pivot = df_pivot.swaplevel(axis=1)
df_pivot = df_pivot.sort_index(axis=1, level=0)

priority_methods = [
    "Full Scan",
    "Train Set",
    "Biharmonic",
    "TV",
    "FMM",
    "Navier Stokes",
    "NVS",
]
current_methods = df_pivot.columns.levels[0].tolist()
new_method_order = priority_methods + [
    m for m in current_methods if m not in priority_methods
]
df_pivot = df_pivot.reindex(columns=new_method_order, level=0)


# Rename metrics with arrows
df_pivot.columns = df_pivot.columns.set_levels(
    [
        f"{lvl}" if lvl in {"PSNR", "SSIM"} else lvl
        for lvl in df_pivot.columns.levels[1]
    ],
    level=1,
)

# Add average row
df_pivot.loc["Avg"] = df_pivot.mean()
n_methods = len(df_pivot.columns.levels[0])
col_format = "|l|" + "|".join(["cc"] * n_methods) + "|"
# Export to LaTeX
latex_table = df_pivot.to_latex(
    multicolumn=True,
    multicolumn_format="c",
    index_names=True,
    escape=False,
    caption="Reconstruction quality metrics per phantom.",
    label="tab:phantom_metrics",
    float_format="%.3f",
    column_format=col_format,
)


table_title = "phantom_metrics"
with open(f"tables/{table_title}.tex", "w") as f:
    f.write(latex_table)

with open(f"tables/{table_title}.csv", "w") as f:
    df_pivot.to_csv(f, index=True)

df_pivot
