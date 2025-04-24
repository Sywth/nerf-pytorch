# %%
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

# %%

row_to_name_idx = {
    "Full Scan Reconstruction": ("SIRT Full Set", 0),
    "Train Set Reconstruction": ("SIRT Train Set", 1),
    "LERP Reconstruction": ("LERP", 2),
    "NeRF Reconstruction": ("NeRF", 3),
    "Biharmonic Reconstruction": ("Biharmonic", 4),
    "NS Reconstruction": ("Navier-Stokes", 5),
    "FMM Reconstruction": ("Fast Marching Method", 6),
    "TV Reconstruction": ("TVR", 7),
    "GS Reconstruction": ("Gaussian Splatting", 8),
}


def get_table(scan_type: str, num_scans: int, ph_indexes=[4, 13, 16]) -> pd.DataFrame:
    tables = []
    for ph_idx in ph_indexes:
        table_name = f"tables/metrics-{scan_type}-{ph_idx}-{num_scans}.csv"
        df = pd.read_csv(table_name)

        df = df[df["Method"].notna()]
        df = df[df["Method"] != "GT Slice"]
        df["Phantom Index"] = f"Phantom {ph_idx}"
        tables.append(df)

    df_all = pd.concat(tables)

    # Pivot
    df_pivot = df_all.pivot(index="Method", columns="Phantom Index")[["SSIM", "PSNR"]]
    df_pivot = df_pivot.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    # Compute means
    ssim_mean = df_pivot.xs("SSIM", axis=1, level=1).mean(axis=1)
    psnr_mean = df_pivot.xs("PSNR", axis=1, level=1).mean(axis=1)
    df_pivot[("Mean", "SSIM")] = ssim_mean
    df_pivot[("Mean", "PSNR")] = psnr_mean

    # Rename and sort index based on row_to_name_idx
    df_pivot = df_pivot[df_pivot.index.isin(row_to_name_idx)]
    df_pivot["__sort_key__"] = df_pivot.index.map(lambda x: row_to_name_idx[x][1])
    df_pivot.index = df_pivot.index.map(lambda x: row_to_name_idx[x][0])
    df_pivot = df_pivot.sort_values("__sort_key__")
    df_pivot = df_pivot.drop(columns="__sort_key__")

    return df_pivot


def table_to_latex(
    df: pd.DataFrame, save_path: Path, scan_type: str, num_scans: int
) -> str:
    df_formatted = df.copy()
    df_formatted = df_formatted.map(lambda x: f"{x:.3f}")

    # Count how many subcolumns per phantom group
    top_level_labels = df_formatted.columns.get_level_values(0)
    unique_phantoms = list(dict.fromkeys(top_level_labels))  # Preserve order
    phantom_counts = [sum(top_level_labels == phantom) for phantom in unique_phantoms]

    # Construct column format string: vertical bars between phantom groups
    col_format = "|l"  # First column (Method)
    for count in phantom_counts:
        col_format += "|" + "c" * count
    col_format += "|"  # Final closing bar

    # Generate LaTeX with custom column format
    latex_str = df_formatted.to_latex(
        column_format=col_format,
        index=True,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        escape=False,
    )

    return latex_str


def save_table(table: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(table)


# %%
scan_types = ["limited", "sparse"]
scan_nums = [64]
save_path = Path("./tables/")

for scan_type, num_scans in product(scan_types, scan_nums):
    df_table = get_table(scan_type, num_scans)
    table = table_to_latex(df_table, save_path, scan_type, num_scans)
    save_table(table, save_path / f"table-{scan_type}-{num_scans}.tex")


# %%
df_table
