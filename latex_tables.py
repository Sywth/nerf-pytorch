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


def round_values(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    return df.round(decimals)


def color_ranked_cells(
    df: pd.DataFrame,
    ignore_rows: tuple[str, ...] = ("SIRT Full Set",),
    decimals: int = 3,
) -> pd.DataFrame:
    df_out = df.copy()
    for col in df.columns:
        # Extract ranking candidates (excluding ignored rows)
        valid_series = df.loc[~df.index.isin(ignore_rows), col]
        sorted_vals = valid_series.sort_values(ascending=False)

        # Assign ranks
        ranks = {}
        if not sorted_vals.empty:
            ranks[sorted_vals.iloc[0]] = "rankfirst"
        if len(sorted_vals) > 1:
            ranks[sorted_vals.iloc[1]] = "ranksecond"
        if len(sorted_vals) > 2:
            ranks[sorted_vals.iloc[2]] = "rankthird"
        if len(sorted_vals) > 0:
            ranks[sorted_vals.iloc[-1]] = "rankworst"

        # Map each cell to its value with optional cellcolor
        def format_cell(x):
            color = ranks.get(x, None)
            value = f"{x:.{decimals}f}"
            return f"\\cellcolor{{{color}}}{value}" if color else value

        df_out[col] = df[col].apply(format_cell)

    return df_out


def table_to_latex(
    df: pd.DataFrame, save_path: Path, scan_type: str, num_scans: int
) -> str:
    df_rounded = df.pipe(round_values)
    df_formatted = df_rounded.pipe(color_ranked_cells)

    # Extract top-level (phantom names) and second-level (metrics)
    top_level_labels = df_formatted.columns.get_level_values(0)
    second_level_labels = df_formatted.columns.get_level_values(1)

    # Count subcolumns per phantom group and preserve order
    unique_phantoms = list(dict.fromkeys(top_level_labels))
    phantom_counts = [sum(top_level_labels == phantom) for phantom in unique_phantoms]

    # Construct column format string with vertical separators between phantom groups
    col_format = "|l"
    for count in phantom_counts:
        col_format += "|" + "c" * count
    col_format += "|"  # Closing vertical bar

    # Add vertical bars after each multicolumn group
    multicolumns = {
        phantom: f"\\multicolumn{{{count}}}{{c|}}{{{phantom}}}"
        for phantom, count in zip(unique_phantoms, phantom_counts)
    }

    # Patch pandas-generated LaTeX to inject vertical lines in \multicolumns
    latex_str = df_formatted.to_latex(
        column_format=col_format,
        index=True,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        escape=False,
    )

    # Replace \multicolumns to include trailing vertical bars
    for phantom, replacement in multicolumns.items():
        latex_str = latex_str.replace(
            f"\\multicolumn{{{phantom_counts[unique_phantoms.index(phantom)]}}}{{c}}{{{phantom}}}",
            replacement,
        )

    return latex_str


def save_table(table: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(table)


# %%
scan_types = ["limited", "sparse"]
scan_nums = [64, 256]
save_path = Path("./tables/")

all_limited = []
all_sparse = []
for scan_type, num_scans in product(scan_types, scan_nums):
    df_table = get_table(scan_type, num_scans)

    if scan_type == "limited":
        all_limited.append(df_table)
    else:
        all_sparse.append(df_table)

    table = table_to_latex(df_table, save_path, scan_type, num_scans)
    save_table(table, save_path / f"table-{scan_type}-{num_scans}.tex")


# %%
# Combine all tables and compute final mean SSIM and PSNR
def compute_final_means(
    all_tables: list[pd.DataFrame], row_to_name_idx: dict
) -> pd.DataFrame:
    df_concat = pd.concat(all_tables)

    # Extract only Mean SSIM and PSNR columns
    df_means = df_concat[[("Mean", "SSIM"), ("Mean", "PSNR")]]

    # Group by method (index), average across all scans
    df_final_means = df_means.groupby(df_means.index).mean()

    # Optional: Round values
    df_final_means = round_values(df_final_means)

    # Convert MultiIndex columns to a single index with two columns: SSIM and PSNR
    df_final_means.columns = ["SSIM", "PSNR"]

    # Order the rows based on row_to_name_idx
    df_final_means = df_final_means.loc[
        [v[0] for k, v in sorted(row_to_name_idx.items(), key=lambda item: item[1][1])]
    ]

    return df_final_means


for scan_type, all_tables in [("limited", all_limited), ("sparse", all_sparse)]:
    df_total = compute_final_means(
        all_tables,
        row_to_name_idx,
    )
    df_total_highlighted = color_ranked_cells(df_total, decimals=3)
    latex_total = df_total_highlighted.to_latex(
        index=True,
        escape=False,
        column_format="|l|c|c|",
        multicolumn=True,
        multicolumn_format="c",
    )
    save_table(latex_total, save_path / f"table-{scan_type}-acc.tex")

# %% Get Z-Tables
from IPython.display import display, HTML


def get_z_table(df):
    return (df - df.mean()) / df.std()


for scan_type, all_tables in [("limited", all_limited), ("sparse", all_sparse)]:
    for i, N in enumerate([64, 256]):
        df_mets = all_tables[i].copy()
        df_mets.columns = df_mets.columns.map(lambda x: f"{x[0]} {x[1]}")
        # remove last 2 columns (mean SSIM and PSNR)
        df_mean_col = df_mets.iloc[:, -2:]

        df_mean_ssim = df_mean_col.iloc[:, 0]
        df_mean_psnr = df_mean_col.iloc[:, 1]

        df_mets = df_mets.iloc[:, :-2]

        # seperate psnrs (even columns) and ssims (odd columns)
        df_mets_psnr = df_mets.iloc[:, ::2].copy()
        df_mets_ssim = df_mets.iloc[:, 1::2].copy()

        df_mets_psnr_z = get_z_table(df_mets_psnr)
        df_mets_ssim_z = get_z_table(df_mets_ssim)

        print(f"Data for {scan_type} {N} scans")
        display(HTML(df_mets_psnr_z.to_html(index=True, escape=False)))
        display(HTML(df_mets_ssim_z.to_html(index=True, escape=False)))
        display(
            HTML(get_z_table(df_mean_psnr).to_frame().to_html(index=True, escape=False))
        )
        display(
            HTML(get_z_table(df_mean_ssim).to_frame().to_html(index=True, escape=False))
        )
