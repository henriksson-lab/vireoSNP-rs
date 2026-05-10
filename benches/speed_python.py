import statistics
import sys
import time
import os
from contextlib import redirect_stdout
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "vireo"))

from vireoSNP.utils.io_utils import match_donor_VCF, read_cellSNP
from vireoSNP.utils.vcf_utils import load_VCF
from vireoSNP.utils.vireo_model import Vireo
from vireoSNP.utils.vireo_wrap import vireo_wrap
from scipy.io import mmread
import numpy as np


cell_snp_dir = Path(os.environ.get("CELL_SNP_DIR", "vireo/data/cellSNP_mat"))
cells_vcf = (
    cell_snp_dir / "cellSNP.cells.vcf.gz"
    if (cell_snp_dir / "cellSNP.cells.vcf.gz").exists()
    else cell_snp_dir / "cellSNP.cells.vcf"
    if (cell_snp_dir / "cellSNP.cells.vcf").exists()
    else Path("vireo/data/cells.cellSNP.vcf.gz")
)
base_vcf_gz = cell_snp_dir / "cellSNP.base.vcf.gz"
base_vcf = base_vcf_gz if base_vcf_gz.exists() else cell_snp_dir / "cellSNP.base.vcf"
repeats = int(os.environ.get("BENCH_REPEATS", "10"))
fit_repeats = max(1, min(5, repeats))


def read_cellsnp_compatible(path):
    path = Path(path)
    if (path / "cellSNP.base.vcf.gz").exists():
        return read_cellSNP(str(path), layers=["AD", "DP"])
    cell_dat = load_VCF(str(path / "cellSNP.base.vcf"), load_sample=False, biallelic_only=False)
    for layer in ["AD", "DP"]:
        cell_dat[layer] = mmread(str(path / f"cellSNP.tag.{layer}.mtx")).tocsc()
    cell_dat["samples"] = np.genfromtxt(str(path / "cellSNP.samples.tsv"), dtype=str)
    return cell_dat


rows = []

samples = []
for _ in range(repeats):
    t0 = time.perf_counter()
    load_VCF(str(cells_vcf), biallelic_only=False, load_sample=True, sparse=True)
    samples.append(time.perf_counter() - t0)
rows.append(("load_cells_vcf", statistics.median(samples)))

samples = []
for _ in range(repeats):
    t0 = time.perf_counter()
    read_cellsnp_compatible(cell_snp_dir)
    samples.append(time.perf_counter() - t0)
rows.append(("read_cellsnp", statistics.median(samples)))

samples = []
for _ in range(repeats):
    cell_dat = read_cellsnp_compatible(cell_snp_dir)
    donor_vcf = load_VCF(
        "vireo/data/donors.two.cellSNP.vcf.gz",
        biallelic_only=False,
        load_sample=True,
        sparse=False,
        format_list=["GT", "PL"],
    )
    t0 = time.perf_counter()
    with open("/dev/null", "w") as devnull, redirect_stdout(devnull):
        match_donor_VCF(cell_dat, donor_vcf)
    samples.append(time.perf_counter() - t0)
rows.append(("match_donor_vcf", statistics.median(samples)))

dat = read_cellsnp_compatible(cell_snp_dir)
n_var80 = min(80, len(dat["variants"]))
n_cell60 = min(60, len(dat["samples"]))
ad80 = dat["AD"][:n_var80, :n_cell60]
dp80 = dat["DP"][:n_var80, :n_cell60]
samples = []
for _ in range(fit_repeats):
    model = Vireo(
        n_cell=n_cell60,
        n_var=n_var80,
        n_donor=2,
        n_GT=3,
        learn_GT=True,
        learn_theta=True,
        ASE_mode=False,
        fix_beta_sum=False,
    )
    t0 = time.perf_counter()
    model.fit(
        ad80,
        dp80,
        max_iter=4,
        min_iter=1,
        epsilon_conv=1e-2,
        delay_fit_theta=1,
        verbose=False,
        n_inits=1,
    )
    samples.append(time.perf_counter() - t0)
rows.append(("vireo_fit_slice", statistics.median(samples)))

n_var50 = min(50, len(dat["variants"]))
n_cell24 = min(24, len(dat["samples"]))
ad50 = dat["AD"][:n_var50, :n_cell24]
dp50 = dat["DP"][:n_var50, :n_cell24]
samples = []
for _ in range(fit_repeats):
    t0 = time.perf_counter()
    with open("/dev/null", "w") as devnull, redirect_stdout(devnull):
        vireo_wrap(
            ad50,
            dp50,
            GT_prior=None,
            n_donor=2,
            learn_GT=True,
            n_init=1,
            random_seed=0,
            check_doublet=False,
            max_iter_init=3,
            delay_fit_theta=1,
            n_extra_donor=0,
            extra_donor_mode="distance",
            check_ambient=False,
            nproc=1,
            ASE_mode=False,
            fix_beta_sum=False,
            n_GT=3,
    )
    samples.append(time.perf_counter() - t0)
rows.append(("vireo_wrap_slice", statistics.median(samples)))

for name, seconds in rows:
    print(f"python\t{name}\t{seconds:.9f}")
