
# <img src="EDBOLogo.png" width="190">

## EDBO+  —  Bayesian reaction optimization as a tool for chemical synthesis

**WebApp:** https://edboplus.org

**Reference:** Garrido Torres, Jose A.; Lau, Sii Hong; Anchuri, Pranay; Stevens, Jason M.; Tabora, Jose E.; Li, Jun; Borovika, Alina; Adams, Ryan P.; Doyle, Abigail G. "A Multi-Objective Active Learning Platform and Web App for Reaction Optimization."

**DOI:** [10.1021/jacs.2c08592](https://doi.org/10.1021/jacs.2c08592) · [ChemRxiv preprint](https://chemrxiv.org/engage/chemrxiv/article-details/62f6966269f3a5df46b5584b)

---

## Installation

EDBO+ depends on a mix of packages that are best installed in a specific order to avoid binary incompatibilities between conda-managed and pip-managed libraries. The procedure below is the one that has been verified to work.

**Prerequisites:** [Miniforge / Mamba](https://github.com/conda-forge/miniforge) is recommended over plain conda because its solver handles complex dependency graphs more reliably. If you only have conda, replace every `mamba` command with `conda`.

### Step 1 — Create a Python 3.9 environment

```bash
mamba create -n edbo_env python=3.9 -c conda-forge
mamba activate edbo_env
```

### Step 2 — Clone the repository

```bash
git clone https://github.com/sergiy-vshyvenko/edboplus.git
cd edboplus
```

### Step 3 — Pin setuptools before anything else

Modern setuptools (≥ 60) breaks the build of several EDBO+ dependencies. Downgrade it first and remove the stale dist-info left behind:

```bash
mamba install setuptools=59.0 -c conda-forge
rm -rf ~/miniforge3/envs/edbo_env/lib/python3.9/site-packages/setuptools-*.dist-info
```

> If your miniforge is installed elsewhere, adjust the path accordingly (e.g. `/opt/homebrew/Caskroom/miniforge/...`).

### Step 4 — Install binary scientific packages via mamba

Installing these through mamba rather than pip avoids numpy binary incompatibility errors (`numpy.dtype size changed`):

```bash
mamba install lxml pandas numpy scipy scikit-learn seaborn matplotlib tqdm -c conda-forge
```

### Step 5 — Editable install of EDBO+ (skip deps — already handled)

```bash
pip install -e . --no-build-isolation --no-deps
```

### Step 6 — Install remaining pip-only dependencies

```bash
pip install \
  botorch==0.5.0 \
  gpytorch==1.5.1 \
  ipykernel==6.5.1 \
  ipython==7.29.0 \
  ipywidgets==7.6.5 \
  Jinja2==3.0.3 \
  joypy==0.2.6 \
  mordred==1.2.0 \
  ordered-set==4.0.2 \
  pareto==1.1.1.post3 \
  pymoo==0.5.0 \
  sympy==1.9 \
  --no-build-isolation
```

### Step 7 — Install IDAES (space-filling samplers)

IDAES is a heavier package installed separately:

```bash
pip install idaes-pse --no-build-isolation
```

### Step 8 — Verify the installation

```bash
python -c "from edbo.plus.optimizer_botorch import EDBOplus; print('OK')"
```

You should see `OK`.

### Step 9 — Install JupyterLab (for notebooks)

```bash
pip install jupyterlab
```

Always launch Jupyter **from within the activated environment**:

```bash
mamba activate edbo_env
jupyter lab
```

---

## Quick Start

The core workflow has three steps: generate a reaction scope, run EDBO+ to get suggested experiments, and enter your observations.

```python
from edbo.plus.optimizer_botorch import EDBOplus

# 1. Define your combinatorial search space
components = {
    'solvent':       ['THF [Tetrahydrofuran]', 'Toluene', 'DMSO [Dimethylsulfoxide]'],
    'base':          ['KOAc', 'Cs2CO3', 'K3PO4'],
    'temperature':   [25, 50, 80],
    'concentration': [0.1, 0.2, 0.5],
}

# 2. Generate the scope CSV (all combinations)
EDBOplus().generate_reaction_scope(
    components=components,
    filename='reaction.csv',
)

# 3. Run EDBO+ to get the first suggested experiments
EDBOplus().run(
    filename='reaction.csv',
    objectives=['yield'],
    objective_mode=['max'],
    batch=4,
)
```

Open `reaction.csv` in any spreadsheet editor, fill in the observed values for the suggested experiments (`priority=1`), then run step 3 again. Repeat until satisfied.

---

## PCA Solvent Encodings

By default, string-valued components (like solvent names) are encoded using **One-Hot Encoding (OHE)**, which treats each solvent as independent. EDBO+ ships with a lookup table of **272 common solvents** described by 5 principal components (PC1–PC5) derived from physicochemical descriptors. Using these continuous coordinates instead of OHE lets the Gaussian Process model learn chemical similarity between solvents.

### Automatic detection

When every solvent name in a component column matches an entry in the bundled `data/Solvent_PC_clean.csv` table, EDBO+ applies PC1–PC4 encodings **automatically** — no extra configuration needed:

```python
components = {
    'solvent':       ['THF [Tetrahydrofuran]', 'Toluene', 'DMSO [Dimethylsulfoxide]',
                      'Acetone', 'Methanol', 'DMF [N,N-Dimethylformamide]'],
    'base':          ['KOAc', 'Cs2CO3'],   # not solvents → will be OHE'd as usual
    'temperature':   [25, 50, 80],
    'concentration': [0.1, 0.2, 0.5],
}

# No encodings dict required — EDBO+ detects the solvents automatically
EDBOplus().generate_reaction_scope(
    components=components,
    filename='reaction.csv',
)
# Prints: "Auto-applied PCA solvent encodings (PC1–PC4) for: ['solvent']."
# Remember to pass exclude_columns=['solvent'] to run() (see below).
```

When running the optimizer, pass the solvent column name to `exclude_columns` so the model uses the PC features rather than the label string:

```python
EDBOplus().run(
    filename='reaction.csv',
    objectives=['yield', 'ee'],
    objective_mode=['max', 'max'],
    exclude_columns=['solvent'],   # label kept in CSV for readability; PC1–PC4 used by model
    batch=4,
)
```

### Solvent lookup table

The lookup lives in `data/Solvent_PC_clean.csv`. Solvent names must match the `Name` column exactly (case-sensitive). You can search it programmatically:

```python
import pandas as pd
lut = pd.read_csv('data/Solvent_PC_clean.csv')
print(f"{len(lut)} solvents available.")
lut[lut['Name'].str.contains('THF|DMSO|Toluene', case=False)]
```

PC1–PC4 together capture **93.9 %** of total solvent variance and are the recommended set of features. PC5 adds only 6.1 % and can be omitted.

### Manual encodings (advanced)

If you need custom feature sets, a different lookup file, or want to encode a non-solvent component, pass an explicit `encodings` dict — it takes priority over auto-detection:

```python
encodings = {
    'solvent': {
        'file':     'data/Solvent_PC_clean.csv',  # path or DataFrame
        'key':      'Name',                        # column to match on
        'features': ['PC1', 'PC2', 'PC3'],         # subset of PCs
    }
}

EDBOplus().generate_reaction_scope(
    components=components,
    encodings=encodings,
    filename='reaction.csv',
)
```

### Solvent mixtures (blends)

Because PCA is a linear transform, a **blend** of two solvents can be represented by
interpolating their PC coordinates. For a mixture with fraction `f` of solvent A and `1 − f` of B:

```
PC_mix = f · PC_A + (1 − f) · PC_B
```

This is exactly the PCA score of the averaged physicochemical descriptors, so a blend lands on
the straight line between its pure components in PC space. The helpers in `edbo.plus.mixtures`
build a lookup-shaped DataFrame you feed straight into `encodings`:

```python
from edbo.plus.mixtures import make_mixtures, binary_ratio_grid

# Named blends (weights are normalized → {A:1, B:1} = 50:50)
blends = make_mixtures({
    'DMSO/DCM 50:50': {'DMSO [Dimethylsulfoxide]': 1, 'DCM [Dichloromethane]': 1},
})

# Or a ratio grid for EDBO+ to optimize over (pure endpoints included)
grid = binary_ratio_grid('DMSO [Dimethylsulfoxide]', 'DCM [Dichloromethane]',
                         fractions=[0, 0.25, 0.5, 0.75, 1.0])

EDBOplus().generate_reaction_scope(
    components={'solvent': grid['Name'].tolist(), 'temperature': [25, 50]},
    encodings={'solvent': {'file': grid, 'key': 'Name', 'features': ['PC1', 'PC2', 'PC3', 'PC4']}},
    filename='reaction.csv',
)
EDBOplus().run(filename='reaction.csv', objectives=['yield'],
               objective_mode=['max'], exclude_columns=['solvent'], batch=4)
```

Use `combine_lookups(pures, blends)` to offer pure solvents and blends as choices in the same
scope. Fractions are a **linear, mole-like basis** used as given (no volume→mole conversion) —
exact for averaging molecular descriptors, an approximation for non-ideal bulk behaviour. See
the [solvent mixtures tutorial](examples/tutorials/solvent_mixtures.ipynb) for a full walkthrough.

---

## Tutorials

| Notebook | Description |
|---|---|
| [`examples/tutorials/1_CLI_example.ipynb`](examples/tutorials/1_CLI_example.ipynb) | Basic workflow: scope generation, initialization, iterative optimization |
| [`examples/tutorials/PCA_example.ipynb`](examples/tutorials/PCA_example.ipynb) | PCA solvent encodings: auto-detection, mixed encoding types, reading model predictions |
| [`examples/tutorials/OHE_vs_PCA_benchmark.ipynb`](examples/tutorials/OHE_vs_PCA_benchmark.ipynb) | Side-by-side benchmark: OHE vs PCA encoding on the BMS cross-coupling dataset (yield + cost) |
| [`examples/tutorials/solvent_mixtures.ipynb`](examples/tutorials/solvent_mixtures.ipynb) | Solvent mixtures: interpolating PCA coordinates to optimize over blend ratios |

---

## Citation

```bibtex
@article{GarridoTorres2022,
  title   = {A Multi-Objective Active Learning Platform and Web App for Reaction Optimization},
  author  = {Garrido Torres, Jose A. and Lau, Sii Hong and Anchuri, Pranay and Stevens, Jason M.
             and Tabora, Jose E. and Li, Jun and Borovika, Alina and Adams, Ryan P. and Doyle, Abigail G.},
  journal = {Journal of the American Chemical Society},
  year    = {2022},
  doi     = {10.1021/jacs.2c08592},
}
```
