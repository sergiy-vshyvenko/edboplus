"""
Solvent mixtures via linear interpolation of PCA coordinates.

Because PCA is a linear transform, interpolating in principal-component space is
mathematically identical to interpolating the underlying physicochemical
descriptors. For a blend with fraction ``f`` of solvent A and ``1 - f`` of B::

    PC_mix = f * PC_A + (1 - f) * PC_B

is exactly the PCA score of the descriptor-average ``f * x_A + (1 - f) * x_B``.

The single chemistry assumption is that descriptors mix *linearly by fraction*
(a "mole-like" basis). This is exact for averaging molecular descriptors and a
reasonable approximation for blends; it does not capture bulk-property
non-idealities (e.g. non-linear dielectric behaviour). Fractions are used as
given — no volume/mass to mole conversion is performed.

The helpers here return a lookup-shaped DataFrame (``Name, PC1, PC2, ...``) that
is a drop-in replacement for ``data/Solvent_PC_clean.csv``. Pass it to
``generate_reaction_scope`` via the ``encodings`` parameter, then exclude the
label column in ``run()``::

    from edbo.plus.mixtures import binary_ratio_grid

    mix = binary_ratio_grid('DMSO [Dimethylsulfoxide]', 'DCM [Dichloromethane]')
    EDBOplus().generate_reaction_scope(
        components={'solvent': mix['Name'].tolist(), ...},
        encodings={'solvent': {'file': mix, 'key': 'Name',
                               'features': ['PC1', 'PC2', 'PC3', 'PC4']}},
        filename='reaction.csv')
    EDBOplus().run(filename='reaction.csv', objectives=['yield'],
                   objective_mode=['max'], exclude_columns=['solvent'])
"""

import re

import pandas as pd

from .scope_generator import _load_solvent_lookup


def _resolve_lookup(lookup):
    """Return a lookup DataFrame, loading the bundled table when None."""
    if lookup is None:
        lookup = _load_solvent_lookup()
    if lookup is None:
        raise ValueError(
            "No solvent lookup available. The bundled data/Solvent_PC_clean.csv "
            "was not found — pass an explicit `lookup` DataFrame."
        )
    return lookup


def _detect_pc_columns(lookup, features):
    """Return the list of PC feature columns to interpolate over."""
    if features is not None:
        missing = [f for f in features if f not in lookup.columns]
        if missing:
            raise ValueError(f"Features not found in lookup: {missing}")
        return list(features)
    pc_cols = [c for c in lookup.columns if re.fullmatch(r'PC\d+', str(c))]
    if not pc_cols:
        raise ValueError(
            "No PC columns (matching 'PC<number>') found in the lookup table."
        )
    return pc_cols


def _short_label(name):
    """Readable short label: text before '[' (e.g. 'DMSO [Dimethylsulfoxide]' -> 'DMSO')."""
    return str(name).split('[')[0].strip()


def make_mixtures(mixtures, lookup=None, key='Name', features=None):
    """
    Build a lookup-shaped DataFrame of solvent blends by linear PCA interpolation.

    Parameters
    ----------
    mixtures : dict
        Maps a blend name to a dict of ``{pure_solvent_name: weight}``. Weights are
        normalized to sum to 1, so ``{A: 1, B: 1}`` means a 50:50 blend.
    lookup : pandas.DataFrame, optional
        Table of pure-solvent PC coordinates. Defaults to the bundled
        data/Solvent_PC_clean.csv.
    key : str
        Name of the solvent-identifier column in *lookup* (default 'Name').
    features : list of str, optional
        PC columns to interpolate. Defaults to auto-detecting all 'PC<n>' columns.

    Returns
    -------
    pandas.DataFrame
        One row per blend with columns ``[key] + features``.
    """
    lookup = _resolve_lookup(lookup)
    pc_cols = _detect_pc_columns(lookup, features)
    known = set(lookup[key])
    coords = lookup.set_index(key)[pc_cols]

    rows = []
    for name, weights in mixtures.items():
        if not weights:
            raise ValueError(f"Mixture '{name}' has no components.")
        missing = set(weights) - known
        if missing:
            raise ValueError(
                f"Mixture '{name}' references solvents not in the lookup table: {missing}"
            )
        total = float(sum(weights.values()))
        if total <= 0:
            raise ValueError(
                f"Mixture '{name}' has non-positive total weight ({total})."
            )
        blended = {c: 0.0 for c in pc_cols}
        for solvent, w in weights.items():
            frac = w / total
            for c in pc_cols:
                blended[c] += frac * float(coords.loc[solvent, c])
        rows.append({key: name, **blended})

    return pd.DataFrame(rows, columns=[key] + pc_cols)


def binary_ratio_grid(solvent_a, solvent_b, fractions=(0, 0.25, 0.5, 0.75, 1.0),
                      lookup=None, label_a=None, label_b=None, key='Name',
                      features=None):
    """
    Build a grid of binary blends across a range of mixing ratios.

    Parameters
    ----------
    solvent_a, solvent_b : str
        Pure-solvent names as they appear in *lookup*.
    fractions : iterable of float
        Fraction of *solvent_a* in each blend; ``0`` -> pure B, ``1`` -> pure A.
        Endpoints are included, so the grid inherently contains the pure solvents.
    lookup : pandas.DataFrame, optional
        Defaults to the bundled data/Solvent_PC_clean.csv.
    label_a, label_b : str, optional
        Short labels for blend names. Default: text before '[' in the full name
        (e.g. 'DMSO [Dimethylsulfoxide]' -> 'DMSO').
    key, features : see :func:`make_mixtures`.

    Returns
    -------
    pandas.DataFrame
        One row per fraction, named like ``'DMSO/DCM 50:50'``.
    """
    la = label_a or _short_label(solvent_a)
    lb = label_b or _short_label(solvent_b)

    mixtures = {}
    for f in fractions:
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"Fraction {f} is outside [0, 1].")
        name = f"{la}/{lb} {f * 100:.0f}:{(1 - f) * 100:.0f}"
        mixtures[name] = {solvent_a: f, solvent_b: 1 - f}

    return make_mixtures(mixtures, lookup=lookup, key=key, features=features)


def combine_lookups(*tables, dedupe_key='Name'):
    """
    Concatenate pure-solvent and mixture lookup tables into one.

    Useful for scopes that offer both pure solvents and blends as choices.
    Duplicate names are dropped (first occurrence kept).

    Parameters
    ----------
    *tables : pandas.DataFrame
        Lookup-shaped tables to stack (same columns).
    dedupe_key : str
        Column used to detect duplicates (default 'Name').

    Returns
    -------
    pandas.DataFrame
    """
    combined = pd.concat(tables, ignore_index=True)
    if dedupe_key in combined.columns:
        combined = combined.drop_duplicates(subset=dedupe_key, keep='first')
    return combined.reset_index(drop=True)
