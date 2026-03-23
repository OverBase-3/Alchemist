from __future__ import annotations
from typing import Dict

_STRAIN_TO_GROUP: Dict[str, str] = {
    "e.coli": "E.coli", "e.coli_bl21": "E.coli", "e.coli_bl21(de3)": "E.coli",
    "e.coli_rv308": "E.coli", "e.coli_hms174": "E.coli",
    "e.coli_mg1655": "E.coli", "e.coli_k-12_mg1655": "E.coli",
    "e.coli_k-12_w3110": "E.coli", "e.coli_w3110": "E.coli",
    "e.coli_b/r": "E.coli", "e.coli_atcc_25922": "E.coli",
    "escherichia_coli": "E.coli",
    "cho": "CHO", "cho-k1": "CHO", "chok1": "CHO",
    "chobc_bc-g": "CHO", "chobc_bc-p": "CHO", "cho_dg44": "CHO",
    "vero": "Vero", "vero_atcc-ccl-81": "Vero", "vero_ccl-81": "Vero",
    "lactobacillus_acidophilus": "Lactobacillus",
    "lacticaseibacillus_paracasei": "Lactobacillus",
    "lactiplantibacillus_pentosus": "Lactobacillus",
    "lactiplantibacillus_plantarum": "Lactobacillus",
    "limosilactobacillus_reuteri": "Lactobacillus",
    "lacticaseibacillus_rhamnosus": "Lactobacillus",
    "lactobacillus_plantarum": "Lactobacillus",
    "lactobacillus_rhamnosus": "Lactobacillus",
    "hek293": "HEK293", "hek293t": "HEK293", "hek-293": "HEK293",
    "bhk-21": "BHK21", "bhk21": "BHK21",
    "mdck": "MDCK",
    "saccharomyces_cerevisiae": "S.cerevisiae",
    "s.cerevisiae": "S.cerevisiae",
    "bacillus_subtilis": "B.subtilis", "b._subtilis": "B.subtilis",
}

GROUP_DENSITY_UNIT: Dict[str, str] = {
    "E.coli": "g/L CDM", "Lactobacillus": "g/L CDM",
    "CHO": "×10⁶ cells/mL", "Vero": "g/L CDM",
    "HEK293": "×10⁶ cells/mL", "BHK21": "×10⁶ cells/mL",
    "MDCK": "g/L CDM", "S.cerevisiae": "g/L CDM", "B.subtilis": "g/L CDM",
}

GROUP_DENSITY_SCALE: Dict[str, float] = {
    "E.coli": 1.0, "Lactobacillus": 1.0,
    "CHO": 1e-6, "Vero": 1.0,
    "HEK293": 1e-6, "BHK21": 1e-6,
    "MDCK": 1.0, "S.cerevisiae": 1.0, "B.subtilis": 1.0,
}


def normalize_cell_type(raw: str) -> str:
    key = (raw.strip().lower()
              .replace(" ", "_").replace("-", "_")
              .replace("(", "").replace(")", "").strip("_"))
    if key in _STRAIN_TO_GROUP:
        return _STRAIN_TO_GROUP[key]
    if any(x in key for x in ("e.coli", "ecoli", "escherichia")):
        return "E.coli"
    if "cho" in key:
        return "CHO"
    if any(x in key for x in ("lactobacillus", "lactiplantibacillus",
                               "lacticaseibacillus", "limosilactobacillus")):
        return "Lactobacillus"
    if "vero" in key:
        return "Vero"
    if "hek" in key:
        return "HEK293"
    if "bhk" in key:
        return "BHK21"
    if "mdck" in key:
        return "MDCK"
    if "cerevisiae" in key or "yeast" in key:
        return "S.cerevisiae"
    if "subtilis" in key:
        return "B.subtilis"
    return raw.strip()


def get_density_unit(group: str) -> str:
    return GROUP_DENSITY_UNIT.get(group, "unid.")


def get_density_scale(group: str) -> float:
    return GROUP_DENSITY_SCALE.get(group, 1.0)


ALL_GROUPS = sorted(set(_STRAIN_TO_GROUP.values()))
