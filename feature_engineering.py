import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from bio_groups import normalize_cell_type

logger = logging.getLogger(__name__)

ENV_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature_C":  (25.0, 42.0),
    "pH":             (5.5,  8.5),
    "carriers":       (0.0,  1.0),
    "agitation_rpm":  (0.0, 300.0),
}
ENV_DEFAULTS: Dict[str, float] = {
    "temperature_C": 37.0, "pH": 7.0,
    "carriers": 0.0, "agitation_rpm": 150.0,
}

CARBON_SOURCES    = ["glucose", "glycerol", "sucrose", "lactose", "galactose",
                     "sodium_acetate", "sodium_citrate", "sodium_pyruvate",
                     "yeast_extract", "tryptone", "casamino_acids",
                     "peptone_from_casein", "meat_extract",
                     "cheese_whey", "cereal_flour_mix"]
NITROGEN_SOURCES  = ["ammonium_sulfate", "ammonium_chloride", "urea",
                     "tryptone", "yeast_extract", "casamino_acids",
                     "peptone_from_casein", "meat_extract", "l_glutamine"]
PHOSPHATE_SOURCES = ["kh2po4", "k2hpo4", "na2hpo4", "nah2po4"]
SALTS             = ["nacl", "kcl", "kh2po4", "k2hpo4", "na2hpo4",
                     "mgso4", "mgcl2", "cacl2", "ammonium_sulfate",
                     "ammonium_chloride", "sodium_citrate"]
COMPLEX_COMPS     = ["yeast_extract", "tryptone", "casamino_acids",
                     "meat_extract", "peptone_from_casein",
                     "cheese_whey", "cereal_flour_mix"]
GROWTH_FACTORS    = ["fbs", "egf", "insulin", "transferrin", "plant_hydrolysate"]


class FeatureEngineer:
    def __init__(self, targets: List[str] = None, freq_threshold: float = 0.30):
        self.targets              = targets or ["growth_rate", "max_cell_density"]
        self.freq_threshold       = freq_threshold
        self.component_columns:   List[str] = []
        self.env_columns:         List[str] = list(ENV_BOUNDS.keys())
        self.cell_type_categories: List[str] = []
        self.scaler               = StandardScaler()
        self.imputer              = SimpleImputer(strategy="constant", fill_value=0.0)
        self.feature_names:       List[str] = []
        self.is_fitted:           bool = False
        self._units_map:          Dict[str, str] = {}
        self._freq_by_group:      Dict[str, Dict[str, float]] = {}
        self._comp_ranges:        Dict[str, Dict[str, Tuple]] = {}

    def compute_frequency_profile(self, df: pd.DataFrame) -> Dict:
        tmp = df.copy()
        tmp["_g"] = tmp["cell_type"].apply(normalize_cell_type)
        for group in tmp["_g"].unique():
            sub    = tmp[tmp["_g"] == group]
            n      = sub["medium_id"].nunique()
            freqs, rng = {}, {}
            for comp in sub["component"].unique():
                nw = sub[sub["component"] == comp]["medium_id"].nunique()
                freqs[comp] = nw / n if n > 0 else 0.0
                vals = pd.to_numeric(
                    sub[sub["component"] == comp]["concentration"], errors="coerce"
                ).dropna()
                vals = vals[vals > 0]
                if len(vals):
                    rng[comp] = (float(vals.min()), float(vals.max()))
            self._freq_by_group[group] = freqs
            self._comp_ranges[group]   = rng
        return self._freq_by_group

    def get_relevant_components(self, group: str,
                                 threshold: Optional[float] = None) -> List[str]:
        thr   = threshold if threshold is not None else self.freq_threshold
        freqs = self._freq_by_group.get(group, {})
        if not freqs:
            return []
        result = [c for c, f in freqs.items() if f >= thr]
        while len(result) < 5 and thr > 0.05:
            thr   -= 0.05
            result = [c for c, f in freqs.items() if f >= thr]
        return sorted(result) if result else sorted(freqs.keys())

    def fit_transform(self, experiment_data: pd.DataFrame,
                      culture_conditions: Optional[pd.DataFrame] = None
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.compute_frequency_profile(experiment_data)
        y            = self._targets(experiment_data)
        media_matrix = self._pivot(experiment_data)
        cell_dummies = self._encode_ct(experiment_data)
        env_matrix   = self._encode_env(experiment_data, culture_conditions)
        X_raw        = media_matrix.join(cell_dummies, how="left").join(env_matrix, how="left")
        X_derived    = self._derived(X_raw)
        X_full       = pd.concat([X_raw, X_derived], axis=1)
        X_full       = X_full.loc[:, ~X_full.columns.duplicated()]
        X_imp        = pd.DataFrame(self.imputer.fit_transform(X_full),
                                    index=X_full.index, columns=X_full.columns)
        X_s          = self._scale(X_imp, fit=True)
        common       = X_s.index.intersection(y.index)
        X_s, y       = X_s.loc[common], y.loc[common]
        self.feature_names = list(X_s.columns)
        self._units_map    = self._build_units(experiment_data)
        self.is_fitted     = True
        logger.info(f"  X: {X_s.shape} | y: {y.shape} | "
                    f"comps: {len(self.component_columns)} | "
                    f"env: {len(self.env_columns)}")
        return X_s, y

    def transform(self, experiment_data: pd.DataFrame,
                  culture_conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Llama fit_transform primero.")
        media_matrix = self._pivot_inference(experiment_data)
        cell_dummies = self._encode_ct_inference(experiment_data)
        env_matrix   = self._encode_env(experiment_data, culture_conditions)
        X_raw        = media_matrix.join(cell_dummies, how="left").join(env_matrix, how="left")
        X_derived    = self._derived(X_raw)
        X_full       = pd.concat([X_raw, X_derived], axis=1)
        X_full       = X_full.loc[:, ~X_full.columns.duplicated()]
        for col in self.feature_names:
            if col not in X_full.columns:
                X_full[col] = 0.0
        X_full = X_full[self.feature_names]
        X_imp  = pd.DataFrame(self.imputer.transform(X_full),
                               index=X_full.index, columns=X_full.columns)
        return self._scale(X_imp, fit=False)

    def medium_to_vector(self, composition: Dict[str, float], cell_type: str,
                          conditions: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Llama fit_transform primero.")
        conds = {k: (conditions or {}).get(k, ENV_DEFAULTS[k]) for k in ENV_BOUNDS}
        rows = []
        for k, v in composition.items():
            norm = k.lower().replace(" ", "_").replace("-", "_")
            if norm in self.component_columns:
                rows.append({"medium_id": "NEW", "cell_type": cell_type,
                              "component": norm, "concentration": float(v),
                              "unit": "custom", "growth_rate": 0.0,
                              "max_cell_density": 0.0})
        if not rows:
            rows = [{"medium_id": "NEW", "cell_type": cell_type,
                     "component": self.component_columns[0] if self.component_columns else "glucose",
                     "concentration": 0.0, "unit": "custom",
                     "growth_rate": 0.0, "max_cell_density": 0.0}]
        return self.transform(pd.DataFrame(rows),
                              pd.DataFrame([{"medium_id": "NEW", **conds}]))

    def _pivot_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pivot para inferencia: NO muta component_columns. Usa las cols del entrenamiento."""
        p = df.pivot_table(index="medium_id", columns="component",
                           values="concentration", aggfunc="mean").fillna(0.0)
        p.columns.name = None
        for col in self.component_columns:
            if col not in p.columns:
                p[col] = 0.0
        return p[self.component_columns] 

    def _encode_ct_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encoding para inferencia: NO muta cell_type_categories."""
        ct = (df[["medium_id", "cell_type"]].drop_duplicates("medium_id")
                .set_index("medium_id"))
        dummies = pd.get_dummies(ct["cell_type"], prefix="celltype", dtype=float)
        for cat in self.cell_type_categories:
            col = f"celltype_{cat}"
            if col not in dummies.columns:
                dummies[col] = 0.0
        return dummies

    def _targets(self, df: pd.DataFrame) -> pd.DataFrame:
        avail = [t for t in self.targets if t in df.columns]
        return (df[["medium_id"] + avail]
                .drop_duplicates("medium_id")
                .set_index("medium_id"))

    def _pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        p = df.pivot_table(index="medium_id", columns="component",
                           values="concentration", aggfunc="mean").fillna(0.0)
        p.columns.name = None
        self.component_columns = list(p.columns)
        return p

    def _encode_ct(self, df: pd.DataFrame) -> pd.DataFrame:
        ct = (df[["medium_id", "cell_type"]].drop_duplicates("medium_id")
                .set_index("medium_id"))
        self.cell_type_categories = sorted(ct["cell_type"].unique().tolist())
        return pd.get_dummies(ct["cell_type"], prefix="celltype", dtype=float)

    def _encode_env(self, df: pd.DataFrame,
                     cc: Optional[pd.DataFrame]) -> pd.DataFrame:
        idx = df["medium_id"].unique()
        if cc is None or cc.empty:
            return pd.DataFrame(ENV_DEFAULTS, index=idx)
        env_cols = [c for c in ENV_BOUNDS if c in cc.columns]
        env = (cc[["medium_id"] + env_cols].drop_duplicates("medium_id")
               .set_index("medium_id").reindex(idx))
        for col, val in ENV_DEFAULTS.items():
            env[col] = env.get(col, pd.Series(dtype=float)).fillna(val)
        return env[list(ENV_BOUNDS.keys())]

    def _derived(self, X: pd.DataFrame) -> pd.DataFrame:
        def _s(cols):
            c = [c for c in cols if c in X.columns]
            return X[c].sum(axis=1) if c else pd.Series(0.0, index=X.index)
        d = pd.DataFrame(index=X.index)
        d["carbon_source_total"]     = _s(CARBON_SOURCES)
        d["nitrogen_source_total"]   = _s(NITROGEN_SOURCES)
        d["phosphate_total"]         = _s(PHOSPHATE_SOURCES)
        d["salt_osmolarity_proxy"]   = _s(SALTS) * 2.0
        d["complex_component_total"] = _s(COMPLEX_COMPS)
        d["growth_factor_total"]     = _s(GROWTH_FACTORS)
        n = _s(NITROGEN_SOURCES).replace(0, np.nan)
        d["carbon_nitrogen_ratio"] = (X["glucose"] / n).fillna(0.0) \
                                      if "glucose" in X.columns else 0.0
        return d

    def _scale(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        v = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return pd.DataFrame(v, index=X.index, columns=X.columns)

    def _build_units(self, df: pd.DataFrame) -> Dict[str, str]:
        if "unit" not in df.columns:
            return {}
        return (df.groupby("component")["unit"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "g/L")
                .to_dict())
