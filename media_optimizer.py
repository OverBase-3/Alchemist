from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from bio_constraints import (
    get_config,
    enforce_env_bounds,
    evaluate_candidate,
    inject_micronutrients,
    calculate_osmolarity,
    cdm_ceiling,
    redundancy_penalty,
)

logger = logging.getLogger(__name__)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

BIO_PENALTY_CUTOFF = 0.70  

@dataclass
class MediaFormulation:
    rank:                    int
    composition:             Dict[str, float]
    conditions:              Dict[str, float]
    cell_type:               str
    predicted_density:       float         = 0.0
    predicted_growth:        float         = 0.0
    predicted_doubling_time: float         = 0.0
    optimization_score:      float         = 0.0
    bio_penalty:             float         = 0.0
    pass_number:             int           = 1
    method:                  str           = "random"
    n_components:            int           = 0
    advertencias:            List[dict]    = field(default_factory=list)
    stoich_detail:           Dict          = field(default_factory=dict)
    osm_detail:              Dict          = field(default_factory=dict)
    cross_detail:            Dict          = field(default_factory=dict)
    redund_detail:           List[dict]    = field(default_factory=list)


class MediaOptimizer:


    MAX_COMPONENTS = 12

    def __init__(self, trainer, fe, exp_data: pd.DataFrame,
                 n_top: int = 10):
        self.trainer   = trainer
        self.fe        = fe
        self.exp_data  = exp_data
        self.n_top     = n_top
        self.group     = "E.coli"


        grp = exp_data.drop_duplicates("medium_id")
        self._mcd_p90  = float(np.percentile(grp["max_cell_density"], 90))
        self._gr_max   = float(grp["growth_rate"].max())


        self._comp_bounds = self._build_component_bounds(exp_data)

        logger.info(
            f"  CDM p90 = {self._mcd_p90:.2f} g/L  |  "
            f"µ max = {self._gr_max:.4f} h⁻¹"
        )

    def _build_component_bounds(self, exp: pd.DataFrame) -> Dict[str, Tuple[float,float]]:

        bounds = {}
        for comp, grp in exp.groupby("component"):
            lo = float(grp["concentration"].min())
            hi = float(grp["concentration"].max()) * 1.05
            if hi > lo:
                bounds[comp] = (max(lo, 0.0), hi)
        return bounds

    def _get_candidate_components(self, freq_threshold: float) -> List[str]:

        return self.fe.get_relevant_components(self.group, freq_threshold)

    def _build_conditions(self) -> Dict[str, float]:

        cfg = get_config(self.group)["condiciones_entorno"]
        return {
            "temperature_C":  cfg["temperature_C"]["optimo"],
            "pH":             cfg["pH"]["optimo"],
            "agitation_rpm":  (cfg["agitation_rpm"]["min"] +
                               cfg["agitation_rpm"]["max"]) / 2,
            "carriers":       0.0,
        }
    def _predict(self, comp: Dict[str, float],
                  conds: Dict[str, float]) -> Tuple[float, float, pd.Series]:

        row = {**{c: v for c, v in comp.items()}, **conds,
               "cell_type": self.group}
        row_df = pd.DataFrame([row])
        row_df["medium_id"]        = "__opt__"
        row_df["component"]        = "__dummy__"
        row_df["concentration"]    = 0.0
        row_df["unit"]             = "g/L"
        row_df["growth_rate"]      = 0.0
        row_df["max_cell_density"] = 0.0
        row_df["doubling_time"]    = 0.0

        comp_rows = []
        for c, v in comp.items():
            comp_rows.append({
                "medium_id": "__opt__", "cell_type": self.group,
                "component": c, "concentration": v, "unit": "g/L",
                "growth_rate": 0.0, "max_cell_density": 0.0, "doubling_time": 0.0,
            })
        exp_mini = pd.DataFrame(comp_rows)
        cc_mini  = pd.DataFrame([{"medium_id": "__opt__", **conds}])

        try:
            X      = self.fe.transform(exp_mini, cc_mini)
            preds  = self.trainer.predict(X)
            mu_raw = float(np.clip(preds.get("growth_rate", pd.Series([0])).iloc[0], 0, None))
            cdm_raw= float(np.clip(preds.get("max_cell_density", pd.Series([0])).iloc[0], 0, None))

            if cdm_raw < 0.01:
                for m_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
                    if m_name in self.trainer.results:
                        rf  = self.trainer.results[m_name]["model"]
                        rfp = rf.predict(X)
                        targets = list(self.trainer.targets)
                        if "max_cell_density" in targets:
                            idx = targets.index("max_cell_density")
                            cdm_raw = float(np.clip(
                                rfp[0, idx] if rfp.ndim > 1 else rfp[0], 0, None))
                        if cdm_raw > 0.01:
                            break

            return cdm_raw, mu_raw, preds

        except Exception as e:
            logger.debug(f"Error en _predict: {e}")
            return 0.0, 0.0, pd.Series()

    def _score(self, comp: Dict[str, float],
                conds: Dict[str, float]) -> Tuple[float, float, float, float, List[dict]]:

        cdm, mu, _ = self._predict(comp, conds)
        import math as _math
        dt  = float(_math.log(2) / mu) if mu > 1e-6 else 99.0

        if cdm < 0.01:
            return -50.0, cdm, mu, 1.0, []

        score_ml = (0.60 * cdm / max(self._mcd_p90, 1e-9)
                  + 0.30 * mu  / max(self._gr_max,  1e-9)
                  + 0.10 * (1.0 / max(dt, 0.1)))

        bio_penalty, advertencias, _ = evaluate_candidate(
            composition          = comp,
            conditions           = conds,
            group                = self.group,
            target_cdm           = cdm,
            available_components = list(self.fe.component_columns),
            predicted_mu         = mu,
        )

        if bio_penalty >= BIO_PENALTY_CUTOFF:
            return -10.0 * bio_penalty, cdm, mu, bio_penalty, advertencias

        return score_ml * (1.0 - bio_penalty), cdm, mu, bio_penalty, advertencias


    def _random_candidate(self, components: List[str],
                           rng: np.random.Generator) -> Dict[str, float]:
        n = rng.integers(3, min(self.MAX_COMPONENTS, len(components)) + 1)
        chosen = rng.choice(components, size=int(n), replace=False)
        comp = {}
        for c in chosen:
            lo, hi = self._comp_bounds.get(c, (0.1, 5.0))
            comp[c] = float(rng.uniform(lo, hi))
        return comp

    def _random_conditions(self, rng: np.random.Generator) -> Dict[str, float]:
        cfg = get_config(self.group)["condiciones_entorno"]
        return {
            "temperature_C": float(rng.uniform(
                cfg["temperature_C"]["min"], cfg["temperature_C"]["max"])),
            "pH": float(rng.uniform(
                cfg["pH"]["min"], cfg["pH"]["max"])),
            "agitation_rpm": float(rng.uniform(
                cfg["agitation_rpm"]["min"], cfg["agitation_rpm"]["max"])),
            "carriers": 0.0,
        }

    def _optuna_study(self, components: List[str],
                       n_trials: int) -> List[Tuple[float, Dict, Dict]]:
        cfg = get_config(self.group)["condiciones_entorno"]
        results = []

        def objective(trial):
            # Componentes activos
            n_comp = trial.suggest_int("n_comp", 3, min(self.MAX_COMPONENTS, len(components)))
            active = []
            for i, c in enumerate(components):
                if len(active) < n_comp:
                    if trial.suggest_categorical(f"use_{i}", [True, False]):
                        active.append(c)
            if len(active) < 3:
                active = components[:3]

            comp = {}
            for c in active:
                lo, hi = self._comp_bounds.get(c, (0.1, 5.0))
                comp[c] = trial.suggest_float(f"c_{c}", lo, hi)

            conds = {
                "temperature_C": trial.suggest_float(
                    "temperature_C",
                    cfg["temperature_C"]["min"], cfg["temperature_C"]["max"]),
                "pH": trial.suggest_float(
                    "pH", cfg["pH"]["min"], cfg["pH"]["max"]),
                "agitation_rpm": trial.suggest_float(
                    "agitation_rpm",
                    cfg["agitation_rpm"]["min"], cfg["agitation_rpm"]["max"]),
                "carriers": 0.0,
            }

            sc, _, _, _, _ = self._score(comp, conds)
            results.append((sc, comp, conds))
            return sc

        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return results

    def _random_search(self, components: List[str],
                        n_trials: int) -> List[Tuple[float, Dict, Dict]]:
        rng     = np.random.default_rng(42)
        results = []
        for _ in range(n_trials):
            comp  = self._random_candidate(components, rng)
            conds = self._random_conditions(rng)
            sc, _, _, _, _ = self._score(comp, conds)
            results.append((sc, comp, conds))
        return results

    def optimize(self, n_trials: int = 200) -> List[MediaFormulation]:

        import math as _math

        cfg_filtro = get_config(self.group)["filtro1_frecuencia"]
        all_results = []

        for pass_n, threshold in enumerate(
            [cfg_filtro["umbral_P1"], cfg_filtro["umbral_P2"]], start=1
        ):
            components = self._get_candidate_components(threshold)
            if not components:
                logger.warning(f"  Pase {pass_n}: sin componentes con umbral {threshold}")
                continue

            logger.info(
                f"  Pase {pass_n}: {len(components)} componentes "
                f"(freq >= {threshold}) | n_trials={n_trials}"
            )

            if OPTUNA_OK:
                raw = self._optuna_study(components, n_trials)
            else:
                raw = self._random_search(components, n_trials)

            for sc, comp, conds in raw:
                if sc < -5:
                    continue

                avail = list(self.fe.component_columns)
                comp_enriq, _ = inject_micronutrients(comp, avail, self.group)
                cdm, mu, _    = self._predict(comp_enriq, conds)
                bio_pen, advertencias, report = evaluate_candidate(
                    composition          = comp_enriq,
                    conditions           = conds,
                    group                = self.group,
                    target_cdm           = cdm,
                    available_components = avail,
                    predicted_mu         = mu,
                )

                sc_final, _, _, _, _ = self._score(comp_enriq, report.get("corrected_conds", conds))
                dt = float(_math.log(2) / mu) if mu > 1e-6 else 99.0

                all_results.append(MediaFormulation(
                    rank                    = 0,
                    composition             = comp_enriq,
                    conditions              = report.get("corrected_conds", conds),
                    cell_type               = self.group,
                    predicted_density       = round(cdm, 4),
                    predicted_growth        = round(mu, 5),
                    predicted_doubling_time = round(dt, 3),
                    optimization_score      = round(sc_final, 4),
                    bio_penalty             = round(bio_pen, 4),
                    pass_number             = pass_n,
                    method                  = "bayesian" if OPTUNA_OK else "random",
                    n_components            = len(comp_enriq),
                    advertencias            = advertencias,
                    stoich_detail           = report.get("stoichiometry", {}),
                    osm_detail              = report.get("osm_detail", {}),
                    cross_detail            = report.get("cross_detail", {}),
                    redund_detail           = report.get("redund_detail", []),
                ))

        all_results.sort(key=lambda f: -f.optimization_score)
        seen  = set()
        top   = []
        for f in all_results:
            key = tuple(sorted((k, round(v, 2)) for k, v in f.composition.items()))
            if key not in seen:
                seen.add(key)
                top.append(f)
            if len(top) >= self.n_top:
                break

        for i, f in enumerate(top, 1):
            f.rank = i

        return top
    def get_filter_report(self, forms: List[MediaFormulation]) -> pd.DataFrame:
        rows = []
        for f in forms:
            for adv in f.advertencias:
                rows.append({
                    "rank":      f.rank,
                    "pass":      f.pass_number,
                    "cell_type": f.cell_type,
                    "capa":      adv.get("capa", ""),
                    "variable":  adv.get("variable", adv.get("componente", "")),
                    "mensaje":   adv.get("mensaje", ""),
                })

            for rd in getattr(f, "redund_detail", []):
                rows.append({
                    "rank":      f.rank,
                    "pass":      f.pass_number,
                    "cell_type": f.cell_type,
                    "capa":      "Capa 6 — Redundancia funcional",
                    "variable":  rd.get("categoria", ""),
                    "mensaje":   (f"Categoria '{rd.get('categoria','')}': "
                                  f"{rd.get('n_activos',0)} componentes activos "
                                  f"(max: {rd.get('max_permitido',1)}). "
                                  f"Redundante(s): {rd.get('redundantes',[])}"),
                })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["rank","pass","cell_type","capa","variable","mensaje"])
