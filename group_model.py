from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional

from bio_groups import normalize_cell_type
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

logger = logging.getLogger(__name__)
_TARGETS = ["growth_rate", "max_cell_density"]


class GroupModelSystem:
    def __init__(self):
        self.trainers:            Dict[str, ModelTrainer]    = {}
        self.feature_engineers:   Dict[str, FeatureEngineer] = {}
        self.exp_data_by_group:   Dict[str, pd.DataFrame]    = {}
        self.conditions_by_group: Dict[str, pd.DataFrame]    = {}
        self._is_fitted = False

    def fit(self, experiment_data: pd.DataFrame,
            culture_conditions: Optional[pd.DataFrame] = None,
            targets: Optional[List[str]] = None,
            test_size: float = 0.20, random_state: int = 42,
            log_fn=None) -> "GroupModelSystem":
        targets = [t for t in (targets or _TARGETS) if t != "doubling_time"]

        def _log(msg, pct=None):
            logger.info(msg)
            if log_fn:
                log_fn(msg, pct)

        df = experiment_data.copy()
        df["cell_type"] = df["cell_type"].apply(normalize_cell_type)
        groups = sorted(df["cell_type"].unique())
        _log(f"Grupos: {groups}", 5)
        self.trainers.clear(); self.feature_engineers.clear()
        self.exp_data_by_group.clear(); self.conditions_by_group.clear()

        step = 80 / max(len(groups), 1)
        for i, group in enumerate(groups):
            sub = df[df["cell_type"] == group].copy()
            n   = sub["medium_id"].nunique()
            _log(f"── {group}: {n} medios ──", int(10 + i * step))
            if n < 3:
                _log(f"  ⚠ {group}: {n} medios — mínimo 3. Saltando.")
                continue
            cc = None
            if culture_conditions is not None and not culture_conditions.empty:
                cc = culture_conditions[
                    culture_conditions["medium_id"].isin(sub["medium_id"].unique())
                ].copy()
            fe = FeatureEngineer(targets=targets)
            try:
                X, y = fe.fit_transform(sub, cc)
            except Exception as e:
                _log(f"  ✗ FE: {e}"); continue
            trainer = ModelTrainer(targets=list(y.columns),
                                   test_size=test_size, random_state=random_state)
            try:
                trainer.fit(X, y)
            except Exception as e:
                _log(f"  ✗ Train: {e}"); continue
            self.trainers[group]            = trainer
            self.feature_engineers[group]   = fe
            self.exp_data_by_group[group]   = sub
            self.conditions_by_group[group] = cc if cc is not None else pd.DataFrame()
            r2 = trainer.get_results_table().iloc[0]["r2_test"]
            _log(f"  ✓ {group} | {trainer.best_model_name} | R²={r2:.4f}")
        self._is_fitted = bool(self.trainers)
        _log("Entrenamiento completado.", 95)
        return self

    def evaluate_within_group(self) -> pd.DataFrame:
        from sklearn.metrics import r2_score, mean_squared_error
        from sklearn.model_selection import LeaveOneOut
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import warnings as _w
        rows = []
        for group, trainer in self.trainers.items():
            n_total = self.exp_data_by_group[group]["medium_id"].nunique()
            row     = {"group": group, "n_total": n_total,
                       "best_model": trainer.best_model_name}
            y_test  = trainer.y_test
            if y_test is not None and len(y_test) >= 2:
                preds = trainer.predict(trainer.X_test)
                row["n_test"] = len(y_test)
                for t in ["growth_rate", "max_cell_density"]:
                    col = f"r2_test_{t.replace('max_cell_density','max_density')}"
                    if t in y_test.columns and t in preds.columns:
                        yt = y_test[t].values; yp = preds[t].values
                        mask = ~(np.isnan(yt) | np.isnan(yp))
                        if mask.sum() >= 2:
                            row[col] = round(float(r2_score(yt[mask], yp[mask])), 4)
            elif n_total >= 3:
                fe  = self.feature_engineers[group]
                sub = self.exp_data_by_group[group]
                cc  = self.conditions_by_group.get(group)
                try:
                    X_all, y_all = fe.fit_transform(sub, cc)
                    loo  = LeaveOneOut()
                    tr_, pr_ = {t: [] for t in y_all.columns}, \
                               {t: [] for t in y_all.columns}
                    for ti, tei in loo.split(X_all):
                        m = MultiOutputRegressor(
                            GradientBoostingRegressor(n_estimators=50, random_state=42))
                        with _w.catch_warnings():
                            _w.simplefilter("ignore")
                            m.fit(X_all.iloc[ti].values, y_all.iloc[ti].values)
                        pred = m.predict(X_all.iloc[tei].values)[0]
                        for idx, t in enumerate(y_all.columns):
                            tr_[t].append(float(y_all.iloc[tei][t].values[0]))
                            pr_[t].append(float(pred[idx]))
                    row["n_loo"] = len(X_all)
                    for t in ["growth_rate", "max_cell_density"]:
                        col = f"r2_loo_{t.replace('max_cell_density','max_density')}"
                        if t in tr_ and len(tr_[t]) >= 3:
                            row[col] = round(float(r2_score(
                                np.array(tr_[t]), np.array(pr_[t]))), 4)
                except Exception as e:
                    logger.debug(f"LOO error {group}: {e}")
            r2s = [row.get(k) for k in
                   ["r2_test_growth_rate","r2_test_max_density",
                    "r2_loo_growth_rate", "r2_loo_max_density"]
                   if row.get(k) is not None]
            avg = float(np.mean(r2s)) if r2s else None
            row["interpretation"] = (
                "Excelente (≥0.85)" if avg and avg >= 0.85 else
                "Bueno (0.70-0.85)"  if avg and avg >= 0.70 else
                "Moderado (0.50-0.70)" if avg and avg >= 0.50 else
                "Débil (<0.50)"       if avg is not None else
                "Insuficiente"
            )
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_trainer(self, ct: str) -> ModelTrainer:
        g = normalize_cell_type(ct)
        if g not in self.trainers:
            raise KeyError(f"Grupo '{g}' sin modelo. Disponibles: {list(self.trainers)}")
        return self.trainers[g]

    def get_fe(self, ct: str) -> FeatureEngineer:
        g = normalize_cell_type(ct)
        if g not in self.feature_engineers:
            raise KeyError(f"Grupo '{g}' sin FE.")
        return self.feature_engineers[g]

    def get_all_results(self) -> pd.DataFrame:
        frames = []
        for g, tr in self.trainers.items():
            t = tr.get_results_table().copy(); t.insert(0, "group", g)
            frames.append(t)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @property
    def groups(self) -> List[str]:
        return list(self.trainers.keys())

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def save(self, path: str = "group_model.pkl"):
        joblib.dump({"trainers": self.trainers,
                     "feature_engineers": self.feature_engineers,
                     "exp_data_by_group": self.exp_data_by_group,
                     "conditions_by_group": self.conditions_by_group}, path)
        logger.info(f"Modelo guardado: {path}")

    @classmethod
    def load(cls, path: str = "group_model.pkl") -> "GroupModelSystem":
        if not Path(path).exists():
            raise FileNotFoundError(f"No se encontró: {path}")
        b   = joblib.load(path)
        gms = cls()
        gms.trainers            = b["trainers"]
        gms.feature_engineers   = b["feature_engineers"]
        gms.exp_data_by_group   = b["exp_data_by_group"]
        gms.conditions_by_group = b.get("conditions_by_group", {})
        gms._is_fitted = bool(gms.trainers)
        return gms
