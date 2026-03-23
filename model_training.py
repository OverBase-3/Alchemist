import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.base import BaseEstimator

try:
    from xgboost import XGBRegressor
    XGB_OK = True
except ImportError:
    XGB_OK = False

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, targets: List[str], test_size: float = 0.20,
                 n_folds: int = 5, random_state: int = 42):
        self.targets      = [t for t in targets if t != "doubling_time"]
        self.test_size    = test_size
        self.n_folds      = n_folds
        self.random_state = random_state
        self.results:     Dict = {}
        self.best_model_name   = None
        self.best_model        = None
        self.feature_names:    List[str] = []
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ModelTrainer":
        self.feature_names = list(X.columns)
        y_f = y[[t for t in self.targets if t in y.columns]]
        if len(X) <= 6:
            self.X_train = self.X_test = X
            self.y_train = self.y_test = y_f
        else:
            (self.X_train, self.X_test,
             self.y_train, self.y_test) = train_test_split(
                X, y_f, test_size=self.test_size, random_state=self.random_state)
        for name, model in self._models().items():
            self._fit_one(name, model, y_f)
        best = max(self.results.items(),
                   key=lambda kv: kv[1]["test_metrics"]["r2_mean"])
        self.best_model_name = best[0]
        self.best_model      = best[1]["model"]
        logger.info(f"  ★ {self.best_model_name} R²={best[1]['test_metrics']['r2_mean']:.4f}")
        return self

    def _models(self) -> Dict:
        m = {
            "GradientBoosting": MultiOutputRegressor(
                GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, random_state=self.random_state)),
            "RandomForest": MultiOutputRegressor(
                RandomForestRegressor(n_estimators=200, max_depth=8,
                                      random_state=self.random_state)),
            "MLP_NeuralNet": MultiOutputRegressor(
                MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.01,
                             max_iter=1000, random_state=self.random_state,
                             early_stopping=True, validation_fraction=0.15)),
            "GaussianProcess": MultiOutputRegressor(
                GaussianProcessRegressor(kernel=1.0*RBF(1.0)+WhiteKernel(0.1),
                                          n_restarts_optimizer=3,
                                          random_state=self.random_state)),
        }
        if XGB_OK:
            m["XGBoost"] = MultiOutputRegressor(
                XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             random_state=self.random_state, verbosity=0))
        return m

    def _fit_one(self, name: str, model: BaseEstimator, y_full: pd.DataFrame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
        yp = pd.DataFrame(model.predict(self.X_test),
                          columns=list(y_full.columns), index=self.y_test.index)
        tm = self._metrics(self.y_test, yp)
        cv_scores = []
        kf = KFold(n_splits=min(self.n_folds, len(self.X_train)),
                   shuffle=True, random_state=self.random_state)
        for t in y_full.columns:
            try:
                est = (model.estimators_[list(y_full.columns).index(t)]
                       if hasattr(model, "estimators_") else model)
                cv_scores.append(float(np.mean(
                    cross_val_score(est, self.X_train, self.y_train[t],
                                    cv=kf, scoring="r2"))))
            except Exception:
                cv_scores.append(np.nan)
        mdi  = self._mdi(model)
        perm = self._perm(model)
        fi   = perm if (perm is not None and perm.std() > 0.002) else mdi
        self.results[name] = {
            "model": model, "test_metrics": tm,
            "cv_r2_mean": float(np.nanmean(cv_scores)),
            "y_pred_test": yp, "feature_importance": fi,
        }
        logger.info(f"  {name:20s} R²={tm['r2_mean']:.4f} "
                    f"RMSE={tm['rmse_mean']:.4f} CV={float(np.nanmean(cv_scores)):.4f}")

    def _metrics(self, yt: pd.DataFrame, yp: pd.DataFrame) -> Dict:
        r2s, rmses, res = [], [], {}
        for col in yt.columns:
            if col not in yp.columns:
                continue
            a, b = yt[col].values, yp[col].values
            mask = ~(np.isnan(a) | np.isnan(b))
            if mask.sum() < 2:
                continue
            r2s.append(float(r2_score(a[mask], b[mask])))
            rmses.append(float(np.sqrt(mean_squared_error(a[mask], b[mask]))))
            res[f"r2_{col}"]   = round(r2s[-1], 4)
            res[f"rmse_{col}"] = round(rmses[-1], 4)
        res["r2_mean"]   = float(np.nanmean(r2s))   if r2s   else 0.0
        res["rmse_mean"] = float(np.nanmean(rmses)) if rmses else 1e6
        return res

    def _mdi(self, model) -> Optional[pd.Series]:
        try:
            if hasattr(model, "estimators_"):
                imps = np.mean([e.feature_importances_ for e in model.estimators_
                                if hasattr(e, "feature_importances_")], axis=0)
            elif hasattr(model, "coefs_"):
                imps = np.abs(model.coefs_[0]).mean(axis=1)
            else:
                return None
            if len(imps) == len(self.feature_names):
                return pd.Series(imps, index=self.feature_names).sort_values(ascending=False)
        except Exception:
            pass
        return None

    def _perm(self, model) -> Optional[pd.Series]:
        try:
            if self.X_test is None or len(self.X_test) < 4:
                return None
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, self.X_test, self.y_test,
                                        n_repeats=15, random_state=42, scoring="r2")
            fi = pd.Series(r.importances_mean, index=self.feature_names)
            return fi.sort_values(ascending=False) if fi.std() > 1e-6 else None
        except Exception:
            return None

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.best_model.predict(X)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        return pd.DataFrame(p, index=X.index, columns=self.targets)

    def get_results_table(self) -> pd.DataFrame:
        rows = [{"model": n, "r2_test": r["test_metrics"]["r2_mean"],
                 "rmse_test": r["test_metrics"]["rmse_mean"],
                 "cv_r2_mean": r.get("cv_r2_mean", np.nan),
                 "is_best": n == self.best_model_name,
                 **{k: v for k, v in r["test_metrics"].items() if k.startswith("r2_")}}
                for n, r in self.results.items()]
        return pd.DataFrame(rows).sort_values("r2_test", ascending=False)
