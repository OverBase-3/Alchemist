
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "medium_id", "cell_type", "component",
    "concentration", "unit", "growth_rate", "max_cell_density",
]

ENV_DEFAULTS = {
    "temperature_C": 37.0,
    "pH":            7.0,
    "carriers":      0.0,
    "agitation_rpm": 150.0,
}


class DataLoader:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath            = Path(filepath) if filepath else None
        self.experiment_data:    Optional[pd.DataFrame] = None
        self.growth_curve_data:  Optional[pd.DataFrame] = None
        self.culture_conditions: Optional[pd.DataFrame] = None

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not (self.filepath and self.filepath.exists()):
            raise FileNotFoundError(
                f"Archivo no encontrado: {self.filepath}\n"
            )
        self._load_excel()
        self._validate()
        self._clean()
        s = self.get_summary()
        logger.info(
            f"Datos cargados → {s['n_media']} medios | "
            f"{s['n_cell_types']} tipos celulares | "
            f"{s['n_components']} componentes"
        )
        return self.experiment_data, self.growth_curve_data, self.culture_conditions

    def _load_excel(self):
        xl     = pd.ExcelFile(self.filepath)
        sheets = xl.sheet_names
        if "experiment_data" not in sheets:
            raise KeyError(
                "Hoja 'experiment_data' no encontrada. "
                f"Hojas disponibles: {sheets}"
            )
        self.experiment_data = xl.parse("experiment_data")

        if "growth_curve" in sheets:
            self.growth_curve_data = xl.parse("growth_curve")
        else:
            logger.warning("Hoja 'growth_curve' no encontrada. Se usará vacía.")
            self.growth_curve_data = pd.DataFrame(
                columns=["medium_id", "time_h", "cell_density"]
            )

        if "culture_conditions" in sheets:
            cc = xl.parse("culture_conditions")
            if "carriers" in cc.columns:
                cc["carriers"] = (cc["carriers"].astype(str).str.strip().str.lower()
                                    .map({"yes": 1.0, "no": 0.0}).fillna(0.0))
            self.culture_conditions = cc
        else:
            logger.warning(
                "Hoja 'culture_conditions' no encontrada. "
                "Se usarán valores por defecto."
            )
            self.culture_conditions = self._default_conditions()

    def _default_conditions(self) -> pd.DataFrame:
        media = self.experiment_data["medium_id"].unique()
        return pd.DataFrame([{"medium_id": m, **ENV_DEFAULTS} for m in media])

    def _validate(self):
        for col in REQUIRED_COLS:
            if col not in self.experiment_data.columns:
                raise ValueError(
                    f"Columna obligatoria '{col}' no encontrada. "
                    f"Columnas disponibles: {list(self.experiment_data.columns)}"
                )

    @staticmethod
    def _to_float(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s.astype(float)
        return (s.astype(str).str.strip()
                 .str.replace(",", ".", regex=False)
                 .pipe(pd.to_numeric, errors="coerce"))

    def _clean(self):
        df = self.experiment_data
        df["component"] = (df["component"].astype(str).str.strip().str.lower()
                            .str.replace(r"[\s\-]+", "_", regex=True)
                            .str.replace(r"[^\w]", "", regex=True))
        df["cell_type"] = df["cell_type"].astype(str).str.strip()
        df["medium_id"] = df["medium_id"].astype(str).str.strip()
        df["unit"]      = df["unit"].astype(str).str.strip()
        df["strain"]    = df["strain"].astype(str).str.strip() if "strain" in df.columns else ""
        df["source"]    = df["source"].astype(str).str.strip() if "source" in df.columns else ""

        for col in ["concentration", "growth_rate", "max_cell_density", "doubling_time"]:
            if col in df.columns:
                df[col] = self._to_float(df[col])

        if "doubling_time" not in df.columns:
            df["doubling_time"] = np.log(2) / df["growth_rate"].replace(0, np.nan)
        else:
            mask = df["doubling_time"].isna() | (df["doubling_time"] <= 0)
            df.loc[mask, "doubling_time"] = (
                np.log(2) / df.loc[mask, "growth_rate"].replace(0, np.nan)
            )

        before = len(df)
        df.dropna(subset=[c for c in REQUIRED_COLS if c in df.columns], inplace=True)
        df.drop_duplicates(inplace=True)
        if before - len(df) > 0:
            logger.info(f"  Limpieza: {before - len(df)} filas eliminadas.")
        self.experiment_data = df

        if not self.growth_curve_data.empty:
            for col in ["time_h", "cell_density"]:
                if col in self.growth_curve_data.columns:
                    self.growth_curve_data[col] = self._to_float(
                        self.growth_curve_data[col]
                    )
            self.growth_curve_data.dropna(inplace=True)
            self.growth_curve_data = self.growth_curve_data[
                self.growth_curve_data["cell_density"] > 0
            ]

        all_media = df["medium_id"].unique()
        cc        = self.culture_conditions
        missing   = set(all_media) - set(cc["medium_id"].astype(str).values)
        if missing:
            for col in ENV_DEFAULTS:
                if col not in cc.columns:
                    cc[col] = ENV_DEFAULTS[col]
            extra = pd.DataFrame([{"medium_id": m, **ENV_DEFAULTS} for m in missing])
            self.culture_conditions = pd.concat([cc, extra], ignore_index=True)

    def get_summary(self) -> Dict:
        df = self.experiment_data
        return {
            "n_media":         df["medium_id"].nunique(),
            "n_cell_types":    df["cell_type"].nunique(),
            "cell_types":      df["cell_type"].unique().tolist(),
            "n_components":    df["component"].nunique(),
            "has_strain":      "strain" in df.columns and df["strain"].ne("").any(),
            "has_conditions":  not self.culture_conditions.empty,
            "has_growth_curve":not self.growth_curve_data.empty,
        }
