
from __future__ import annotations
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
# Carga del JSON

def _load_config(group: str = "E.coli") -> dict:
    config_dir = Path(__file__).parent / "config"
    filename_map = {
        "E.coli":     "bio_config_ecoli.json",
        "Lactobacillus": "bio_config_ecoli.json",  
    }
    filename = filename_map.get(group, "bio_config_ecoli.json")
    path = config_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo de configuracion no encontrado: {path}\n"
            f"Crea 'config/{filename}' o usa 'bio_config_ecoli.json' como plantilla."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)

_CONFIG_CACHE: Dict[str, dict] = {}


def get_config(group: str = "E.coli") -> dict:
    if group not in _CONFIG_CACHE:
        _CONFIG_CACHE[group] = _load_config(group)
    return _CONFIG_CACHE[group]


def reload_config(group: str = "E.coli") -> dict:
    """Fuerza la recarga del JSON desde disco, descartando la cache."""
    if group in _CONFIG_CACHE:
        del _CONFIG_CACHE[group]
    return get_config(group)


#Capa 1

def get_env_hard_bounds(group: str = "E.coli") -> Dict[str, Tuple[float, float]]:
    """Devuelve (min, max) por variable de entorno para el grupo."""
    cfg = get_config(group)["condiciones_entorno"]
    bounds = {}
    for var in ["temperature_C", "pH", "agitation_rpm"]:
        if var in cfg:
            bounds[var] = (cfg[var]["min"], cfg[var]["max"])
    carriers_cfg = cfg.get("carriers", {})
    if carriers_cfg.get("permitido", True):
        bounds["carriers"] = (0.0, 1.0)
    else:
        bounds["carriers"] = (0.0, 0.0)
    return bounds


def enforce_env_bounds(conds: Dict[str, float],
                        group: str = "E.coli") -> Tuple[Dict, List[dict]]:
    """
    Aplica limites duros de entorno.
    Retorna (condiciones_corregidas, lista_de_advertencias).
    Cada advertencia es un dict: {variable, valor_original, valor_corregido, mensaje}.
    """
    cfg    = get_config(group)["condiciones_entorno"]
    bounds = get_env_hard_bounds(group)
    corrected  = dict(conds)
    advertencias = []

    for var, (lo, hi) in bounds.items():
        if var not in corrected:
            continue
        val = corrected[var]
        if val < lo:
            msg_key = "advertencia_min" if var != "agitation_rpm" else "advertencia_min"
            var_cfg  = cfg.get(var, {})
            mensaje  = var_cfg.get("advertencia_min",
                                   f"{var} = {val:.2f} esta por debajo del minimo {lo:.2f}")
            advertencias.append({
                "variable":       var,
                "valor_original": val,
                "valor_corregido": lo,
                "capa": "Capa 1 — Entorno",
                "mensaje": mensaje
            })
            corrected[var] = lo
        elif val > hi:
            var_cfg = cfg.get(var, {})
            mensaje = var_cfg.get("advertencia_max",
                                   f"{var} = {val:.2f} supera el maximo {hi:.2f}")
            advertencias.append({
                "variable":       var,
                "valor_original": val,
                "valor_corregido": hi,
                "capa": "Capa 1 — Entorno",
                "mensaje": mensaje
            })
            corrected[var] = hi
    carriers_cfg = cfg.get("carriers", {})
    if not carriers_cfg.get("permitido", True) and corrected.get("carriers", 0) > 0:
        advertencias.append({
            "variable": "carriers",
            "valor_original": corrected["carriers"],
            "valor_corregido": 0.0,
            "capa": "Capa 1 — Entorno",
            "mensaje": carriers_cfg.get("advertencia",
                                         "Carriers no aplican para este tipo celular.")
        })
        corrected["carriers"] = 0.0

    return corrected, advertencias

# Capa 2

def calc_available_nitrogen(composition: Dict[str, float],
                              group: str = "E.coli") -> float:
    """Calcula el N total disponible (g/L) usando fracciones del JSON."""
    n_fracs = get_config(group)["fuentes_nitrogeno"]
    return sum(conc * n_fracs.get(comp, 0.0) for comp, conc in composition.items())


def calc_available_carbon(composition: Dict[str, float],
                           group: str = "E.coli") -> float:
    """Calcula el C total disponible (g/L)."""
    c_fracs = get_config(group)["fuentes_carbono"]
    return sum(conc * c_fracs.get(comp, 0.0) for comp, conc in composition.items())


def stoichiometric_min_nitrogen(target_cdm: float, group: str = "E.coli") -> float:
    """N minimo requerido (g/L) para CDM objetivo. Ecuacion de Pirt."""
    return target_cdm * get_config(group)["estequiometria_biomasa"]["N_fraction"]


def stoichiometric_min_carbon(target_cdm: float, group: str = "E.coli",
                               aerobic: bool = True) -> float:
    """
    C minimo necesario en g C / L para producir target_cdm g CDM / L.

    Usa C_fraction_biomass del JSON (g C / g CDM), que permite comparar
    directamente con calc_available_carbon() que tambien devuelve g C / L.

    NOTA: NO usar Yx_s para este calculo — Yx_s tiene unidades g CDM / g glucosa
    y mezclar con g C/L genera un error de unidades que infla artificialmente
    la penalizacion de carbono.
    """
    bio = get_config(group)["estequiometria_biomasa"]
    C_frac = bio.get("C_fraction_biomass", 0.50)
    return target_cdm * C_frac


def stoichiometric_penalty(composition: Dict[str, float],
                            target_cdm: float,
                            group: str = "E.coli") -> Tuple[float, List[dict]]:

    cfg  = get_config(group)["estequiometria_biomasa"]
    tol  = cfg["N_deficit_tolerancia"]
    disc = cfg["N_deficit_descarte"]

    N_avail = calc_available_nitrogen(composition, group)
    N_req   = stoichiometric_min_nitrogen(target_cdm, group)
    N_def   = max(0.0, (N_req - N_avail) / max(N_req, 1e-9))

    C_avail = calc_available_carbon(composition, group)
    C_req   = stoichiometric_min_carbon(target_cdm, group)
    C_def   = max(0.0, (C_req - C_avail) / max(C_req, 1e-9))

    advertencias = []

    if N_def > disc:
        advertencias.append({
            "capa": "Capa 2 — Estequiometria N (Pirt)",
            "mensaje": (f"DEFICIT CRITICO de nitrogeno: N_disponible = {N_avail:.3f} g/L, "
                        f"N_requerido = {N_req:.3f} g/L (deficit = {N_def*100:.1f}%). "
                        + cfg["advertencia_deficit_N"]),
            "N_disponible_g_L": round(N_avail, 4),
            "N_requerido_g_L":  round(N_req, 4),
            "deficit_pct":      round(N_def * 100, 1),
        })
        return 1.0, advertencias

    if N_def > tol:
        advertencias.append({
            "capa": "Capa 2 — Estequiometria N (Pirt)",
            "mensaje": (f"Deficit de nitrogeno: N_disponible = {N_avail:.3f} g/L, "
                        f"N_requerido = {N_req:.3f} g/L (deficit = {N_def*100:.1f}%). "
                        + cfg["advertencia_deficit_N"]),
            "N_disponible_g_L": round(N_avail, 4),
            "N_requerido_g_L":  round(N_req, 4),
            "deficit_pct":      round(N_def * 100, 1),
        })

    if C_avail > 0 and N_avail > 0:
        ratio_cn = (C_avail / 0.40) / (N_avail / 0.132)  # C-moles / N-moles aprox
        if ratio_cn > 10:
            advertencias.append({
                "capa": "Capa 2 — Estequiometria C:N",
                "mensaje": (f"Ratio C:N = {ratio_cn:.1f}:1. "
                             + cfg.get("advertencia_exceso_C",
                                       "Exceso de carbono relativo a N.")),
                "ratio_CN": round(ratio_cn, 2),
            })

    N_pen = min(1.0, N_def * 4.0)    
    C_pen = min(1.0, C_def * 2.0)
    return max(N_pen, C_pen), advertencias

# Capa 3

def inject_micronutrients(composition: Dict[str, float],
                           available_components: List[str],
                           group: str = "E.coli") -> Tuple[Dict[str, float], List[str]]:
    """
    Inyecta micronutrientes esenciales a su concentracion minima.
    Solo los inyecta si el componente existe en el espacio de busqueda.
    Retorna (composicion_enriquecida, notas_de_inyeccion).
    """
    required = get_config(group)["micronutrientes_esenciales"]
    enriched = dict(composition)
    notas    = []

    for comp, datos in required.items():
        if not isinstance(datos, dict): 
            continue
        if comp not in available_components:
            continue
        min_conc = datos["conc_min_g_L"]
        current  = enriched.get(comp, 0.0)
        if current < min_conc:
            enriched[comp] = min_conc
            if current < 1e-6:
                notas.append(f"{comp} inyectado a {min_conc} g/L — {datos['funcion']}")
            else:
                notas.append(f"{comp} elevado {current:.4f}→{min_conc} g/L")

    return enriched, notas


def check_missing_micronutrients(composition: Dict[str, float],
                                  group: str = "E.coli") -> List[dict]:
    """
    Verifica si faltan micronutrientes esenciales en la composicion
    (para el predictor, donde no se inyectan automaticamente).
    """
    required = get_config(group)["micronutrientes_esenciales"]
    faltantes = []
    for comp, datos in required.items():
        if not isinstance(datos, dict):    
            continue
        if composition.get(comp, 0.0) < datos["conc_min_g_L"] * 0.1:
            faltantes.append({
                "capa": "Capa 3 — Micronutrientes esenciales",
                "componente": comp,
                "mensaje": datos["advertencia"],
                "funcion": datos["funcion"],
                "conc_minima_g_L": datos["conc_min_g_L"],
                "conc_actual_g_L": round(composition.get(comp, 0.0), 6),
            })
    return faltantes

# Capa 4

def cdm_ceiling(mu: float, group: str = "E.coli") -> float:
    """CDM maxima biologicamente plausible para un µ dado. Sigmoide calibrada."""
    cfg = get_config(group)["metabolitos_secundarios"]["acetato"]
    umbral   = cfg["mu_umbral_h"]
    cdm_low  = cfg["cdm_max_zona_alta_mu_g_L"]
    cdm_high = cfg["cdm_max_zona_baja_mu_g_L"]
    k        = cfg["sigmoid_k"]
    return cdm_low + (cdm_high - cdm_low) / (1.0 + math.exp(k * (mu - umbral)))


def cross_penalty_mu_cdm(predicted_mu: float,
                          predicted_cdm: float,
                          group: str = "E.coli") -> Tuple[float, List[dict]]:
    """
    Penalizacion cruzada µ×CDM. Detecta overflow metabolico.
    Retorna (penalty [0,1], advertencias).
    """
    if predicted_mu <= 0 or predicted_cdm <= 0:
        return 0.0, []

    cfg_acetato  = get_config(group)["metabolitos_secundarios"]["acetato"]
    cfg_formiato = get_config(group)["metabolitos_secundarios"].get("formiato_lactato", {})
    ceiling = cdm_ceiling(predicted_mu, group)
    advertencias = []

    if cfg_formiato and predicted_mu >= cfg_formiato.get("mu_umbral_h", 1.20):
        advertencias.append({
            "capa": "Capa 4 — Metabolitos secundarios (formiato/lactato)",
            "mensaje": cfg_formiato.get("advertencia", ""),
            "mu_predicho": round(predicted_mu, 4),
            "umbral_h": cfg_formiato.get("mu_umbral_h", 1.20),
        })

    if predicted_mu >= cfg_acetato["mu_umbral_h"]:
        advertencias.append({
            "capa": "Capa 4 — Overflow metabolico (acetato)",
            "mensaje": cfg_acetato["advertencia"],
            "mu_predicho":  round(predicted_mu, 4),
            "umbral_h":     cfg_acetato["mu_umbral_h"],
            "cdm_ceiling":  round(ceiling, 2),
            "cdm_predicho": round(predicted_cdm, 2),
        })

    if predicted_cdm <= ceiling:
        return 0.0, advertencias

    excess    = (predicted_cdm - ceiling) / max(ceiling, 1e-6)
    steepness = cfg_acetato.get("cross_penalty_steepness", 2.0)
    penalty   = min(1.0, excess * steepness)
    return round(penalty, 4), advertencias


# Capa 5

_ORGANIC_COEFF: Dict[str, float] = {
    "tryptone":              15.0,  
    "yeast_extract":         20.0,
    "casamino_acids":        18.0,
    "peptone_from_casein":   15.0,
    "meat_extract":          15.0,
    "cheese_whey":           10.0,
    "enbase_flo_mineral_medium": 5.0,
    "trace_metals_solution":  0.0,   
    "trace_element_sol":      0.0,
}


def calculate_osmolarity(composition: Dict[str, float]) -> Tuple[float, Dict[str, float]]:

    from pubchem_lookup import get_mw_and_ions as _get_mw_ions

    osm_total  = 0.0
    breakdown: Dict[str, float] = {}

    for comp, conc in composition.items():
        if conc <= 0:
            continue
        key = comp.lower()

        # Mezclas organicas: coeficiente empirico
        if key in _ORGANIC_COEFF:
            contrib = conc * _ORGANIC_COEFF[key]
        else:
            # Compuesto puro: MW e iones via PubChem (o fallback)
            mw, n_ions, _ = _get_mw_ions(comp)
            contrib = (conc * 1000.0 / mw) * n_ions if mw > 0 else 0.0

        if contrib > 0:
            breakdown[comp] = round(contrib, 2)
            osm_total += contrib

    return round(osm_total, 1), breakdown


def osmolarity_penalty(composition: Dict[str, float],
                        group: str = "E.coli") -> Tuple[float, List[dict]]:

    cfg = get_config(group)["osmolaridad"]
    osm_soft  = cfg["umbral_suave_mOsm_kg"]
    osm_hard  = cfg["umbral_fuerte_mOsm_kg"]
    osm_cap   = cfg["cap_letal_mOsm_kg"]
    osm_total, breakdown = calculate_osmolarity(composition)

    advertencias = []
    top3 = dict(sorted(breakdown.items(), key=lambda x: -x[1])[:3])

    if osm_total < osm_soft:
        return 0.0, []

    if osm_total >= osm_cap:
        advertencias.append({
            "capa": "Capa 5 — Osmolaridad",
            "mensaje": f"CRITICO: {cfg['advertencia_letal']} Osmolaridad = {osm_total:.0f} mOsm/kg.",
            "osm_total": osm_total, "zona": "letal",
            "principales_contribuyentes": top3,
        })
        return 1.0, advertencias

    if osm_total >= osm_hard:
        penalty = 0.5 + 0.5 * (osm_total - osm_hard) / (osm_cap - osm_hard)
        advertencias.append({
            "capa": "Capa 5 — Osmolaridad",
            "mensaje": f"{cfg['advertencia_fuerte']} Osmolaridad = {osm_total:.0f} mOsm/kg (umbral fuerte: {osm_hard} mOsm/kg).",
            "osm_total": osm_total, "zona": "hiperosmótico_severo",
            "principales_contribuyentes": top3,
        })
        return round(min(1.0, penalty), 4), advertencias

    penalty = 0.5 * (osm_total - osm_soft) / (osm_hard - osm_soft)
    advertencias.append({
        "capa": "Capa 5 — Osmolaridad",
        "mensaje": f"{cfg['advertencia_suave']} Osmolaridad = {osm_total:.0f} mOsm/kg (umbral recomendado: {osm_soft} mOsm/kg).",
        "osm_total": osm_total, "zona": "hiperosmótico_leve",
        "principales_contribuyentes": top3,
    })
    return round(penalty, 4), advertencias

# Capa 6

def redundancy_penalty(composition: Dict[str, float],
                        group: str = "E.coli") -> Tuple[float, List[dict]]:
    """
    Penaliza formulaciones con multiples componentes cumpliendo el mismo
    rol funcional.

    Retorna:
        penalty      [0, 1]
        advertencias List[dict]  — una por categoria con redundancia
    """
    roles = get_config(group).get("roles_funcionales", {})
    advertencias = []
    total_penalty = 0.0

    for categoria, datos in roles.items():
        if categoria.startswith("_"):
            continue
        if not isinstance(datos, dict):
            continue

        miembros      = datos.get("componentes", [])
        max_perm      = datos.get("_max_permitido", 1)
        pen_por_exceso= datos.get("_penalizacion_por_exceso", 0.30)

        activos = [
            (comp, conc) for comp, conc in composition.items()
            if comp in miembros and conc > 1e-4
        ]
        n_activos = len(activos)

        if n_activos > max_perm:
            exceso    = n_activos - max_perm
            pen       = min(1.0, exceso * pen_por_exceso)
            total_penalty = min(1.0, total_penalty + pen)

            activos_sorted = sorted(activos, key=lambda x: -x[1])
            principal   = [a[0] for a in activos_sorted[:max_perm]]
            redundantes = [a[0] for a in activos_sorted[max_perm:]]

            razon = datos.get("_razon", (
                f"La categoria '{categoria}' permite maximo {max_perm} componente(s). "
                f"Tener {n_activos} es redundante: los componentes adicionales duplican "
                f"la funcion sin aporte diferencial."
            ))

            advertencias.append({
                "capa":        "Capa 6 — Redundancia funcional",
                "categoria":   categoria,
                "n_activos":   n_activos,
                "max_permitido": max_perm,
                "principal":   principal,
                "redundantes": redundantes,
                "mensaje": (
                    f"Redundancia en '{categoria}': {n_activos} componentes activos "
                    f"(max permitido: {max_perm}). "
                    f"Principal: {principal}. Redundante(s): {redundantes}. "
                    + razon
                ),
                "penalizacion": round(pen, 4),
            })

    return round(total_penalty, 4), advertencias

# Evaluacion de candidatos

def evaluate_candidate(composition: Dict[str, float],
                        conditions: Dict[str, float],
                        group: str,
                        target_cdm: float,
                        available_components: Optional[List[str]] = None,
                        predicted_mu: float = 0.0,
                        ) -> Tuple[float, List[dict], Dict]:

    cfg_pesos = get_config(group)["pesos_de_penalizacion"]
    todas_advertencias = []

    # Capa 1 — env_penalty ponderado por gravedad de cada variable
    conds_corregidas, adv1 = enforce_env_bounds(conditions, group)
    todas_advertencias.extend(adv1)

    env_penalty = 0.0
    if adv1:
        sub_pesos = cfg_pesos.get("sub_pesos_capa1", {
            "temperature_C": 0.50, "pH": 0.35, "agitation_rpm": 0.15
        })
        cfg_env = get_config(group)["condiciones_entorno"]
        for adv in adv1:
            var     = adv["variable"]
            val_raw = adv["valor_original"]
            val_ok  = adv["valor_corregido"]
            if var in cfg_env and isinstance(cfg_env[var], dict):
                lo   = cfg_env[var]["min"]
                hi   = cfg_env[var]["max"]
                rang = max(hi - lo, 1e-9)
                # Exceso normalizado por rango
                exceso_norm = min(1.0, abs(val_raw - val_ok) / rang * 2.0)
            else:
                exceso_norm = 0.5
            sub_w = sub_pesos.get(var, 0.15)
            env_penalty = min(1.0, env_penalty + sub_w * exceso_norm)

    # Capa 2
    stoich_pen, adv2 = stoichiometric_penalty(composition, target_cdm, group)
    todas_advertencias.extend(adv2)

    # Capa 3
    avail = available_components or []
    enriched, notas3 = inject_micronutrients(composition, avail, group)
    for n in notas3:
        todas_advertencias.append({
            "capa": "Capa 3 — Micronutrientes",
            "mensaje": f"Inyeccion automatica: {n}",
        })

    # Capa 4
    cross_pen, adv4 = cross_penalty_mu_cdm(predicted_mu, target_cdm, group)
    todas_advertencias.extend(adv4)

    # Capa 5
    osm_pen, adv5 = osmolarity_penalty(composition, group)
    todas_advertencias.extend(adv5)

    # Capa 6 
    redund_pen, adv6 = redundancy_penalty(composition, group)
    todas_advertencias.extend(adv6)

    # Penalizacion total
    total = min(1.0,
                cfg_pesos.get("capa1_entorno",         0.28) * env_penalty  +
                cfg_pesos.get("capa2_estequiometria_N", 0.12) * stoich_pen   +
                cfg_pesos.get("capa4_overflow_mu_cdm",  0.30) * cross_pen    +
                cfg_pesos.get("capa5_osmolaridad",      0.15) * osm_pen      +
                cfg_pesos.get("capa6_redundancia",      0.15) * redund_pen)

    report = {
        "env_penalty":    round(env_penalty, 4),
        "stoich_penalty": round(stoich_pen,  4),
        "cross_penalty":  round(cross_pen,   4),
        "osm_penalty":    round(osm_pen,     4),
        "redund_penalty": round(redund_pen,  4),
        "total_penalty":  round(total, 4),
        "corrected_conds": conds_corregidas,
        "enriched_composition": enriched,
        "osm_detail":    next((a for a in adv5 if "osm_total"   in a), {}),
        "cross_detail":  next((a for a in adv4 if "cdm_ceiling" in a), {}),
        "redund_detail": adv6,
    }
    return total, todas_advertencias, report
