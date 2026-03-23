from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger(__name__)

_CACHE_PATH  = Path(__file__).parent / "config" / "pubchem_cache.json"
_PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/MolecularWeight,MolecularFormula,IUPACName/JSON"
)
_TIMEOUT_S   = 8     
_RETRY_MAX   = 2      
_RATE_LIMIT_S= 0.25   

_FALLBACK_MW: Dict[str, float] = {
    "glucose":180.16,"glycerol":92.09,"sucrose":342.30,"lactose":342.30,
    "galactose":180.16,"fructose":180.16,"sodium_acetate":82.03,
    "sodium_pyruvate":110.04,"kh2po4":136.09,"k2hpo4":174.18,
    "k2hpo4_3h2o":228.22,"nah2po4":119.98,"na2hpo4":141.96,
    "na2hpo4_2h2o":177.99,"nh4h2po4":115.03,"nh42hpo4":132.06,
    "ammonium_dihydrogen_phosphate":115.03,"ammonium_chloride":53.49,
    "ammonium_sulfate":132.14,"nacl":58.44,"kcl":74.55,"k2so4":174.26,
    "mgso4":120.37,"mgso4_7h2o":246.47,"mgcl2":95.21,"mgcl2_6h2o":203.30,
    "cacl2":110.98,"cacl2_2h2o":147.01,"feso4_7h2o":278.01,"feso4":151.91,
    "mnso4_h2o":169.01,"znso4_7h2o":287.56,"citric_acid":192.12,
    "sodium_citrate":258.07,"sodium_citrate_2h2o":294.10,
    "mops":209.26,"tricine":179.17,"thiamine_hcl":337.27,"urea":60.06,
    "l_glutamine":146.15,"tryptophan":204.23,"sodium_chloride":58.44,
}
_FALLBACK_IONS: Dict[str, int] = {
    "glucose":1,"glycerol":1,"sucrose":1,"lactose":1,"galactose":1,
    "fructose":1,"sodium_acetate":2,"sodium_pyruvate":2,"kh2po4":2,
    "k2hpo4":3,"k2hpo4_3h2o":3,"nah2po4":2,"na2hpo4":3,"na2hpo4_2h2o":3,
    "nh4h2po4":2,"nh42hpo4":3,"ammonium_dihydrogen_phosphate":2,
    "ammonium_chloride":2,"ammonium_sulfate":3,"nacl":2,"kcl":2,
    "k2so4":3,"mgso4":2,"mgso4_7h2o":2,"mgcl2":3,"mgcl2_6h2o":3,
    "cacl2":3,"cacl2_2h2o":3,"feso4_7h2o":2,"feso4":2,"mnso4_h2o":2,
    "znso4_7h2o":2,"citric_acid":1,"sodium_citrate":4,
    "sodium_citrate_2h2o":4,"mops":1,"tricine":1,"thiamine_hcl":2,
    "urea":1,"l_glutamine":1,"tryptophan":1,"sodium_chloride":2,
}


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Caché de PubChem corrupta, empezando nueva.")
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except OSError as e:
        logger.warning(f"No se pudo guardar la caché: {e}")


_CACHE: dict = _load_cache()
_CACHE_DIRTY: bool = False

_NAME_ALIASES: Dict[str, str] = {
    # Sales
    "mgso4_7h2o":                    "magnesium sulfate heptahydrate",
    "mgso4":                         "magnesium sulfate",
    "mgcl2_6h2o":                    "magnesium chloride hexahydrate",
    "mgcl2":                         "magnesium chloride",
    "cacl2_2h2o":                    "calcium chloride dihydrate",
    "cacl2":                         "calcium chloride",
    "feso4_7h2o":                    "iron(II) sulfate heptahydrate",
    "feso4":                         "iron(II) sulfate",
    "fecl3_6h2o":                    "iron(III) chloride hexahydrate",
    "mnso4_h2o":                     "manganese(II) sulfate monohydrate",
    "znso4_7h2o":                    "zinc sulfate heptahydrate",
    "cuso4_5h2o":                    "copper(II) sulfate pentahydrate",
    "k2hpo4_3h2o":                   "dipotassium phosphate trihydrate",
    "na2hpo4_2h2o":                  "disodium phosphate dihydrate",
    "na2hpo4_7h2o":                  "disodium phosphate heptahydrate",
    "nah2po4_h2o":                   "monosodium phosphate monohydrate",
    "sodium_citrate_2h2o":           "trisodium citrate dihydrate",
    "sodium_citrate_3h2o":           "trisodium citrate dihydrate",
    "alcl3_6h2o":                    "aluminum chloride hexahydrate",
    "cobalt_chloride_6h2o":          "cobalt(II) chloride hexahydrate",
    "na2moo4_2h2o":                  "sodium molybdate dihydrate",
    "cucl2_2h2o":                    "copper(II) chloride dihydrate",
    # Fuentes de Nitrogeno
    "ammonium_chloride":             "ammonium chloride",
    "ammonium_sulfate":              "ammonium sulfate",
    "ammonium_dihydrogen_phosphate": "ammonium dihydrogen phosphate",
    "nh4h2po4":                      "ammonium dihydrogen phosphate",
    "nh42hpo4":                      "diammonium phosphate",
    # Sales fosfato
    "kh2po4":                        "potassium dihydrogen phosphate",
    "k2hpo4":                        "dipotassium phosphate",
    "k2so4":                         "potassium sulfate",
    "nah2po4":                       "monosodium phosphate",
    "na2hpo4":                       "disodium phosphate",
    # Ácidos orgánicos y buffers
    "mops":                          "3-(N-morpholino)propanesulfonic acid",
    "tricine":                       "N-(tris(hydroxymethyl)methyl)glycine",
    "hepes":                         "4-(2-hydroxyethyl)piperazine-1-ethanesulfonic acid",
    "tris":                          "tris(hydroxymethyl)aminomethane",
    "sodium_acetate":                "sodium acetate",
    "sodium_pyruvate":               "sodium pyruvate",
    "citric_acid":                   "citric acid",
    "sodium_citrate":                "trisodium citrate",
    # Vitaminas y cofactores
    "thiamine_hcl":                  "thiamine hydrochloride",
    "thiamine_HCl":                  "thiamine hydrochloride",
    "riboflavin":                    "riboflavin",
    "folic_acid":                    "folic acid",
    "cyanocobalamin":                "cyanocobalamin",
    "biotin":                        "biotin",
    "niacinamide":                   "nicotinamide",
    "pantothenate_ca":               "calcium pantothenate",
    "pyridoxine_hcl":                "pyridoxine hydrochloride",
    # Aminoácidos
    "l_glutamine":                   "L-glutamine",
    "l_alanine":                     "L-alanine",
    "l_arginine_hcl":                "L-arginine monohydrochloride",
    "l_cystine_2hcl":                "L-cystine dihydrochloride",
    "l_histidine_hcl_h2o":           "L-histidine monohydrochloride monohydrate",
    "l_isoleucine":                  "L-isoleucine",
    "l_leucine":                     "L-leucine",
    "l_lysine_hcl":                  "L-lysine monohydrochloride",
    "l_methionine":                  "L-methionine",
    "l_phenylalanine":               "L-phenylalanine",
    "l_serine":                      "L-serine",
    "l_threonine":                   "L-threonine",
    "l_tryptophan":                  "L-tryptophan",
    "l_tyrosine_2na_2h2o":           "L-tyrosine disodium salt dihydrate",
    "l_valine":                      "L-valine",
    "l_proline":                     "L-proline",
    "l_glycine":                     "glycine",
    "l_asparagine_h2o":              "L-asparagine monohydrate",
    "l_aspartic_acid":               "L-aspartic acid",
    "l_glutamic_acid":               "L-glutamic acid",
    "l_hydroxyproline":              "trans-4-hydroxy-L-proline",
    # Azúcares y carbono
    "glucose":                       "D-glucose",
    "fructose":                      "D-fructose",
    "galactose":                     "D-galactose",
    "sucrose":                       "sucrose",
    "lactose":                       "lactose",
    "glycerol":                      "glycerol",
    "trehalose":                     "alpha-D-trehalose",
    # Miscelaneo
    "urea":                          "urea",
    "nacl":                          "sodium chloride",
    "kcl":                           "potassium chloride",
    "h3bo3":                         "boric acid",
    "boric_acid":                    "boric acid",
    "myo_inositol":                  "myo-inositol",
    "choline_chloride":              "choline chloride",
    "hypoxanthine":                  "hypoxanthine",
    "adenosine":                     "adenosine",
    "uridine":                       "uridine",
    "adenine":                       "adenine",
    "thymidine":                     "thymidine",
    "putrescine_2hcl":               "putrescine dihydrochloride",
    "lipoic_acid":                   "alpha-lipoic acid",
}


def _normalize_name(raw: str) -> str:
    raw = raw.strip()
    if raw in _NAME_ALIASES:
        return _NAME_ALIASES[raw]
    if raw.lower() in _NAME_ALIASES:
        return _NAME_ALIASES[raw.lower()]
    return raw.replace("_", " ")


def _count_ions_from_formula(formula: str, compound_name: str) -> int:

    formula_clean = re.sub(r'[·•]\s*\d*H2O$', '', formula, flags=re.IGNORECASE)
    formula_clean = re.sub(r'\d+H2O$', '', formula_clean, flags=re.IGNORECASE)

    metal_patterns = {
        'Na': r'Na(\d*)',
        'K':  r'(?<![A-Z])K(\d*)(?![a-z])',
        'Ca': r'Ca(\d*)',
        'Mg': r'Mg(\d*)',
        'Fe': r'Fe(\d*)',
        'Mn': r'Mn(\d*)',
        'Zn': r'Zn(\d*)',
        'Cu': r'Cu(\d*)',
        'Co': r'Co(\d*)',
        'Al': r'Al(\d*)',
        'Ni': r'Ni(\d*)',
    }

    def _count(pat, f):
        m = re.search(pat, f)
        if not m:
            return 0
        return int(m.group(1)) if m.group(1) else 1

    metal_count = sum(_count(p, formula_clean) for p in metal_patterns.values())
    anion_patterns = {
        'Cl':  r'Cl(\d*)(?![a-z])',
        'SO4': r'SO4|S(?=O)',
        'PO4': r'PO4|P(?=O)',
        'NO3': r'NO3',
        'CO3': r'CO3',
        'HCO3':r'HCO3',
        'OH':  r'OH(?!\w)',
        'F':   r'F(?![e\w])',
    }
    paren_nh4 = re.search(r'\(NH4\)(\d*)', formula_clean)
    if paren_nh4:
        nh4_count = int(paren_nh4.group(1)) if paren_nh4.group(1) else 1
        formula_clean = re.sub(r'\(NH4\)\d*', '', formula_clean)
    else:
        nh4_count = len(re.findall(r'NH4', formula_clean))

    name_lower = compound_name.lower()

    organic_non_ionic = ['glucose','sucrose','glycerol','fructose','galactose',
                          'lactose','trehalose','urea','myo','inositol',
                          'adenosine','uridine','thymidine','adenine',
                          'hypoxanthine','lipoic']
    if any(k in name_lower for k in organic_non_ionic):
        return 1
    weak_acids = ['citric','mops','tricine','hepes','tris','acetate buffer',
                  'pipes','mes','bes','aces','ches']
    if any(k in name_lower for k in weak_acids):
        return 1
    if nh4_count > 0:

        if 'SO4' in formula_clean or ('S' in formula_clean and 'O' in formula_clean and 'N' in formula_clean):
            anion_n = 1
        elif 'PO4' in formula_clean or ('P' in formula_clean and 'O' in formula_clean):
            anion_n = 1
        elif 'Cl' in formula_clean:
            cl_m2 = re.search(r'Cl(\d*)', formula_clean)
            cl_n = int(cl_m2.group(1)) if cl_m2 and cl_m2.group(1) else 1
            return nh4_count + cl_n
        else:
            anion_n = 1
        return nh4_count + anion_n

    if 'ammonium' in compound_name.lower():
        n_lower = compound_name.lower()
        if 'sulfate' in n_lower:
            return 3  
        if 'chloride' in n_lower:
            return 2  
        if 'phosphate' in n_lower and 'di' in n_lower:
            return 3   
        if 'phosphate' in n_lower:
            return 2   
        return 2       

    if metal_count > 0:        
        cl_m = re.search(r'Cl(\d*)', formula_clean)
        cl_n = int(cl_m.group(1)) if cl_m and cl_m.group(1) else (1 if cl_m else 0)
        poly = 0
        for pat in [r'PO4', r'SO4', r'CO3', r'HCO3', r'NO3']:
            poly += len(re.findall(pat, formula_clean))
        total_anions = cl_n + poly
        if total_anions == 0:
            total_anions = 1  
        return metal_count + total_anions

    aa_indicators = ['amine','amino','glycine','alanine','leucine','valine',
                     'serine','threonine','proline','phenylalanine','tryptophan',
                     'methionine','isoleucine','cystine','tyrosine','histidine',
                     'lysine hcl','arginine','aspartic','glutamic','glutamine',
                     'asparagine','hydroxyproline']
    if any(k in name_lower for k in aa_indicators):
        if 'hcl' in name_lower or 'hydrochloride' in name_lower:
            return 2
        return 1

    if 'thiamine' in name_lower or 'pyridoxine' in name_lower:
        return 2 
    return 1


def _query_pubchem(name: str) -> Optional[Tuple[float, str, str]]:

    url = _PUBCHEM_URL.format(name=name.replace(" ", "%20").replace("(", "%28").replace(")", "%29"))

    for attempt in range(_RETRY_MAX):
        try:
            req = Request(url, headers={"User-Agent": "Alchemist/1.0 (contact: alchemist@lab)"})
            with urlopen(req, timeout=_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            props = data["PropertyTable"]["Properties"][0]
            mw      = float(props["MolecularWeight"])
            formula = props.get("MolecularFormula", "")
            iupac   = props.get("IUPACName", name)
            return mw, formula, iupac

        except HTTPError as e:
            if e.code == 404:
                logger.debug(f"PubChem: '{name}' no encontrado (404)")
                return None
            logger.debug(f"PubChem HTTP {e.code} para '{name}' (intento {attempt+1})")
        except URLError as e:
            logger.debug(f"PubChem no accesible: {e.reason}")
            return None  
        except (KeyError, json.JSONDecodeError, IndexError) as e:
            logger.debug(f"PubChem respuesta inesperada para '{name}': {e}")
            return None

        if attempt < _RETRY_MAX - 1:
            time.sleep(_RATE_LIMIT_S * 2)

    return None


def get_mw_and_ions(compound: str,
                     force_refresh: bool = False
                     ) -> Tuple[float, int, str]:
    global _CACHE, _CACHE_DIRTY
    key = compound.lower().strip()

    if not force_refresh and key in _CACHE:
        entry = _CACHE[key]
        return float(entry["mw"]), int(entry["n_ions"]), entry.get("formula", "")

    pubchem_name = _normalize_name(compound)
    result = _query_pubchem(pubchem_name)

    if result is not None:
        mw, formula, iupac = result
        n_ions = _count_ions_from_formula(formula, pubchem_name)
        _CACHE[key] = {
            "mw":          mw,
            "n_ions":      n_ions,
            "formula":     formula,
            "iupac":       iupac,
            "pubchem_name": pubchem_name,
            "source":      "pubchem",
        }
        _CACHE_DIRTY = True
        _save_cache(_CACHE)
        logger.info(f"  PubChem: {compound} → MW={mw} g/mol, formula={formula}, n_ions={n_ions}")
        time.sleep(_RATE_LIMIT_S) 
        return mw, n_ions, formula

    mw_fb = _FALLBACK_MW.get(key, 200.0) 
    ni_fb = _FALLBACK_IONS.get(key, 1)
    if mw_fb == 200.0 and key not in _FALLBACK_MW:
        logger.warning(
            f"  Compuesto '{compound}' no encontrado en PubChem ni en fallback. "
            f"Usando MW=200 g/mol, n_ions=1 (estimación)."
        )
    else:
        logger.debug(f"  Fallback hardcodeado: {compound} → MW={mw_fb}, n_ions={ni_fb}")

    _CACHE[key] = {
        "mw":     mw_fb,
        "n_ions": ni_fb,
        "formula": "",
        "source": "fallback",
    }
    _CACHE_DIRTY = True
    _save_cache(_CACHE)
    return mw_fb, ni_fb, ""


def get_mw(compound: str) -> float:
    return get_mw_and_ions(compound)[0]


def get_ions(compound: str) -> int:
    return get_mw_and_ions(compound)[1]


def preload_components(compounds: list[str],
                        force_refresh: bool = False,
                        verbose: bool = True) -> dict:

    results = {}
    n = len(compounds)
    for i, comp in enumerate(compounds):
        mw, n_ions, formula = get_mw_and_ions(comp, force_refresh=force_refresh)
        results[comp] = {"mw": mw, "n_ions": n_ions, "formula": formula}
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"  Pre-carga PubChem: {i+1}/{n} compuestos procesados")
    if verbose:
        logger.info(f"  Pre-carga completada: {n} compuestos")
    return results


def refresh_cache(compounds: list[str] | None = None) -> None:

    global _CACHE
    if compounds is None:
        compounds = list(_CACHE.keys())
    logger.info(f"Actualizando {len(compounds)} compuestos desde PubChem...")
    preload_components(compounds, force_refresh=True)


def cache_stats() -> dict:
    sources = {}
    for v in _CACHE.values():
        s = v.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    return {
        "total_compuestos":  len(_CACHE),
        "por_fuente":        sources,
        "ruta_cache":        str(_CACHE_PATH),
        "cache_existe":      _CACHE_PATH.exists(),
    }
