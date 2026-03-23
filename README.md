
# Alquemist
**¿Qué es Alchemist?**
 
Alchemist es un sistema de inteligencia artificial para el **diseño racional y predictivo de medios de cultivo microbiológico**. A partir de un dataset experimental de medios conocidos (composición, condiciones de entorno y resultados de crecimiento), el modelo es capaz de:
 
1. **Aprender** los patrones entre composición química y desempeño de crecimiento mediante modelos de Machine Learning.
2. **Predice** la tasa de crecimiento (µ, h⁻¹) y la densidad celular máxima (CDM, g/L) de cualquier medio propuesto por el usuario.
3. **Optimiza** formulaciones nuevas usando búsqueda bayesiana, maximizando simultáneamente CDM y µ bajo restricciones biológicas estrictas.
 
El sistema aplica una capa de **restricciones biológicas expertas** — pH, temperatura, osmolaridad, balance estequiométrico de nitrógeno, overflow metabólico por acetato, redundancia funcional de componentes — que penalizan o descartan formulaciones biológicamente inviables y/o redundantes en su formulación antes de presentarlas al usuario.
 
El modelo ha sido entrenado y validado con **más de 40 medios de cultivo individuales** de *E. coli*, cubriendo desde medios mínimos clásicos (M9, MOPS, M63, Davis-Mingioli) hasta medios de alta densidad celular (HCDC) de la literatura primaria (1950–2024). Se planea extender el sistema a células CHO, HEK293, Vero, *S. cerevisiae*, *Bacillus subtilis* y *Lactobacillus* en versiones futuras.

Características principales
 
- **Cinco modelos ML en paralelo**: Gradient Boosting, Random Forest, MLP Neural Network, Gaussian Process Regressor y XGBoost. Se selecciona automáticamente el de mayor R² en test.
- **Optimización bayesiana** con Optuna TPE.
- **Dos pases de optimización**: exploración amplia (P1, umbral frecuencia 10%) + refinamiento estricto (P2, 20%).
- **Seis capas de validación biológica** configurables por organismo desde JSON:
  - Capa 1 — Restricciones de entorno (pH, T°, rpm)
  - Capa 2 — Balance estequiométrico de nitrógeno 
  - Capa 3 — Inyección automática de micronutrientes esenciales
  - Capa 4 — Penalización por overflow metabólico de acetato(para E.coli)
  - Capa 5 — Osmolaridad 
  - Capa 6 — Redundancia funcional de componentes (fuentes de C, N, fosfato, etc.)
- **Lookup automático de pesos moleculares** vía PubChem API con caché local.
- **Exportación a Excel** con 5 hojas: medios sugeridos, resumen, advertencias biológicas, reporte de filtros y restricciones de referencia.
- **Soporte multi-grupo**: arquitectura lista para múltiples organismos, cada uno con su propio modelo y configuración biológica.
## Instalacion

### Requisitos
 
- Python 3.9 o superior
- pip
```bash
pip install -r requirements.txt
```
## Formato del dataset de entrada

### Hoja `experiment_data` (obligatoria)
 
Una fila por componente por medio. Las columnas obligatorias son:
 
| Columna | Tipo | Descripción |
|---|---|---|
| `medium_id` | str | Identificador único del medio (ej: `RIE_91_HCDC`) |
| `cell_type` | str | Tipo celular / cepa (ej: `E.coli_BL21`) |
| `component` | str | Nombre del componente en snake_case (ej: `kh2po4`) |
| `concentration` | float | Concentración en g/L (sólidos) o mL/L (líquidos) |
| `unit` | str | `g/L` o `mL/L` |
| `growth_rate` | float | Tasa de crecimiento µ (h⁻¹) — se repite en todas las filas del medio |
| `max_cell_density` | float | CDM máxima (g/L para bacterias, ×10⁶ cells/mL para mamíferos) |
 
Columnas opcionales: `doubling_time` (h), `strain` (cepa completa), `source` (referencia bibliográfica).
 
### Hoja `growth_curve` (opcional)
 
| Columna | Descripción |
|---|---|
| `medium_id` | Identificador del medio |
| `time_h` | Tiempo en horas |
| `cell_density` | Densidad celular (g/L CDM) |
 
### Hoja `culture_conditions` (opcional)
 
| Columna | Descripción |
|---|---|
| `medium_id` | Identificador del medio |
| `temperature_C` | Temperatura de cultivo (°C) |
| `pH` | pH de trabajo |
| `agitation_rpm` | Agitación (rpm) |
| `carriers` | `yes` / `no` (microcarriers) |

## Uso
 
### Modo 1 — Optimizar (formular los mejores medios)
 
```bash
python main.py --mode optimize --data datos.xlsx
```
 Opciones adicionales:
 
```bash
# Más iteraciones para mayor calidad (recomendado para datasets grandes)
python main.py --mode optimize --data datos.xlsx --n_trials 400 --n_top 10
 
# Salida personalizada
python main.py --mode optimize --data datos.xlsx --output resultados/mis_medios.xlsx
```
 
 
### Modo 2 — Predecir (evaluar un medio propuesto)
 
```bash
python main.py --mode predict --data data/data.xlsx --input mi_medio.xlsx
```
 
El archivo `mi_medio.xlsx` debe tener el mismo formato que la hoja `experiment_data`, con exactamente un `medium_id`. El sistema predice µ y CDM, aplica todas las capas de validación biológica y devuelve advertencias si la formulación es problemática.
 
```bash
# Con salida personalizada
python main.py --mode predict --data data/data.xlsx \
               --input mi_medio.xlsx --output evaluacion.xlsx
```

## Restricciones biológicas

Las restricciones para *E. coli* se encuentran en `config/bio_config_ecoli.json`. El archivo es completamente editable y está documentado internamente. Parámetros clave:
 
```json
{
  "condiciones_entorno": {
    "temperature_C": { "min": 28.0, "max": 40.0, "optimo": 37.0 },
    "pH":            { "min": 6.8,  "max": 7.5,  "optimo": 7.0  },
    "agitation_rpm": { "min": 150.0,"max": 300.0 }
  },
  "metabolitos_secundarios": {
    "acetato": { "mu_umbral_h": 0.8 }
  },
  "osmolaridad": {
    "umbral_suave_mOsm_kg": 800,
    "cap_letal_mOsm_kg": 1200
  }
}
```
 
Para añadir soporte a un nuevo organismo, basta con crear un archivo `bio_config_<organismo>.json` equivalente y añadir la cepa al diccionario en `bio_groups.py`.
 
## Limitaciones

**Relacionadas con los datos:**
- El modelo solo es tan bueno como los datos de entrenamiento. Con menos de ~15 medios por grupo, los modelos de ML tienen alta varianza y los resultados deben tomarse como orientativos.
- Las curvas de crecimiento incluidas son en su mayoría **estimadas logísticamente** (no extraídas de figuras experimentales o utilizano alguna cinética de crecimiento), lo que limita el módulo de curvas en versiones futuras.
- Muchas concentraciones de elementos traza o similares se reportan en la literatura con alta variabilidad de lote; el sistema las trata como valores puntuales.
 
**Relacionadas con el modelo ML:**
- Los modelos no capturan interacciones no lineales complejas entre micronutrientes (efectos sinérgicos/antagónicos de Zn²⁺ + Fe²⁺, por ejemplo) con los datasets actuales.
- El *feature engineering.py* actual no incluye descriptores moleculares,por tanto, toda la información es a nivel de concentración g/L.
- Extrapolación fuera del rango de concentraciones del *training set* no está garantizada. El optimizador limita las concentraciones a `[min_experimental, max_experimental × 1.05]`.
 
**Relacionadas con la biología:**
- La capa de osmolaridad usa una estimación simplificada (número de iones × molaridad); no calcula actividades termodinámicas reales.
- El umbral de overflow por acetato (µ > 0.80 h⁻¹) es una simplificación del fenómeno real, que depende también de la fuente de carbono, el diseño del biorreactor y la cepa específica.
- La penalización de redundancia funcional refleja criterios de formulación racional, pero en casos específicos (ej: co-cultivos con múltiples fuentes de carbono) puede ser demasiado restrictiva.
- El sistema **no simula cinética** de consumo de sustrato ni modelos de crecimiento (Monod, Pirt); predice puntos finales, no trayectorias temporales.
 
**Organismos soportados actualmente:**
- La versión 1.0 está optimizada exclusivamente para ***E. coli***. La arquitectura multi-grupo (`GroupModelSystem`) está implementada pero los configs biológicos para CHO, HEK293, Vero, *Lactobacillus* y *S. cerevisiae* están en desarrollo.
 
## Contribuciones
 
Las contribuciones son bienvenidas, especialmente:
 
- Nuevos datasets de medios (otras cepas, organismos no cubiertos).
- Configuraciones biológicas (`bio_config_*.json`) para nuevos organismos.
- Mejoras al pipeline de *feature engineering* (descriptores moleculares, etc.).
- Validaciones experimentales de formulaciones generadas por Alchemist.
- Mejoras al código o razonamiento en general.
 
Por favor abre un issue antes de enviar un PR para discutir el cambio.

---
## Cita

Si usas Alchemist en publicaciones o trabajos académicos, por favor cita este repositorio.
 
## Referencias bibliográficass
 
Las restricciones del motor biológico están fundamentadas en:
 
- Pirt SJ (1965) *Proc R Soc Lond B* 163:224 — Mantenimiento energético y balance de nitrogeno
- Roels JA (1980) *Biotechnol Bioeng* 22:2457 — Composición elemental de biomasa
- Luli GW & Strohl WR (1990) *Appl Env Microbiol* 56:1004 — Umbral de overflow de acetato
- Cayley S et al. (1991) *J Bacteriol* 173:3946 — Límites de osmolaridad en *E. coli*
- Riesenberg D et al. (1991) *J Biotechnol* 20:17 — Protocolo HCDC de referencia
## Anexo

Los datos tuvieron un tratamiento especial. Los detalles son los siguientes:

### Normalizacion de los nombres de los componentes
La literatura primaria usa nomenclatura inconsistente para el mismo compuesto (MgSO4·7H2O, magnesium sulfate heptahydrate, MgSO4 7H2O). La normalización a snake_case minúsculo elimina varianza ortográfica espuria que el modelo interpretaría como componentes distintos, inflandoartificialmente la dimensionalidad del espacio de features sin aporte informativo real. 

### Calculo del tiempo de duplicacion

La relación td = ln(2)/µ es una identidad matemática exacta del modelo de crecimiento exponencial de primer orden, válida durante la fase de crecimiento balanceado (Monod, 1949; Pirt, 1975). Su derivación no introduce supuestos adicionales más allá de asumir crecimiento exponencial, lo cual es inherente a la definición de µ reportada en los papers fuente. Se imputa únicamente cuando el valor no está reportado explícitamente, priorizando siempre el dato experimental.

### Filtro de componentes

Los modelos de ensemble son susceptibles al ruido de alta dimensionalidad cuando el número de features supera el número de muestras, un problema especialmente severo en datasets de medios de cultivo donde el número de medios es bajo relativo al número de componentes posibles. Retener solo componentes presentes en ≥20% de los medios del grupo reduce la dimensionalidad preservando los componentes con suficiente varianza observable para estimar su efecto, siguiendo el principio de parsimonia de Occam aplicado a ML biológico (Hastie et al., 2009, The Elements of Statistical Learning). El mecanismo de fallback (reducción del umbral hasta mínimo 5 componentes) garantiza que el modelo siempre tiene suficiente información mineral para hacer predicciones, incluso en grupos con pocas observaciones.

### Conversión OD₆₀₀ → CDM (g/L peso seco)
Suposición: Relación lineal entre absorbancia y biomasa seca con factor de conversión constante de 0.35 g CDM / unidad OD₆₀₀. Es cierto que el factor 0.35 es un promedio de cepa y condición. Cepas como BL21(DE3) bajo expresión activa de proteína recombinante pueden presentar factores ligeramente distintos por cambios en el tamaño y densidad celular. Esta variabilidad contribuye a la varianza del target max_cell_density en el dataset y se propaga al error de predicción del modelo.

### Estimaciones del rendimiento y CDM

Suposición: Consumo completo de la fuente de carbono limitante y rendimiento biomasico aerobio constante.
El rendimiento biomasico aerobio de E. coli sobre glucosa, Y_xs, tiene un valor termodinámico máximo de ~0.55 g CDM/g glucosa (Roels, 1980, Biotechnol Bioeng 22:2457) y valores experimentales típicos de 0.40–0.50 g/g en cultivos aerobios con control de pH y DO (Riesenberg et al., 1991; Luli & Strohl, 1990). Se usó Y_xs = 0.45 g/g como valor central (Roels, 1980) para medios minerales definidos, y 0.50 g/g para medios semi-definidos con extracto de levadura como fuente adicional de carbono reducido

### Curvas de crecimiento 

El crecimiento sigue un modelo logístico de tres parámetros con µ constante durante la fase exponencial e inhibición por producto al acercarse a CDM_max.

### CDM_max de la fase batch cuando el paper reporta fed-batch

El objetivo de Alchemist es optimizar la formulación del medio base (batch), no la estrategia de alimentación (fed-batch), que es una variable de proceso independiente de la composición del medio

### Conversiones de unidades

La conversión molar-másica es exacta si se conoce el peso molecular del compuesto exacto (incluyendo agua de cristalización). Se usaron pesos moleculares de referencia estándar 

### Ausencia de elementos traza

Suposición: Si un componente no aparece en la tabla de composición publicada, su concentración en el medio es efectivamente cero o negligible.


