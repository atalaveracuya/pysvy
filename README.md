# stata_svy

Reimplementación en Python de la metodología `svyset` / `svy:` de Stata
(mean, total, ratio, proportion, y estimación por dominios con `over()`),
siguiendo las fórmulas publicadas en el manual oficial *Stata Survey Data
Reference Manual*, capítulo "Variance estimation for survey data" y
"Subpopulation estimation for survey data".

Validado contra Stata real (decimal a decimal) con `auto.dta` y con datos
reales de la ENA (INEI).

## Instalación

Desde GitHub, directamente con pip (recomendado):

```bash
pip install git+https://github.com/TU_USUARIO/TU_REPO.git
```

En Google Colab, la misma línea funciona igual dentro de una celda:

```python
!pip install git+https://github.com/TU_USUARIO/TU_REPO.git
```

## Uso básico

```python
import pandas as pd
from stata_svy import SvySet

df = pd.read_stata("mi_base.dta")

design = SvySet(
    df,
    psu="CONGLOMERADO",
    weight="FACTORFINAL",
    strata="ESTRATO_ANIO",
    singleunit="certainty",
)

print(design.mean("mi_variable"))
print(design.total("mi_variable"))
print(design.ratio("y_var", "x_var"))

tabla, cov = design.proportion_table("variable_categorica")

resultado = design.total_over(
    ["var1", "var2"],
    ["REGION", "ANIO"]
)
```

## Cobertura actual

- Diseño estratificado de una etapa (vectorizado, apto para datasets grandes)
- Diseño de dos etapas (PSU + SSU), vía loop -- más lento
- `mean`, `total`, `ratio`, `proportion` (con matriz de covarianza cruzada
  entre categorías e IC en escala logit, igual que Stata)
- `total(..., over(...))` -- estimación simultánea por dominios/subpoblaciones
- FPC, `singleunit()` (`missing`/`certainty`/`scaled`/`centered`)

## No implementado todavía

- `vce(brr/bootstrap/jackknife/sdr)`
- Modelos de regresión (`svy: regress`, `svy: logit`, etc.)
- Postestratificación, calibración
- `subpop()` de una única subpoblación (solo `over()` simultáneo)
