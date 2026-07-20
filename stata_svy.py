"""
stata_svy.py
============
Reimplementación en Python de la metodología de `svyset` / `svy:` de Stata,
siguiendo EXACTAMENTE las fórmulas publicadas en:

    StataCorp. 2025. Stata 19 Survey Data Reference Manual.
    Capítulo "Variance estimation — Variance estimation for survey data" (pp. 221-235)

No se copió ni se leyó código fuente interno (.ado/Mata) de Stata para construir
este módulo. Toda la lógica proviene de las fórmulas matemáticas publicadas
en el manual, reimplementadas de forma independiente.

Cobertura actual:
  - Diseño estratificado de una etapa:  svyset psu [pweight=w], strata(strata) fpc(fpc)
  - Diseño estratificado de dos etapas: svyset psu [pweight=w], strata(strata) fpc(fpc1) || ssu, fpc(fpc2)
  - Estimadores: total, mean, ratio, proportion (vía variable de puntaje / linealización)
  - FPC (finite population correction)
  - Estratos con un solo PSU: singleunit = 'missing' | 'certainty' | 'scaled' | 'centered'
  - Grados de libertad de diseño (d = n_psu - n_strata) e IC con t de Student

NO implementado todavía (fuera del alcance de esta primera versión):
  - vce(brr), vce(bootstrap), vce(jackknife), vce(sdr)  (métodos de replicación)
  - regresión / modelos (svy: regress, svy: logit, etc.)
  - postestratificación, calibración
  - dofsubpop / over() con grados de libertad por subpoblación
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional, Sequence


# --------------------------------------------------------------------------- #
# Estructura de resultados
# --------------------------------------------------------------------------- #
@dataclass
class SvyEstimate:
    name: str
    estimate: float
    variance: float
    se: float
    df: int
    ci_low: float
    ci_high: float
    n_obs: int
    n_psu: int
    n_strata: int

    def __repr__(self):
        return (f"{self.name}: {self.estimate:.10g}  "
                f"SE={self.se:.10g}  df={self.df}  "
                f"IC95%=[{self.ci_low:.10g}, {self.ci_high:.10g}]")


# --------------------------------------------------------------------------- #
# Declaración del diseño muestral (equivalente a svyset)
# --------------------------------------------------------------------------- #
class SvySet:
    """
    Equivalente a:
        svyset psu [pweight=weight], strata(strata) fpc(fpc) singleunit(...)
    o, para dos etapas:
        svyset psu [pweight=weight], strata(strata) fpc(fpc1) || ssu, fpc(fpc2)
    """

    def __init__(self, df: pd.DataFrame, psu: str, weight: str,
                 strata: Optional[str] = None,
                 fpc: Optional[str] = None,
                 ssu: Optional[str] = None,
                 fpc2: Optional[str] = None,
                 singleunit: str = "missing"):
        assert singleunit in ("missing", "certainty", "scaled", "centered")

        self.df = df.copy()
        self.psu_v = psu
        self.weight_v = weight
        self.strata_v = strata
        self.fpc_v = fpc
        self.ssu_v = ssu
        self.fpc2_v = fpc2
        self.singleunit = singleunit
        self.two_stage = ssu is not None

        if strata is None:
            # una sola "seudo-estrato" si no se especifica strata()
            self.df["_stratum_"] = 1
            self.strata_v = "_stratum_"

    # ------------------------------------------------------------------ #
    def _sampling_rate(self, fpc_vals: np.ndarray, n_units: int) -> float:
        """
        Traduce la variable fpc() a la tasa de muestreo f_h, según la regla
        del manual:
          - si fpc >= n_units  -> se interpreta como N_h (tamaño poblacional);
                                   f_h = n_units / N_h
          - si fpc <= 1        -> se interpreta como tasa directa f_h
          - si no hay fpc()    -> f_h = 0
        """
        if fpc_vals is None:
            return 0.0
        v = np.unique(fpc_vals)
        # Se asume constante dentro del estrato/PSU; se toma el primer valor.
        val = v[0]
        if val >= n_units:
            return n_units / val
        elif val <= 1:
            return val
        else:
            # Valor ambiguo (entre 1 y n_units): se sigue la misma regla >= n_units
            return n_units / val

    # ------------------------------------------------------------------ #
    def _stratum_var_component(self, y_hi: np.ndarray, f_h: float,
                                overall_mean: Optional[float] = None) -> tuple[float, bool]:
        """
        Calcula el componente de varianza de UNA estrato para la fórmula (1):
            (1 - f_h) * n_h/(n_h - 1) * sum_i (y_hi - ybar_h)^2

        Devuelve (componente, es_singleton).
        Maneja singleunit() según lo documentado.
        """
        n_h = len(y_hi)

        if n_h == 1:
            if self.singleunit == "missing":
                return np.nan, True
            elif self.singleunit == "certainty":
                # unidad de certeza: no aporta varianza
                return 0.0, True
            elif self.singleunit == "scaled":
                # se resuelve después (requiere factor global); aquí se marca
                return 0.0, True
            elif self.singleunit == "centered":
                center = overall_mean if overall_mean is not None else y_hi[0]
                # n_h/(n_h-1) se toma como 1 cuando n_h=1
                return (1 - f_h) * 1.0 * (y_hi[0] - center) ** 2, True

        ybar_h = y_hi.mean()
        comp = (1 - f_h) * (n_h / (n_h - 1)) * np.sum((y_hi - ybar_h) ** 2)
        return comp, False

    # ------------------------------------------------------------------ #
    def _total_and_variance(self, y_col: np.ndarray) -> tuple[float, float, int, int]:
        """
        Implementa las ecuaciones (1) y (2) del manual para un total ponderado.
        y_col: array ya multiplicado o no por el peso -- se pondera aquí.
        Devuelve (Y_hat, Var(Y_hat), n_psu_total, n_strata_total).

        Ruta rápida (vectorizada con pandas groupby) para diseños de una
        etapa -- necesaria para datasets con miles de estratos/PSU. Para
        diseños de dos etapas (con ssu) se usa la versión con loop por
        estrato, más lenta pero más simple de mantener.
        """
        if self.two_stage:
            return self._total_and_variance_loop(y_col)

        w = self.df[self.weight_v].values
        d = pd.DataFrame({
            self.strata_v: self.df[self.strata_v].values,
            self.psu_v: self.df[self.psu_v].values,
            "_wy_": w * y_col,
        })
        Y_hat = float(d["_wy_"].sum())

        g = d.groupby([self.strata_v, self.psu_v])["_wy_"].sum().reset_index()
        n_h = g.groupby(self.strata_v)[self.psu_v].size()
        L = len(n_h)
        n_psu_total = int(n_h.sum())
        sum_h = g.groupby(self.strata_v)["_wy_"].sum()
        ybar_h = sum_h / n_h

        gm = g.merge(ybar_h.rename("_yb_"), left_on=self.strata_v, right_index=True)
        gm["_dev2_"] = (gm["_wy_"] - gm["_yb_"]) ** 2
        ss_h = gm.groupby(self.strata_v)["_dev2_"].sum()

        if self.fpc_v:
            fpc_by_stratum = self.df.groupby(self.strata_v)[self.fpc_v].first()
            f_h = pd.Series(
                {h: self._sampling_rate(np.array([fpc_by_stratum.get(h, np.nan)]), n_h[h])
                 for h in n_h.index})
        else:
            f_h = pd.Series(0.0, index=n_h.index)

        is_singleton = n_h == 1
        multi = ~is_singleton
        comp = pd.Series(0.0, index=n_h.index)
        comp[multi] = (1 - f_h[multi]) * (n_h[multi] / (n_h[multi] - 1)) * ss_h[multi]

        if is_singleton.any():
            if self.singleunit == "missing":
                comp[is_singleton] = np.nan
            elif self.singleunit == "certainty":
                comp[is_singleton] = 0.0
            elif self.singleunit == "centered":
                overall_mean = g["_wy_"].mean()
                single_rows = g[g[self.strata_v].isin(n_h[is_singleton].index)]
                dev2 = (single_rows.set_index(self.strata_v)["_wy_"] - overall_mean) ** 2
                comp[is_singleton] = (1 - f_h[is_singleton]) * dev2.groupby(level=0).sum()
            elif self.singleunit == "scaled":
                comp[is_singleton] = 0.0

        if self.singleunit == "scaled" and is_singleton.any():
            L_s = int(is_singleton.sum())
            denom = L - L_s
            if denom > 0:
                scale = L / denom  # (L - L_c)/(L - L_c - L_s) con L_c = 0
                comp[multi] *= scale

        V_hat = float(comp.sum()) if not comp.isna().any() else np.nan
        return Y_hat, V_hat, n_psu_total, L

    def _total_and_variance_loop(self, y_col: np.ndarray) -> tuple[float, float, int, int]:
        """Versión original con loop por estrato -- usada solo para diseños de dos etapas."""
        df = self.df.copy()
        df["_y_"] = y_col
        df["_wy_"] = df[self.weight_v] * df["_y_"]

        strata_list = df[self.strata_v].unique()
        L = len(strata_list)

        Y_hat = df["_wy_"].sum()

        var_components = []
        singleton_flags = []
        n_psu_total = 0

        overall_mean_psu = None
        if self.singleunit == "centered":
            # media de los totales de PSU a nivel global (aprox. para centrar)
            psu_tot_all = df.groupby(self.psu_v)["_wy_"].sum()
            overall_mean_psu = psu_tot_all.mean()

        for h in strata_list:
            dh = df[df[self.strata_v] == h]
            psu_totals = dh.groupby(self.psu_v)["_wy_"].sum().values  # y_hi
            n_h = len(psu_totals)
            n_psu_total += n_h

            fpc_vals = dh[self.fpc_v].values if self.fpc_v else None
            f_h = self._sampling_rate(fpc_vals, n_h) if self.fpc_v else 0.0

            comp, is_single = self._stratum_var_component(
                psu_totals, f_h, overall_mean_psu)

            # --- segunda etapa (SSU), si aplica ---
            if self.two_stage and not is_single:
                second_term = 0.0
                for psu_id, dpsu in dh.groupby(self.psu_v):
                    ssu_totals = dpsu.groupby(self.ssu_v)["_wy_"].sum().values
                    m_hi = len(ssu_totals)
                    fpc2_vals = dpsu[self.fpc2_v].values if self.fpc2_v else None
                    f_hi = self._sampling_rate(fpc2_vals, m_hi) if self.fpc2_v else 0.0
                    if m_hi > 1:
                        ybar_hi = ssu_totals.mean()
                        second_term += (1 - f_hi) * (m_hi / (m_hi - 1)) * \
                            np.sum((ssu_totals - ybar_hi) ** 2)
                    # si m_hi==1, ese PSU no aporta varianza de 2da etapa
                comp = comp + f_h * second_term

            var_components.append(comp)
            singleton_flags.append(is_single)

        var_components = np.array(var_components, dtype=float)
        singleton_flags = np.array(singleton_flags)

        if self.singleunit == "scaled":
            L_c = 0  # no se modelan unidades de certeza explícitas en esta versión
            L_s = singleton_flags.sum()
            denom = (L - L_c - L_s)
            if denom > 0 and L_s > 0:
                scale = (L - L_c) / denom
                var_components[~singleton_flags] *= scale

        V_hat = np.nansum(var_components)
        if np.any(np.isnan(var_components)) and self.singleunit == "missing":
            V_hat = np.nan

        return Y_hat, V_hat, n_psu_total, L

    # ------------------------------------------------------------------ #
    def _design_df(self, n_psu: int, n_strata: int) -> int:
        return n_psu - n_strata

    def _ci(self, estimate: float, var: float, df: int, level: float = 0.95):
        se = np.sqrt(var) if var >= 0 else np.nan
        alpha = 1 - level
        tcrit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else np.nan
        return se, estimate - tcrit * se, estimate + tcrit * se

    # ====================================================================== #
    # ESTIMADORES PÚBLICOS  (equivalentes a svy: total / mean / ratio / proportion)
    # ====================================================================== #

    def total(self, varname: str, level: float = 0.95) -> SvyEstimate:
        y = self.df[varname].astype(float).values
        Y_hat, V_hat, n_psu, L = self._total_and_variance(y)
        d = self._design_df(n_psu, L)
        se, lo, hi = self._ci(Y_hat, V_hat, d, level)
        return SvyEstimate(f"total({varname})", Y_hat, V_hat, se, d, lo, hi,
                            len(self.df), n_psu, L)

    def ratio(self, y_var: str, x_var: str, level: float = 0.95) -> SvyEstimate:
        """
        Estimador de razón R = Y_hat / X_hat, linealizado vía variable de
        puntaje z_j = (y_j - R*x_j) / X_hat  (Shah 2004 / manual SVY, sec.
        'Ratios and other functions of survey data').
        """
        y = self.df[y_var].astype(float).values
        x = self.df[x_var].astype(float).values

        w = self.df[self.weight_v].values
        Y_hat = np.sum(w * y)
        X_hat = np.sum(w * x)
        R_hat = Y_hat / X_hat

        z = (y - R_hat * x) / X_hat  # variable de puntaje
        _, V_hat, n_psu, L = self._total_and_variance(z)

        d = self._design_df(n_psu, L)
        se, lo, hi = self._ci(R_hat, V_hat, d, level)
        return SvyEstimate(f"ratio({y_var}/{x_var})", R_hat, V_hat, se, d, lo, hi,
                            len(self.df), n_psu, L)

    def mean(self, varname: str, level: float = 0.95) -> SvyEstimate:
        """
        La media es la razón entre el total de la variable y el tamaño
        poblacional estimado (x = 1 para cada observación).
        """
        tmp_x = "_ones_"
        self.df[tmp_x] = 1.0
        est = self.ratio(varname, tmp_x, level)
        est.name = f"mean({varname})"
        return est

    def proportion(self, varname: str, category, level: float = 0.95) -> SvyEstimate:
        """
        Proporción de `category` dentro de varname: se construye el indicador
        0/1 y se trata como una media (caso particular de ratio con x=1).
        """
        tmp_y = "_ind_"
        self.df[tmp_y] = (self.df[varname] == category).astype(float)
        est = self.mean(tmp_y)
        est.name = f"proportion({varname}=={category})"
        del self.df[tmp_y]
        return est

    # ------------------------------------------------------------------ #
    def _total_covariance(self, y_col: np.ndarray, x_col: np.ndarray) -> float:
        """
        Covarianza linealizada Cov(Y_hat, X_hat), siguiendo la misma
        estructura de las ecuaciones (1)/(2) del manual pero con producto
        cruzado en vez de cuadrado:
            Cov = sum_h (1-f_h) n_h/(n_h-1) sum_i (y_hi-ybar_h)(x_hi-xbar_h)
        Implementación vectorizada con pandas groupby (necesaria para
        datasets con muchos miles de estratos/PSU; la versión con loop por
        estrato es demasiado lenta a esa escala).
        Nota: el término de 2da etapa (SSU) para la covarianza cruzada
        entre dos variables no está implementado todavía; solo se usa esta
        función para diseños de una etapa (p.ej. proportion_table()).
        """
        w = self.df[self.weight_v].values
        d = pd.DataFrame({
            self.strata_v: self.df[self.strata_v].values,
            self.psu_v: self.df[self.psu_v].values,
            "_wy_": w * y_col,
            "_wx_": w * x_col,
        })

        g = d.groupby([self.strata_v, self.psu_v])[["_wy_", "_wx_"]].sum().reset_index()
        n_h = g.groupby(self.strata_v)[self.psu_v].size()
        ybar_h = g.groupby(self.strata_v)["_wy_"].sum() / n_h
        xbar_h = g.groupby(self.strata_v)["_wx_"].sum() / n_h

        gm = g.merge(ybar_h.rename("_yb_"), left_on=self.strata_v, right_index=True)
        gm = gm.merge(xbar_h.rename("_xb_"), left_on=self.strata_v, right_index=True)
        gm["_cross_"] = (gm["_wy_"] - gm["_yb_"]) * (gm["_wx_"] - gm["_xb_"])
        cross_h = gm.groupby(self.strata_v)["_cross_"].sum()

        if self.fpc_v:
            fpc_by_stratum = self.df.groupby(self.strata_v)[self.fpc_v].first()
            f_h = pd.Series(
                {h: self._sampling_rate(np.array([fpc_by_stratum.get(h, np.nan)]), n_h[h])
                 for h in n_h.index})
        else:
            f_h = pd.Series(0.0, index=n_h.index)

        is_singleton = n_h == 1
        if is_singleton.any() and self.singleunit == "missing":
            return np.nan

        comp = pd.Series(0.0, index=n_h.index)
        multi = ~is_singleton
        comp[multi] = (1 - f_h[multi]) * (n_h[multi] / (n_h[multi] - 1)) * cross_h[multi]
        # singleton -> 0 (certainty/scaled/centered): mismo criterio aproximado
        # usado en _total_and_variance para el caso univariado.

        return float(comp.sum())

    # ====================================================================== #
    # ESTIMACIÓN POR DOMINIOS (equivalente a la opción over() de Stata)
    # Implementación vectorizada (pandas groupby) para datasets grandes.
    # ====================================================================== #

    def _domain_total_fast(self, y_col: np.ndarray, domain_mask: np.ndarray,
                            level: float = 0.95):
        """
        Total de una subpoblación/dominio, SIN restringir la muestra
        (siguiendo la fórmula (2)-(3) de [SVY] Subpopulation estimation):
        se pone en cero la variable fuera del dominio pero se conserva el
        diseño completo (todos los PSU de los estratos que sí tienen algún
        miembro del dominio). Los estratos sin ningún miembro del dominio se
        excluyen por completo (ni n ni L los cuentan), tal como hace Stata
        por default (sin dofsubpop).
        Vectorizado con pandas groupby para datasets grandes.
        """
        w = self.df[self.weight_v].values
        d = pd.DataFrame({
            self.strata_v: self.df[self.strata_v].values,
            self.psu_v: self.df[self.psu_v].values,
            "_wy_": w * y_col * domain_mask,
            "_dom_": domain_mask,
        })

        # Estratos con al menos una observación del dominio -> se conservan
        strata_dom_n = d.groupby(self.strata_v)["_dom_"].sum()
        strata_keep = strata_dom_n[strata_dom_n > 0].index
        d = d[d[self.strata_v].isin(strata_keep)]

        if len(d) == 0:
            return 0.0, np.nan, 0, 0

        # Totales por PSU (dentro del diseño completo restringido a estratos con dominio)
        psu_tot = d.groupby([self.strata_v, self.psu_v])["_wy_"].sum().reset_index()

        n_h = psu_tot.groupby(self.strata_v)[self.psu_v].size()
        sum_h = psu_tot.groupby(self.strata_v)["_wy_"].sum()
        ybar_h = sum_h / n_h

        merged = psu_tot.merge(ybar_h.rename("_ybar_"), left_on=self.strata_v, right_index=True)
        merged["_dev2_"] = (merged["_wy_"] - merged["_ybar_"]) ** 2
        ss_h = merged.groupby(self.strata_v)["_dev2_"].sum()

        L = len(n_h)
        n_total = int(n_h.sum())

        # f_h = 0 si no hay fpc() svyset (como en el diseño real sin fpc)
        if self.fpc_v:
            fpc_by_stratum = self.df.groupby(self.strata_v)[self.fpc_v].first()
            f_h = n_h.index.map(lambda h: self._sampling_rate(
                np.array([fpc_by_stratum.get(h, np.nan)]), n_h[h]))
            f_h = pd.Series(f_h, index=n_h.index)
        else:
            f_h = pd.Series(0.0, index=n_h.index)

        is_singleton = n_h == 1
        comp = pd.Series(0.0, index=n_h.index)

        multi = ~is_singleton
        comp[multi] = (1 - f_h[multi]) * (n_h[multi] / (n_h[multi] - 1)) * ss_h[multi]

        if is_singleton.any():
            if self.singleunit == "missing":
                comp[is_singleton] = np.nan
            elif self.singleunit == "certainty":
                comp[is_singleton] = 0.0
            elif self.singleunit == "centered":
                overall_mean = psu_tot["_wy_"].mean()
                single_rows = psu_tot[psu_tot[self.strata_v].isin(n_h[is_singleton].index)]
                dev2 = (single_rows.set_index(self.strata_v)["_wy_"] - overall_mean) ** 2
                comp[is_singleton] = (1 - f_h[is_singleton]) * dev2.groupby(level=0).sum()
            elif self.singleunit == "scaled":
                comp[is_singleton] = 0.0  # se escalan las no-singleton abajo

        if self.singleunit == "scaled" and is_singleton.any():
            L_s = int(is_singleton.sum())
            denom = (L - L_s)
            if denom > 0:
                scale = L / denom
                comp[multi] *= scale

        Y_hat = float(sum_h.sum())
        V_hat = float(comp.sum()) if not comp.isna().any() else np.nan

        d_df = n_total - L
        se, lo, hi = self._ci(Y_hat, V_hat, d_df, level)
        return Y_hat, V_hat, se, d_df, lo, hi, n_total, L

    def total_over(self, varlist: Sequence[str], over_vars: Sequence[str],
                    level: float = 0.95) -> pd.DataFrame:
        """
        Equivalente a:  svy linearized: total varlist, over(over_vars)

        Calcula, para cada combinación observada de over_vars, el total de
        cada variable en varlist, usando la fórmula correcta de estimación
        por subpoblación (no un simple subconjunto de filas).

        IMPORTANTE sobre grados de libertad: a diferencia de subpop() con
        una sola subpoblación, over() estima varias subpoblaciones de forma
        SIMULTÁNEA. El manual [SVY] Subpopulation estimation señala
        explícitamente que en ese caso Stata usa el mismo df del diseño
        completo (n_psu_total - n_strata_total) para todas las filas, en
        vez de recalcular n/L por dominio (que sí aplicaría con subpop()
        para una única subpoblación). Replicamos ese comportamiento aquí.
        """
        # df del diseño completo (igual para todas las filas, como en Stata)
        n_psu_total = self.df[self.psu_v].nunique()
        n_strata_total = self.df[self.strata_v].nunique()
        d_global = n_psu_total - n_strata_total

        work = self.df.copy()
        domain_key = work[list(over_vars)].astype(str).agg("#".join, axis=1)
        domains = sorted(domain_key.unique())

        alpha = 1 - level
        tcrit = stats.t.ppf(1 - alpha / 2, d_global) if d_global > 0 else np.nan

        rows = []
        for dom in domains:
            mask = (domain_key == dom).astype(float).values
            for var in varlist:
                y = self.df[var].astype(float).values
                Y_hat, V_hat, se, d_df_local, lo_local, hi_local, n_psu, L = \
                    self._domain_total_fast(y, mask, level)
                se = np.sqrt(V_hat) if V_hat >= 0 else np.nan
                lo = Y_hat - tcrit * se
                hi = Y_hat + tcrit * se
                rows.append(dict(domain=dom, variable=var, total=Y_hat, se=se,
                                  df=d_global, ci_low=lo, ci_high=hi,
                                  n_psu=n_psu_total, n_strata=n_strata_total,
                                  n_obs=int(mask.sum())))
        return pd.DataFrame(rows)

    def proportion_table(self, varname: str, level: float = 0.95):
        """
        Equivalente completo a `svy: proportion varname`: devuelve un
        DataFrame con la proporción y SE de cada categoría, y la matriz de
        covarianza conjunta e(V) (igual formato que Stata: incluye las
        covarianzas cruzadas entre categorías, no solo las varianzas).
        """
        w = self.df[self.weight_v].values
        N_hat = w.sum()
        categories = sorted(self.df[varname].dropna().unique().tolist())
        K = len(categories)

        scores = {}
        props = {}
        for cat in categories:
            ind = (self.df[varname] == cat).astype(float).values
            p_hat = np.sum(w * ind) / N_hat
            props[cat] = p_hat
            scores[cat] = (ind - p_hat) / N_hat

        V = np.zeros((K, K))
        for a in range(K):
            for b in range(a, K):
                c = self._total_covariance(scores[categories[a]], scores[categories[b]])
                V[a, b] = V[b, a] = c

        n_psu = self.df[self.psu_v].nunique()
        L = self.df[self.strata_v].nunique()
        d = self._design_df(n_psu, L)
        alpha = 1 - level
        tcrit = stats.t.ppf(1 - alpha / 2, d) if d > 0 else np.nan

        rows = []
        for k, cat in enumerate(categories):
            se = np.sqrt(V[k, k])
            p = props[cat]
            # IC lineal (normal) -- se mantiene disponible mas no es el default de Stata
            ci_lin_low, ci_lin_high = p - tcrit * se, p + tcrit * se
            # IC con transformación logit (comportamiento real y default de
            # `svy: proportion` en Stata; manual SVY, sec. svy: tabulate twoway
            # / estat, "Confidence intervals" para proporciones)
            if 0 < p < 1:
                f_p = np.log(p / (1 - p))
                se_logit = se / (p * (1 - p))
                lo_logit = f_p - tcrit * se_logit
                hi_logit = f_p + tcrit * se_logit
                ci_low = np.exp(lo_logit) / (1 + np.exp(lo_logit))
                ci_high = np.exp(hi_logit) / (1 + np.exp(hi_logit))
            else:
                ci_low = ci_high = p  # borde 0 o 1

            rows.append(dict(category=cat, proportion=p, se=se, df=d,
                              ci_low=ci_low, ci_high=ci_high,
                              ci_low_lineal=ci_lin_low, ci_high_lineal=ci_lin_high))
        table = pd.DataFrame(rows)
        return table, pd.DataFrame(V, index=categories, columns=categories)
