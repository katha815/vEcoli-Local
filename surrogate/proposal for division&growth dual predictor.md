For this surrogate, the target is derived from pq time-series dry-mass trajectories rather than summary CSV growth averages. This aligns with simulation mechanics: advancement to the next generation is driven by mass accumulation, so per-generation dry-mass fold change is directly tied to viability and growth dynamics.

## Notation and target construction
Let run $i$ have a maximum horizon of $G=8$ generations.

- $M_{i,g}^{\text{start}}$: dry mass at start of generation $g$
- $M_{i,g}^{\text{end}}$: dry mass at end of generation $g$
- Per-generation fold change:
$$f_{i,g}=\frac{M_{i,g}^{\text{end}}}{M_{i,g}^{\text{start}}}$$
- Log fold change:
$$z_{i,g}=\log f_{i,g}$$

If the run reaches only $n_i<8$ generations (early termination), define a fixed penalty fold change $f_{\text{pen}}\in(0,1]$ for missing generations, with $z_{\text{pen}}=\log(f_{\text{pen}})$.
The horizon-normalized scalar target is then
$$\bar z_i=\frac{1}{G}\left(\sum_{g=1}^{n_i} z_{i,g}+\sum_{g=n_i+1}^{G} z_{\text{pen}}\right), \qquad y_i=\exp(\bar z_i).$$

Interpretation: $y_i$ is an effective per-generation fold-change score over a fixed 8-generation horizon, where early failure is encoded through penalty-filled missing generations.

## Why this form is useful
- Multiplicative growth becomes additive in log space, making aggregation stable and interpretable.
- A fixed horizon gives targets comparable across runs with different stopping points.
- Penalty-filling keeps a single scalar regression target while still encoding failure severity.

## Possible Training target
Use $\bar z_i$ (log-space target) for regression, then map back with $\hat y_i=\exp(\hat z_i)$ for interpretation.
A simple weighted objective is
$$\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N} w_i\,(\hat z_i-\bar z_i)^2,\quad w_i=1+\lambda\,\mathbf{1}[n_i<8],$$
where $\lambda\ge 0$ increases emphasis on failed runs if needed.

## Notes on censoring semantics
Early-terminated runs are effectively right-censored with respect to full-horizon growth potential. The penalty-fill approach is a pragmatic surrogate-target design that preserves single-output simplicity while reflecting censoring/failure information.

## References
- Kalbfleisch, J. D., and Prentice, R. L. (2002). The Statistical Analysis of Failure Time Data (2nd ed.).
