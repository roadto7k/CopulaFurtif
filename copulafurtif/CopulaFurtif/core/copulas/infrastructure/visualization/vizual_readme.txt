CopulaFurtif — Visualization README
===================================

Purpose
-------
This file describes the visualization tools available in the project, what they show,
when to use them, and how to interpret them.

The goal is to help someone new to the project understand the graphs quickly and use
them for:
- pedagogy (explaining a copula),
- model selection,
- debugging,
- goodness-of-fit diagnostics,
- tail dependence analysis,
- conditional relationship analysis.

General Notes
-------------
1) Most plots work either on:
   - a fitted copula object (theoretical view), or
   - data (X, Y) / pseudo-observations (U, V), or both.

2) Many diagnostics assume pseudo-observations (U, V) in (0,1)^2.
   If you pass raw data (X, Y), some helper functions convert them to pseudo-observations
   using rank transform (pseudo_obs).

3) Recommended workflow for a new dataset:
   a) Fit one or several copulas
   b) Look at shape plots (CDF/PDF/Corner)
   c) Check diagnostics (PIT, residual heatmap, K-plot, chi-plot)
   d) Inspect tails (tail dependence, tail concentration, Pickands if relevant)
   e) Inspect conditional structure (arbitrage frontiers / fan)

4) A good model is not just "high likelihood":
   it should also behave correctly in tails and conditional views.

-------------------------------------------------------------------------------
SECTION A — Existing “Classic” Visualizations
-------------------------------------------------------------------------------

1) plot_cdf(...)
----------------
What it shows:
- The copula cumulative distribution function C(u, v).
- Can usually be displayed in 3D or contour mode.

Why it is useful:
- Gives an intuitive picture of the dependence structure.
- Good for pedagogy and quick family comparison.
- Helps detect obvious asymmetry or concentration effects.

How to interpret:
- Independence copula looks like C(u,v)=u*v (smooth baseline).
- Stronger dependence usually bends the surface away from the independence shape.
- Lower-tail / upper-tail asymmetry can sometimes be seen in contour curvature.

Typical usage:
- First visual to understand "what kind of copula shape" you fitted.

Common options:
- plot_type = 3D or CONTOUR
- Nsplit = grid resolution
- cmap = colormap

Limitations:
- CDF shape alone is not enough for model validation.
- Different families can look similar in CDF but differ in tails or conditionals.


2) plot_pdf(...)
----------------
What it shows:
- Copula density c(u, v), usually as 3D surface or contour.

Why it is useful:
- More informative than CDF for local dependence structure.
- Shows where probability mass is concentrated.
- Great for distinguishing families with similar global fit.

How to interpret:
- Peaks near (0,0) suggest stronger lower-tail concentration.
- Peaks near (1,1) suggest stronger upper-tail concentration.
- Symmetric vs asymmetric densities become more obvious here.
- A log-scale contour can reveal structure that is hidden on normal scale.

Typical usage:
- Compare candidate copulas after fitting.
- Check whether the fitted family concentrates mass where the data do.

Common options:
- plot_type = 3D or CONTOUR
- log_scale = True/False (very useful)
- levels = contour levels
- Nsplit = grid resolution

Limitations:
- Very sharp densities can be visually unstable if resolution is too low.
- Numerical noise may appear near boundaries for some families.


3) plot_mpdf(...)
-----------------
What it shows:
- "Marginalized" or joint PDF view combining:
  - copula dependence structure
  - chosen marginal distributions for X and Y

Why it is useful:
- Bridges the gap between copula-space (U,V) and real variable-space (X,Y).
- Useful when presenting results to non-specialists who think in original units.

How to interpret:
- Highlights where joint density is high in actual data units.
- Helps connect dependence diagnostics to practical ranges of X and Y.

Typical usage:
- Presentation / communication plot after marginals + copula are fixed.
- Sanity check that marginals and copula combine into a sensible joint shape.

Limitations:
- Strongly depends on marginal specification.
- A bad marginal fit can make the joint plot misleading even if the copula is good.


4) plot_arbitrage_frontiers(...)
--------------------------------
What it shows:
- Conditional quantile frontiers (typically low/high alpha levels), often used as
  decision boundaries or "frontiers" in copula space.

Why it is useful:
- Practical conditional view.
- Helps understand "given one variable level, what range is plausible for the other?"
- Useful in trading / risk heuristics and threshold-based systems.

How to interpret:
- Lower frontier (e.g. alpha=0.05): pessimistic conditional bound
- Upper frontier (e.g. alpha=0.95): optimistic conditional bound
- Wider gap = higher conditional uncertainty
- Nonlinearity = dependence structure is not purely linear

Typical usage:
- Decision support / visualization of conditional regimes.
- Can be overlaid with scatter data to compare theory vs observations.

Limitations:
- Usually shows only a few quantiles (e.g. 5% and 95%).
- Use the "conditional simulation fan" for a richer distributional view.


-------------------------------------------------------------------------------
SECTION B — Existing Diagnostics (already present in visualizer)
-------------------------------------------------------------------------------

5) Residual Heatmap (MatplotlibCopulaVisualizer.plot_residual_heatmap)
----------------------------------------------------------------------
What it shows:
- A 2D grid of residuals between empirical dependence structure and fitted copula
  (usually empirical copula vs theoretical copula / density-based residuals depending
  on implementation).

Why it is useful:
- Fast visual goodness-of-fit check.
- Shows where the model underestimates / overestimates dependence.

How to interpret:
- Neutral / near-zero residuals across the grid = good broad fit.
- Systematic hotspots in corners = tail mismatch.
- Bands or diagonal patterns = structural misspecification (e.g. wrong family).

Typical usage:
- Core debugging plot after fitting.
- Compare several candidate families visually.

Limitations:
- Bin-dependent (resolution matters).
- Can look "fine" globally while missing subtle conditional features.
- Should be complemented with PIT / K-plot / tail plots.


6) Tail Dependence Plot (MatplotlibCopulaVisualizer.plot_tail_dependence)
-------------------------------------------------------------------------
What it shows:
- Empirical vs model tail dependence behavior, typically using lower/upper tail
  estimates and/or comparison points across candidate copulas.

Why it is useful:
- Tail behavior is often the main reason to choose one copula over another.
- Helps identify if a family is underestimating extreme co-movements.

How to interpret:
- If empirical tail clustering is strong but the model tail dependence is weak,
  the fitted copula may be too light-tailed or wrong family.
- Compare lower and upper tails separately to detect asymmetry.

Typical usage:
- Model selection and risk-focused validation.
- Especially important for financial / energy / stress-event modeling.

Limitations:
- Tail estimation is noisy for small sample sizes.
- Choice of quantile thresholds (q_low / q_high) matters.


-------------------------------------------------------------------------------
SECTION C — New Diagnostics / Debug Visualizations (added)
-------------------------------------------------------------------------------

7) Simulated Corner Plot with Marginal KDEs
   (plot_simulated_corner_with_kdes)
-------------------------------------------
What it shows:
- Samples simulated from the fitted copula (U,V) as a scatter plot
- Marginal histograms + KDEs (kernel density estimates) on top/right axes
- Optional overlay of observed data (after conversion to pseudo-observations if needed)

Why it is useful:
- Extremely communicative to both technical and non-technical users.
- Lets you compare "what the fitted copula generates" vs "what the data looks like".

How to interpret:
- Scatter shape should resemble the observed dependence cloud.
- If overlayed data looks much more concentrated in corners / diagonals than simulation,
  the copula may be misspecified.
- Marginal KDEs in copula space should be approximately uniform (for U and V).

Typical usage:
- Daily visual sanity check after fit.
- Presentation plot for model behavior.

Limitations:
- In copula space, marginals are uniform by construction (unless there are issues).
- This plot is qualitative, not a formal GOF test.


8) Rosenblatt PIT Corner Diagnostic
   (plot_rosenblatt_pit_corner)
--------------------------------
What it shows:
- Rosenblatt transform outputs (Z1, Z2) of the observed data under the fitted copula.
- If the model is correct, transformed points should be approximately i.i.d. Uniform(0,1)^2.
- Includes marginal histograms and KS-test summary (depending on implementation).

Why it is useful:
- Stronger and more formal than a visual scatter alone.
- Rigorous complement to the residual heatmap.

How to interpret:
- Scatter should look like a uniform cloud (no pattern, no clustering).
- Histograms should be roughly flat.
- Low KS p-values may indicate model misspecification.

Typical usage:
- Goodness-of-fit validation after selecting a candidate family.
- Compare two close models when log-likelihoods are similar.

Limitations:
- Can be sensitive to sample size.
- Subsampling may be used for speed and tail balancing.


9) Tail Concentration Curves
   (plot_tail_concentration_curves)
-----------------------------------
What it shows:
- Empirical and theoretical probability mass concentration in:
  - lower-left corner (LL)
  - upper-right corner (UR)
- as a function of threshold t
- often shown as LL(t)/t and UR(t)/t for better comparability

Why it is useful:
- Richer than a single scalar tail dependence coefficient.
- Reveals how tail concentration evolves across scales.

How to interpret:
- Large gap between empirical and theoretical curves = tail mismatch.
- Asymmetric mismatch (lower good, upper bad) suggests wrong family for asymmetry.
- Curves help identify whether the model is too weak/too strong in extremes.

Typical usage:
- Tail-focused model selection.
- Debugging why a copula "looks okay" globally but fails in extremes.

Limitations:
- Noisy at very small t (few data points).
- Threshold range choice matters (e.g. t_max too large mixes tail with center).


10) Kendall K-plot
    (plot_kendall_k_plot)
-------------------------
What it shows:
- Non-parametric dependence diagnostic based on order statistics of the empirical joint CDF.
- Compares empirical behavior vs a reference (often independence benchmark in this version).

Why it is useful:
- Sensitive dependence diagnostic.
- Often detects misspecification that is not obvious in scatter plots.

How to interpret:
- Strong systematic deviations from the reference line indicate dependence structure
  differences from the benchmark.
- Compare K-plots across fitted models to judge which better captures dependence shape.

Typical usage:
- Advanced diagnostic after initial fit.
- Useful when multiple copulas have similar AIC/BIC/log-likelihood.

Limitations:
- Less intuitive for beginners than scatter/PDF plots.
- More diagnostic than presentation-oriented.


11) Chi-plot
    (plot_chi_plot)
-------------------
What it shows:
- χ_i vs λ_i diagnostic scatter (rank-based dependence diagnostic).
- Includes approximate significance bands (depending on implementation).

Why it is useful:
- Classical non-parametric dependence check.
- Helps distinguish independence vs systematic dependence.

How to interpret:
- Points clustered near zero suggest near-independence.
- Systematic positive/negative structure suggests dependence.
- Clear structure outside the bands indicates non-random dependence pattern.

Typical usage:
- Quick non-parametric diagnostic before/after fitting.
- Complements K-plot and residual heatmap.

Limitations:
- Can look noisy for small samples.
- Interpretation is easier when used comparatively (before/after, model A vs B).


12) Pickands Dependence Function
    (plot_pickands_dependence_function)
---------------------------------------
What it shows:
- Empirical vs theoretical Pickands dependence function A(ω)
- with reference bounds:
  - independence bound
  - comonotonic bound

Why it is useful:
- Essential for extreme-value copulas / max-stable families
  (e.g. Gumbel, Galambos, Husler-Reiss, etc.).
- Still informative as a diagnostic comparison for non-EV families.

How to interpret:
- The theoretical A(ω) must stay between the reference bounds.
- Agreement between empirical and theoretical curves supports tail dependence fit.
- Shape asymmetry and curvature can reveal EV-structure mismatch.

Typical usage:
- Mandatory when working seriously with EV copulas.
- Tail-focused validation tool.

Limitations:
- Empirical estimate can be noisy.
- Interpretation is less familiar to users without EV-copula background.


13) Conditional Simulation Fan
    (plot_conditional_simulation_fan)
-------------------------------------
What it shows:
- Multiple conditional quantile curves at once (e.g. α = 5%, 10%, 25%, 50%, 75%, 90%, 95%)
- Generalizes the idea of arbitrage frontiers from 2 quantiles to a full "fan"

Why it is useful:
- Gives a full distributional conditional view instead of only low/high bounds.
- Excellent for decision support, scenario analysis, and regime interpretation.

How to interpret:
- Middle curve (around α=0.5) is the conditional median-like frontier.
- Fan width indicates conditional uncertainty.
- Nonlinear / asymmetric fan shape reveals non-Gaussian dependence effects.

Typical usage:
- Trading / risk rules based on conditional quantiles.
- Presentation of conditional structure to stakeholders.

Limitations:
- Requires conditional CDF implementation to be stable for the chosen copula.
- Root solving may fail for some incomplete / experimental families.


-------------------------------------------------------------------------------
SECTION D — How to Use These Plots in Practice
-------------------------------------------------------------------------------

Recommended “Debug / Validation” Sequence
-----------------------------------------
1) Fit a copula (and marginals if needed)
2) plot_cdf / plot_pdf
   -> learn the shape
3) Simulated corner plot
   -> compare generated dependence cloud vs observed data
4) Residual heatmap
   -> locate mismatch regions
5) Rosenblatt PIT
   -> formal-ish GOF visual check
6) Tail dependence + tail concentration
   -> validate extreme co-movements
7) K-plot + chi-plot
   -> non-parametric diagnostics
8) Arbitrage frontiers / conditional fan
   -> conditional use-case behavior

Recommended “Presentation” Sequence
-----------------------------------
1) plot_mpdf (if stakeholders think in X/Y units)
2) plot_pdf contour (optional)
3) simulated corner plot (very intuitive)
4) arbitrage frontiers or conditional fan (actionable conditional insight)

If a Plot Looks Wrong — Quick Checklist
---------------------------------------
- Did you pass raw data (X,Y) where pseudo-observations (U,V) were expected?
- Is the copula actually fitted, or still at default parameters?
- Are the parameters near a boundary (causing numerical instability)?
- Is the sample size too small for tail diagnostics?
- Are the marginals fitted correctly (for MPDF / IFM / MLE-based plots)?
- Is the family suitable for the tail asymmetry observed in the data?

Common Caveats
--------------
- Tail diagnostics are noisy with limited data.
- High log-likelihood does NOT guarantee good tail behavior.
- A visually nice PDF can still fail the PIT diagnostic.
- A copula can fit central mass well and still miss extreme dependence.
- Boundary parameters may require careful numerical handling.

Naming Conventions (informal)
-----------------------------
- "classic plots" = CDF / PDF / MPDF / arbitrage frontiers
- "diagnostics" = residual heatmap / PIT / K-plot / chi-plot
- "tail diagnostics" = tail dependence / tail concentration / Pickands
- "conditional diagnostics" = arbitrage frontiers / conditional fan

Final Advice
------------
Use at least one plot from each category:
- shape,
- diagnostic,
- tail,
- conditional.

That combination catches most modeling mistakes faster than likelihood alone.
If two copulas have similar AIC/BIC/log-likelihood, these visualizations usually show
which one is actually safer / more faithful for your use case.