# DRUM

### Distributionally Robust Unsupervised transfer learning with structurally Missing covariates

[Preprint](https://arxiv.org/abs/2605.24212)

## Overview

Prediction models are increasingly trained at high-resource centers and then deployed across systems whose data infrastructure is far leaner. This creates a recurring failure mode: some of the most predictive covariates recorded during training are **never collected** at the deployment site — not missing at random for a handful of subjects, but *absent for the entire target population*. Compounding the problem, outcome labels at the new site are frequently unavailable, so the model cannot simply be retrained or supervised into shape.

**DRUM** (Distributionally Robust Unsupervised transfer learning with structurally Missing covariates — also known as unsupervised domain adaptation) is built for exactly this setting. It partitions the covariates into a **shared** block $X$, observed everywhere, and a **structurally missing** block $A$, observed only in the source. Rather than imputing $A$ under untestable assumptions, DRUM learns a predictor $m(X)$ — using only the shared covariates — that optimizes **worst-case** predictive performance over the unknown target distribution of $A \mid X$. 

<p align="center">
  <img src="figures/fig1.png" alt="DRUM workflow" width="750">
  <br>
  <em>Figure 1. Overview of the DRUM framework.</em>
</p>

The result is a single, label-free, deployable predictor that transfers to many downstream populations — whether $A$ is mildly shifted, heavily missing, or entirely unavailable.

## Why DRUM?

- **Structural missingness, handled directly.** Existing transfer-learning and blockwise-missingness methods assume a shared covariate space, or assume the conditional law $A \mid X$ is stable across sites. DRUM requires neither.
- **No target labels needed.** Estimation never uses target outcomes — the framework is genuinely unsupervised in the target domain.
- **Worst-case guarantees, not point imputation.** Instead of guessing the target $A \mid X$, DRUM optimizes against an adversarial neighborhood of it, controlled by $\delta$.
- **Bias-corrected for finite samples.** A Neyman-orthogonal pseudo-outcome with cross-fitting removes first-order sensitivity to nuisance estimation error.

<p align="center">
  <img src="figures/fig2.png" alt="DRUM estimation pipeline: three stages and bias correction" width="900">
  <br>
  <em>Figure 2. Estimation procedure for DRUM.</em>
</p>


## Beyond missing data

Although DRUM is framed around structurally missing covariates, its formulation applies to any setting where covariates are available during training but absent or restricted at deployment. The same worst-case machinery carries over, with $A$ and the robustness parameter $\delta$ reinterpreted to fit the problem:

- **Unstable covariates.** In pharmaceutical stability prediction, models trained under controlled laboratory conditions — with environmental factors such as temperature or humidity — must generalize to real-world settings where these factors shift unpredictably. Here $A$ represents covariates that are *observed but unstable*, and $\delta$ controls the degree of protection against environmental variation.

- **Deliberately excluded covariates.** In algorithmic fairness, protected attributes such as race or sex may be available during training but impermissible to use at deployment. Here $A$ represents covariates that are *observed but deliberately excluded*, and optimizing over the worst-case conditional distribution can reduce the predictor's dependence on the realized distribution of protected attributes at deployment.

In each case, the underlying mathematical structure — a predictor restricted to $X$, robust to an adversarial conditional distribution of $A \mid X$ — is unchanged.
