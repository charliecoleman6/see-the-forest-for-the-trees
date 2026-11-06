#!/usr/bin/env Rscript
# =============================================================================
# Covariance-floor decomposition of augmented bagging (AugBagg)
#
# Combines:
#   Mentch & Zhou (2022, JMLR) - augmented bagging, Figure 1 setup
#   O'Connell (2026, arXiv 2602.13104) - variance identity and covariance floor
#
# Question: when extra noise features improve bagging, is the gain coming from
# a lower covariance floor C_T(x) (less tree dependence), and at what point does
# the loss of resolution (bias) outweigh it?
#
# For every design, at fixed training covariates X and fixed test points, the
# script redraws Y | X R times. In each replicate it fits TWO forests with
# independent tree seeds. Then, at each test point x:
#
#   C_T(x)    = covariance across replicates of the two forests' predictions
#               (paired-forest estimator; unbiased for the floor at any B,
#               following the logic of O'Connell's Lemma 2)
#   MC(x)     = within-forest variance of tree predictions / B
#   bias2(x)  = (mean prediction across replicates - true mean)^2,
#               corrected for replicate noise
#
#   MSE(x) = E[(f_B(x) - mu(x))^2 | X]  ~=  bias2 + C_T + MC
#
# The empirical MSE is also recorded, so you can check that the decomposition
# adds up.
#
# Designs compared at each SNR:
#   - AugBagg: bagging on [X, N_1..N_q] with mtry = p + q (q = 0 is bagging)
#   - Random forests on the original X with mtry = 1, ..., p - 1
#
# Outputs (in cfg$out_dir):
#   results_pointwise_rXX.csv  one row per (X draw, SNR, design, test point)
#   results_summary_rXX.csv    averaged over test points and X draws
#   decomposition_rXX.pdf      plots (only if ggplot2 is installed)
#
# Usage:
#   Rscript covfloor_augbagg.R          # full run (slow; see runtime note)
#   QUICK=1 Rscript covfloor_augbagg.R  # small smoke test, a few minutes
#
# Runtime note: total forests fitted = n_X * |snr| * n_designs * R * 2.
# With the defaults: 3 * 4 * 12 * 150 * 2 = 43,200 forests of 500 trees, n = 100.
# Expect several hours on a laptop; it is parallel over replicates. Results are
# appended to the CSV as each design finishes, so an interrupted run keeps what
# it has done.
# =============================================================================

suppressPackageStartupMessages({
  library(ranger)
  library(MASS)      # mvrnorm
  library(parallel)
})

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
cfg <- list(
  n        = 100,                       # training size (Mentch & Zhou Fig. 1)
  p        = 5,                         # signal features
  beta     = rep(1, 5),
  rho      = 0.35,                      # Sigma_ij = rho^|i-j|
  snr      = c(0.01, 0.05, 0.09, 0.14),
  q_grid   = c(0, 5, 10, 25, 50, 100, 150, 250),
  rf_mtry  = 1:4,                       # RF baselines (mtry = 5 is q = 0 bagging)
  r_corr   = 0,                         # noise/signal correlation, eq. (3); try 0.7
  n_test   = 200,                       # fixed test points
  n_X      = 3,                         # independent fixed-X configurations
  R        = 150,                       # outcome replicates per design
  B        = 500,                       # trees per forest
  min_node = 1,                         # full-depth trees, as in the paper
  cores    = max(1, detectCores() - 1),
  seed     = 2026,
  out_dir  = "covfloor_results"
)

if (nzchar(Sys.getenv("QUICK"))) {
  message("QUICK mode: small settings for a smoke test")
  cfg$snr    <- c(0.01, 0.14)
  cfg$q_grid <- c(0, 25, 100)
  cfg$rf_mtry <- 1
  cfg$n_test <- 50
  cfg$n_X    <- 1
  cfg$R      <- 30
  cfg$B      <- 200
}

dir.create(cfg$out_dir, showWarnings = FALSE)
tag <- sprintf("r%02d", round(100 * cfg$r_corr))
file_point <- file.path(cfg$out_dir, sprintf("results_pointwise_%s.csv", tag))
file_summ  <- file.path(cfg$out_dir, sprintf("results_summary_%s.csv", tag))
if (file.exists(file_point)) file.remove(file_point)

# Parallelism uses a PSOCK cluster (fresh R processes), not forking: forking is
# blocked in Positron and unsafe in other GUI front ends.
use_cluster <- cfg$cores > 1
# With a cluster, each ranger call is single-threaded. Without it, ranger threads.
rf_threads <- if (use_cluster) 1 else cfg$cores
if (use_cluster) {
  # Clean up a cluster left behind by an earlier interrupted run in this session
  if (exists("cl", inherits = FALSE)) try(stopCluster(cl), silent = TRUE)
  cl <- makePSOCKcluster(cfg$cores)
  clusterCall(cl, .libPaths, .libPaths())
  clusterEvalQ(cl, suppressPackageStartupMessages(library(ranger)))
  clusterExport(cl, c("cfg", "rf_threads"))
}
par_lapply <- function(X, FUN, ...) {
  if (use_cluster) parLapply(cl, X, FUN, ...) else lapply(X, FUN, ...)
}

set.seed(cfg$seed)
Sigma <- cfg$rho^abs(outer(seq_len(cfg$p), seq_len(cfg$p), "-"))
signal_var <- drop(t(cfg$beta) %*% Sigma %*% cfg$beta)

# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
# Noise columns per Mentch & Zhou eq. (3): N = r * X_j + sqrt(1 - r^2) * Z,
# with j fixed per noise column (so train and test use the same mapping).
make_noise <- function(X, jmap, r) {
  qmax <- length(jmap)
  Z <- matrix(rnorm(nrow(X) * qmax), nrow(X), qmax)
  if (r == 0) return(Z)
  r * X[, jmap, drop = FALSE] + sqrt(1 - r^2) * Z
}

design_matrix <- function(X, N, q) {
  M <- cbind(X, N[, seq_len(q), drop = FALSE])
  colnames(M) <- c(paste0("X", seq_len(ncol(X))),
                   if (q > 0) paste0("N", seq_len(q)))
  M
}

# Designs: AugBagg over q, plus RF baselines on original features
designs <- rbind(
  data.frame(type = "AugBagg", q = cfg$q_grid, mtry = cfg$p + cfg$q_grid),
  data.frame(type = "RF", q = 0, mtry = cfg$rf_mtry)
)
designs$label <- ifelse(designs$type == "AugBagg",
                        sprintf("AugBagg q=%d", designs$q),
                        sprintf("RF mtry=%d", designs$mtry))

# ----------------------------------------------------------------------------
# One replicate: two independent forests on the same (X, y)
# ----------------------------------------------------------------------------
fit_pair <- function(r, Xd, Xte, Y, mtry, seedsA, seedsB) {
  y <- Y[, r]
  common <- list(x = Xd, y = y, num.trees = cfg$B, mtry = mtry,
                 replace = TRUE, sample.fraction = 1,
                 min.node.size = cfg$min_node, num.threads = rf_threads,
                 verbose = FALSE)
  fA <- do.call(ranger, c(common, list(seed = seedsA[r])))
  fB <- do.call(ranger, c(common, list(seed = seedsB[r])))
  pa <- predict(fA, data = Xte, predict.all = TRUE,
                num.threads = rf_threads)$predictions   # n_test x B
  pb <- predict(fB, data = Xte, num.threads = rf_threads)$predictions
  list(a  = rowMeans(pa),
       b  = pb,
       mc = apply(pa, 1, var) / cfg$B)
}

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
t0 <- Sys.time()
n_total <- cfg$n_X * length(cfg$snr) * nrow(designs)
k <- 0

for (ix in seq_len(cfg$n_X)) {
  # Fixed covariate configuration (train + test), including the noise columns.
  # AugBagg draws its noise once and shares it across trees, so the noise
  # columns are part of the fixed design here.
  qmax  <- max(cfg$q_grid)
  jmap  <- sample(cfg$p, qmax, replace = TRUE)
  X_tr  <- mvrnorm(cfg$n, rep(0, cfg$p), Sigma)
  X_te  <- mvrnorm(cfg$n_test, rep(0, cfg$p), Sigma)
  N_tr  <- make_noise(X_tr, jmap, cfg$r_corr)
  N_te  <- make_noise(X_te, jmap, cfg$r_corr)
  mu_tr <- drop(X_tr %*% cfg$beta)
  mu_te <- drop(X_te %*% cfg$beta)
  
  for (snr in cfg$snr) {
    sigma2 <- signal_var / snr
    # Common random numbers: the same R outcome vectors for every design
    Y <- mu_tr + matrix(rnorm(cfg$n * cfg$R, sd = sqrt(sigma2)), cfg$n, cfg$R)
    seedsA <- sample.int(.Machine$integer.max, cfg$R)
    seedsB <- sample.int(.Machine$integer.max, cfg$R)
    
    for (d in seq_len(nrow(designs))) {
      k <- k + 1
      des <- designs[d, ]
      Xd  <- design_matrix(X_tr, N_tr, des$q)
      Xte <- design_matrix(X_te, N_te, des$q)
      
      reps <- par_lapply(seq_len(cfg$R), fit_pair, Xd = Xd, Xte = Xte, Y = Y,
                         mtry = des$mtry, seedsA = seedsA, seedsB = seedsB)
      PA <- do.call(rbind, lapply(reps, `[[`, "a"))    # R x n_test
      PB <- do.call(rbind, lapply(reps, `[[`, "b"))
      MC <- do.call(rbind, lapply(reps, `[[`, "mc"))
      
      # Covariance floor: paired-forest covariance across replicates
      cA <- sweep(PA, 2, colMeans(PA))
      cB <- sweep(PB, 2, colMeans(PB))
      C_T <- colSums(cA * cB) / (cfg$R - 1)
      
      # Resolution / bias^2 of the infinite-forest target, corrected for the
      # sampling noise of the replicate mean
      AV    <- (PA + PB) / 2
      v_avg <- apply(AV, 2, var)
      bias2 <- (colMeans(AV) - mu_te)^2 - v_avg / cfg$R
      
      mc      <- colMeans(MC)
      mse_emp <- colMeans(sweep(PA, 2, mu_te)^2)
      
      out <- data.frame(
        x_draw = ix, snr = snr, sigma2 = sigma2,
        type = des$type, q = des$q, mtry = des$mtry, label = des$label,
        test_pt = seq_len(cfg$n_test),
        C_T = C_T, bias2 = bias2, mc = mc,
        mse_decomp = bias2 + C_T + mc,
        mse_emp = mse_emp
      )
      write.table(out, file_point, sep = ",", row.names = FALSE,
                  col.names = !file.exists(file_point),
                  append = file.exists(file_point))
      
      el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
      message(sprintf(
        "[%d/%d] X%d SNR=%.2f %-16s C_T/s2=%.4f bias2/s2=%.4f relErr=%.4f  (%.1f min, ~%.0f min left)",
        k, n_total, ix, snr, des$label,
        mean(C_T) / sigma2, mean(bias2) / sigma2, 1 + mean(mse_emp) / sigma2,
        el, el / k * (n_total - k)))
    }
  }
}
if (use_cluster) stopCluster(cl)

# ----------------------------------------------------------------------------
# Summary: average over test points, then over X draws. Everything is also
# scaled by sigma^2 so it is on the same scale as Mentch & Zhou's
# "relative error" (test MSE / sigma^2 = 1 + MSE(f)/sigma^2).
# ----------------------------------------------------------------------------
pt <- read.csv(file_point)
per_x <- aggregate(cbind(C_T, bias2, mc, mse_decomp, mse_emp) ~
                     x_draw + snr + sigma2 + type + q + mtry + label,
                   data = pt, FUN = mean)
summ <- aggregate(cbind(C_T, bias2, mc, mse_decomp, mse_emp) ~
                    snr + sigma2 + type + q + mtry + label,
                  data = per_x, FUN = mean)
for (v in c("C_T", "bias2", "mc", "mse_decomp", "mse_emp")) {
  summ[[paste0(v, "_rel")]] <- summ[[v]] / summ$sigma2
}
summ$rel_err      <- 1 + summ$mse_emp_rel
summ$floor_share  <- summ$C_T / summ$mse_emp     # share of error from the floor
summ$decomp_check <- summ$mse_decomp / summ$mse_emp  # should be close to 1
summ <- summ[order(summ$snr, summ$type, summ$q, summ$mtry), ]
write.csv(summ, file_summ, row.names = FALSE)
message("Wrote ", file_summ)

# ----------------------------------------------------------------------------
# Plots (optional)
# ----------------------------------------------------------------------------
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  comp <- c(rel_err = "Relative test error",
            C_T_rel = "Covariance floor C_T / sigma^2",
            bias2_rel = "Bias^2 (resolution) / sigma^2",
            mc_rel = "Monte Carlo / sigma^2")
  long <- do.call(rbind, lapply(names(comp), function(v)
    data.frame(summ[, c("snr", "type", "q", "mtry", "label")],
               component = comp[[v]], value = summ[[v]])))
  long$component <- factor(long$component, levels = comp)
  long$snr_lab   <- factor(sprintf("SNR = %.2f", long$snr))
  
  aug <- subset(long, type == "AugBagg")
  rf  <- subset(long, type == "RF")
  
  g <- ggplot(aug, aes(q, value)) +
    geom_hline(data = rf, aes(yintercept = value, colour = factor(mtry)),
               linetype = "dashed") +
    geom_line(linewidth = 0.8) + geom_point(size = 1.5) +
    facet_grid(component ~ snr_lab, scales = "free_y") +
    labs(x = "# extra noise features (q)", y = NULL,
         colour = "RF mtry\n(original X)",
         title = "Augmented bagging: covariance floor vs resolution",
         subtitle = sprintf("n = %d, p = %d, r = %.2f, R = %d, B = %d, %d X draws",
                            cfg$n, cfg$p, cfg$r_corr, cfg$R, cfg$B, cfg$n_X)) +
    theme_bw()
  f_pdf <- file.path(cfg$out_dir, sprintf("decomposition_%s.pdf", tag))
  ggsave(f_pdf, g, width = 12, height = 10)
  message("Wrote ", f_pdf)
} else {
  message("ggplot2 not installed; skipping plots")
}

message(sprintf("Done in %.1f min",
                as.numeric(difftime(Sys.time(), t0, units = "mins"))))