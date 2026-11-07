#!/usr/bin/env Rscript
# =============================================================================
# Empirical check of O'Connell's finite-sample variance identity
#
#   O'Connell (2026, arXiv 2602.13104), Theorem 1:
#     Var(f_B(x) | X) = (1/B) sigma_T^2(x) + ((B-1)/B) C_T(x)
#   and its operational form, eq. (8):
#     Var(f_B(x) | X) = (1/B) E[sigma_{T,theta}^2(x) | X] + C_T(x)
#
#   sigma_T^2(x)        = Var(T_theta(x) | X): variance of ONE tree's prediction
#                         over both the outcome draw Y and the tree randomness
#   sigma_{T,theta}^2   = Var(T_theta(x) | X, Y): spread of trees within one
#                         forest at fixed data (the Monte Carlo part)
#   C_T(x)              = covariance floor between two independent trees
#
# Each of these is estimated SEPARATELY from simulated data, then plugged into
# the two formulas and compared with the forest's actual variance across
# outcome draws. Theorem 1 is an identity, so what is being checked is that the
# components can be estimated independently and still add up.
#
# Two scenarios:
#   "oconnell"  Replication of O'Connell's Figure 3: n = 400, his 12 core
#               covariates (Appendix B, Table 1), nonlinear mean, heteroscedastic
#               noise, mtry = 4, bootstrap, node size 5 (ranger default), 20
#               jittered test points, R = 300 outcome draws, B up to 8,000.
#               He reports total variance within 1.3% of the floor at large B.
#   "mz"        The Mentch & Zhou setup used in covfloor_augbagg.R: n = 100,
#               p = 5, rho = 0.35, beta = 1, one SNR, plain bagging
#               (mtry = p), fully grown trees (node size 1).
#
# At each test point x, for each of R outcome draws (X fixed, Y redrawn):
#   - Forest A: B_max trees; every tree's prediction is kept. The B-tree forest
#     is the average of A's first B trees, so one fit covers the whole B grid.
#   - Forest B: B_max trees, independent seed; only its average is kept.
# Then:
#   actual variance(B)   sample variance across draws of A's B-tree average
#   C_T                  covariance across draws of A and B at B_max
#                        (O'Connell's paired estimator, Lemma 2 / eq. 5;
#                        does not use Theorem 1)
#   sigma_T^2            for each tree position b, variance across draws of
#                        tree b's prediction; averaged over b
#   E[sigma_{T,theta}^2] within-forest variance of A's trees, averaged over draws
#
# Precision: with R draws, a single test point's sample variance has relative
# standard error about sqrt(2 / (R - 1)) (about 8% at R = 300). Averaged over
# test points the error is smaller, so expect averaged ratios within a few
# percent of 1 and more scatter pointwise.
#
# Outputs (in cfg$out_dir):
#   theorem1_pointwise.csv   one row per (scenario, test point, B)
#   theorem1_summary.csv     averaged over test points
#   fig_variance_vs_B.pdf    Figure 3 replica (+ Mentch-Zhou panel)
#   fig_pointwise.pdf        predicted vs actual variance at each test point
#
# Usage:
#   Rscript theorem1_check.R          # full run (roughly 30-90 min)
#   QUICK=1 Rscript theorem1_check.R  # small smoke test
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
  R        = 300,                         # outcome draws (O'Connell: 300)
  B_max    = 8000,                        # trees per forest (O'Connell: 8,000)
  B_grid   = c(1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000, 8000),
  n_test   = 20,                          # test points (O'Connell: 20)
  run_mz   = TRUE,                        # include the Mentch-Zhou panel
  oc = list(n = 400, mtry = 4, min_node = 5, jitter_sd = 0.02),
  mz = list(n = 100, p = 5, rho = 0.35, snr = 0.05, min_node = 1),
  cores    = max(1, detectCores() - 1),
  seed     = 2026,
  out_dir  = "theorem1_results"
)

if (nzchar(Sys.getenv("QUICK"))) {
  message("QUICK mode: small settings for a smoke test")
  cfg$R      <- 40
  cfg$B_max  <- 500
  cfg$n_test <- 10
}
cfg$B_grid <- sort(unique(c(cfg$B_grid[cfg$B_grid <= cfg$B_max], cfg$B_max)))

dir.create(cfg$out_dir, showWarnings = FALSE)

# Parallelism uses a PSOCK cluster (fresh R processes), not forking: forking is
# blocked in Positron and unsafe in other GUI front ends. Workers each handle a
# block of outcome draws and return running sums, so the full set of tree
# predictions never has to be held in memory at once.
use_cluster <- cfg$cores > 1
if (use_cluster) {
  if (exists("cl", inherits = FALSE)) try(stopCluster(cl), silent = TRUE)
  cl <- makePSOCKcluster(cfg$cores)
  clusterCall(cl, .libPaths, .libPaths())
  clusterEvalQ(cl, suppressPackageStartupMessages(library(ranger)))
}
par_lapply <- function(X, FUN, ...) {
  if (use_cluster) parLapply(cl, X, FUN, ...) else lapply(X, FUN, ...)
}

# ----------------------------------------------------------------------------
# Data: O'Connell Appendix B, continuous outcome, core covariates x1..x12
# ----------------------------------------------------------------------------
# All continuous covariates are standardized after generation (so the scale
# parameters of the Gamma, LogNormal etc. do not matter, only their shapes).
# Binary and categorical covariates are left as 0/1 and 0/1/2. x7 and x10 do
# not enter the mean and act as pure noise, as in his specification.
oc_raw_X <- function(n) {
  cbind(
    x1  = rnorm(n, 0, 1.2),                 # N(0, 1.44)
    x2  = 1.5 * rt(n, df = 5),
    x3  = rnorm(n, 0, 1),
    x4  = rnorm(n, 0, 0.25),                # N(0, 0.0625)
    x5  = rgamma(n, shape = 2, rate = 0.5),
    x6  = rbinom(n, 1, 0.4),
    x7  = rbinom(n, 1, 0.5),
    x8  = sample(0:2, n, replace = TRUE, prob = c(0.6, 0.3, 0.1)),
    x9  = rnorm(n, 0, 3),                   # N(0, 9)
    x10 = rbinom(n, 1, 0.5),
    x11 = rlnorm(n, 0, 0.6),
    x12 = rbeta(n, 2, 5)
  )
}
oc_continuous <- c("x1", "x2", "x3", "x4", "x5", "x9", "x11", "x12")

oc_mu <- function(X, x3sq_mean) {
  x <- function(j) X[, j]
  0.90 * sin(1.1 * x("x1")) + 0.35 * x("x2") +
    0.55 * (x("x3")^2 - x3sq_mean) +       # centered squared term
    0.18 * x("x4") + 0.30 * (x("x5") > 0.4) + 0.22 * x("x6") +
    0.18 * (x("x8") == 1) + 0.28 * (x("x8") == 2) +
    0.45 * sin(x("x11")) + 0.25 * (x("x12") > 1.0) +
    0.18 * x("x1") * x("x2") +
    0.12 * sin(x("x11")) * (x("x12") > 1.0)
}

oc_sigma <- function(X) {
  pmax(0.65 + 0.25 * abs(X[, "x1"]) + 0.15 * abs(X[, "x2"]) +
         0.15 * (X[, "x5"] > 0.4) + 0.12 * (X[, "x12"] > 1.0), 0.15)
}

make_oc <- function(n, n_test, jitter_sd) {
  X <- oc_raw_X(n)
  for (j in oc_continuous) X[, j] <- (X[, j] - mean(X[, j])) / sd(X[, j])
  # Test points: anchored jittered cloud (training rows + small noise on the
  # continuous covariates)
  idx <- sample(n, n_test, replace = TRUE)
  Xte <- X[idx, , drop = FALSE]
  Xte[, oc_continuous] <- Xte[, oc_continuous] +
    matrix(rnorm(n_test * length(oc_continuous), 0, jitter_sd), n_test)
  x3sq_mean <- mean(X[, "x3"]^2)
  list(Xtr = X, Xte = Xte,
       mu_tr = oc_mu(X, x3sq_mean), sd_tr = oc_sigma(X),
       mu_te = oc_mu(Xte, x3sq_mean))
}

# ----------------------------------------------------------------------------
# Data: Mentch & Zhou linear model (as in covfloor_augbagg.R)
# ----------------------------------------------------------------------------
make_mz <- function(n, n_test, p, rho, snr) {
  Sigma <- rho^abs(outer(seq_len(p), seq_len(p), "-"))
  beta  <- rep(1, p)
  sigma2 <- drop(t(beta) %*% Sigma %*% beta) / snr
  X   <- mvrnorm(n, rep(0, p), Sigma)
  Xte <- mvrnorm(n_test, rep(0, p), Sigma)
  colnames(X) <- colnames(Xte) <- paste0("X", seq_len(p))
  list(Xtr = X, Xte = Xte,
       mu_tr = drop(X %*% beta), sd_tr = rep(sqrt(sigma2), n),
       mu_te = drop(Xte %*% beta))
}

# ----------------------------------------------------------------------------
# Worker: a block of outcome draws. All inputs are passed explicitly so the
# function does not depend on anything in the main session.
# ----------------------------------------------------------------------------
fit_block <- function(rs, Xtr, Xte, Y, mtry, min_node, B_max, B_grid,
                      seedsA, seedsB) {
  nt <- nrow(Xte)
  S1 <- matrix(0, nt, B_max)                 # running sum of tree predictions
  S2 <- matrix(0, nt, B_max)                 # running sum of squares
  sub    <- array(NA_real_, c(length(rs), nt, length(B_grid)))
  within <- matrix(NA_real_, length(rs), nt)
  fB     <- matrix(NA_real_, length(rs), nt)
  for (j in seq_along(rs)) {
    r <- rs[j]
    common <- list(x = Xtr, y = Y[, r], num.trees = B_max, mtry = mtry,
                   replace = TRUE, sample.fraction = 1,
                   min.node.size = min_node, num.threads = 1, verbose = FALSE)
    fitA <- do.call(ranger::ranger, c(common, list(seed = seedsA[r])))
    pa <- predict(fitA, data = Xte, predict.all = TRUE,
                  num.threads = 1)$predictions        # nt x B_max
    rm(fitA)
    fitB <- do.call(ranger::ranger, c(common, list(seed = seedsB[r])))
    fB[j, ] <- predict(fitB, data = Xte, num.threads = 1)$predictions
    rm(fitB)

    S1 <- S1 + pa
    S2 <- S2 + pa^2
    cs <- matrix(t(apply(pa, 1, cumsum)), nrow = nt)  # running tree averages
    sub[j, , ] <- sweep(cs[, B_grid, drop = FALSE], 2, B_grid, "/")
    within[j, ] <- apply(pa, 1, var)
  }
  list(rs = rs, S1 = S1, S2 = S2, sub = sub, within = within, fB = fB)
}

# ----------------------------------------------------------------------------
# Run one scenario
# ----------------------------------------------------------------------------
run_scenario <- function(name, dat, mtry, min_node) {
  R <- cfg$R; nt <- nrow(dat$Xte); nB <- length(cfg$B_grid)
  Y <- dat$mu_tr + dat$sd_tr * matrix(rnorm(length(dat$mu_tr) * R),
                                      length(dat$mu_tr), R)
  seedsA <- sample.int(.Machine$integer.max, R)
  seedsB <- sample.int(.Machine$integer.max, R)

  # Blocks of draws, sent out in rounds of one block per core
  n_blocks <- min(R, max(1, cfg$cores) * 5)
  blocks <- split(seq_len(R), cut(seq_len(R), n_blocks, labels = FALSE))
  rounds <- split(blocks, ceiling(seq_along(blocks) / max(1, cfg$cores)))

  S1 <- matrix(0, nt, cfg$B_max); S2 <- S1
  sub    <- array(NA_real_, c(R, nt, nB))
  within <- matrix(NA_real_, R, nt)
  fB     <- matrix(NA_real_, R, nt)

  t0 <- Sys.time(); done <- 0
  for (rd in rounds) {
    res <- par_lapply(rd, fit_block, Xtr = dat$Xtr, Xte = dat$Xte, Y = Y,
                      mtry = mtry, min_node = min_node, B_max = cfg$B_max,
                      B_grid = cfg$B_grid, seedsA = seedsA, seedsB = seedsB)
    for (b in res) {
      S1 <- S1 + b$S1; S2 <- S2 + b$S2
      sub[b$rs, , ] <- b$sub
      within[b$rs, ] <- b$within
      fB[b$rs, ] <- b$fB
      done <- done + length(b$rs)
    }
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    message(sprintf("[%s] %d/%d draws  (%.1f min, ~%.0f min left)",
                    name, done, R, el, el / done * (R - done)))
  }

  # --- Components (each estimated separately) -------------------------------
  fA_full <- sub[, , nB]                                    # R x nt, B = B_max
  C_T <- sapply(seq_len(nt), function(i) cov(fA_full[, i], fB[, i]))
  var_tree_b <- (S2 - S1^2 / R) / (R - 1)                   # nt x B_max
  sig2_T   <- rowMeans(var_tree_b)
  E_sig2_th <- colMeans(within)

  # --- Actual variance and the two predictions, per test point and B --------
  emp <- apply(sub, c(2, 3), var)                           # nt x nB
  Bm  <- matrix(cfg$B_grid, nt, nB, byrow = TRUE)
  thm1 <- sig2_T / Bm + (Bm - 1) / Bm * C_T
  eq8  <- E_sig2_th / Bm + C_T

  data.frame(
    scenario = name,
    test_pt  = rep(seq_len(nt), times = nB),
    B        = rep(cfg$B_grid, each = nt),
    emp_var  = as.vector(emp),
    pred_thm1 = as.vector(thm1),
    pred_eq8  = as.vector(eq8),
    C_T       = rep(C_T, times = nB),
    sig2_T    = rep(sig2_T, times = nB),
    E_sig2_theta = rep(E_sig2_th, times = nB),
    mc_eq8    = as.vector(E_sig2_th / Bm)
  )
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
set.seed(cfg$seed)
scen_labels <- c(
  oconnell = sprintf("O'Connell Fig. 3 setup: n = %d, 12 covariates, mtry = %d, node size %d",
                     cfg$oc$n, cfg$oc$mtry, cfg$oc$min_node),
  mz = sprintf("Mentch-Zhou setup: n = %d, p = %d, bagging, node size %d, SNR = %.2f",
               cfg$mz$n, cfg$mz$p, cfg$mz$min_node, cfg$mz$snr))

t_start <- Sys.time()
message(sprintf("R = %d draws, B_max = %d trees, %d test points, %d cores",
                cfg$R, cfg$B_max, cfg$n_test, cfg$cores))

dat_oc <- make_oc(cfg$oc$n, cfg$n_test, cfg$oc$jitter_sd)
res <- run_scenario("oconnell", dat_oc, cfg$oc$mtry, cfg$oc$min_node)

if (cfg$run_mz) {
  dat_mz <- make_mz(cfg$mz$n, cfg$n_test, cfg$mz$p, cfg$mz$rho, cfg$mz$snr)
  res <- rbind(res, run_scenario("mz", dat_mz, cfg$mz$p, cfg$mz$min_node))
}
if (use_cluster) stopCluster(cl)

write.csv(res, file.path(cfg$out_dir, "theorem1_pointwise.csv"), row.names = FALSE)

# ----------------------------------------------------------------------------
# Summary: average over test points (as in O'Connell's Figure 3)
# ----------------------------------------------------------------------------
summ <- aggregate(cbind(emp_var, pred_thm1, pred_eq8, C_T, mc_eq8) ~ scenario + B,
                  data = res, FUN = mean)
summ$ratio_thm1 <- summ$pred_thm1 / summ$emp_var
summ$ratio_eq8  <- summ$pred_eq8 / summ$emp_var
summ <- summ[order(summ$scenario, summ$B), ]
write.csv(summ, file.path(cfg$out_dir, "theorem1_summary.csv"), row.names = FALSE)

for (s in unique(summ$scenario)) {
  d <- summ[summ$scenario == s, ]
  message("\n", scen_labels[[s]])
  message(sprintf("%6s %12s %12s %12s %10s %10s",
                  "B", "actual var", "Theorem 1", "eq. (8)", "Thm1/act", "eq8/act"))
  for (i in seq_len(nrow(d))) {
    message(sprintf("%6d %12.5f %12.5f %12.5f %10.3f %10.3f",
                    d$B[i], d$emp_var[i], d$pred_thm1[i], d$pred_eq8[i],
                    d$ratio_thm1[i], d$ratio_eq8[i]))
  }
  last <- d[nrow(d), ]
  message(sprintf(
    "At B = %d: actual variance is %.1f%% from the floor C_T (O'Connell reports 1.3%% at B = 8,000)",
    last$B, 100 * abs(last$emp_var - last$C_T) / last$C_T))
}

# ----------------------------------------------------------------------------
# Figures (optional, need ggplot2 >= 3.4 for linewidth)
# ----------------------------------------------------------------------------
if (requireNamespace("ggplot2", quietly = TRUE)) {
  library(ggplot2)
  res$scen_lab  <- factor(scen_labels[res$scenario], levels = scen_labels)
  summ$scen_lab <- factor(scen_labels[summ$scenario], levels = scen_labels)

  # Figure 1: Figure 3 replica. Faint lines: actual variance at each test
  # point. Points: actual variance averaged over test points. Solid line:
  # Theorem 1 prediction. Dashed: Monte Carlo part E[sigma_{T,theta}^2]/B.
  # Horizontal: covariance floor C_T.
  floor_df <- aggregate(C_T ~ scen_lab, data = summ, FUN = mean)
  lev <- c("Actual Var(f_B(x) | X)", "Theorem 1 prediction",
           "Monte Carlo part (1/B) E[sigma^2_T,theta]", "Floor C_T")
  g1 <- ggplot() +
    geom_line(data = res, aes(B, emp_var, group = test_pt),
              colour = "grey80", linewidth = 0.3) +
    geom_hline(data = floor_df, aes(yintercept = C_T, colour = lev[4]),
               linewidth = 0.7) +
    geom_line(data = summ, aes(B, mc_eq8, colour = lev[3]),
              linetype = "dashed", linewidth = 0.7) +
    geom_line(data = summ, aes(B, pred_thm1, colour = lev[2]), linewidth = 0.9) +
    geom_point(data = summ, aes(B, emp_var, colour = lev[1]), size = 2.2) +
    scale_colour_manual(values = setNames(c("black", "#D55E00", "grey40", "#0072B2"), lev),
                        breaks = lev) +
    scale_x_log10(breaks = cfg$B_grid) +
    facet_wrap(~ scen_lab, ncol = 1, scales = "free_y") +
    labs(x = "Number of trees (B)", y = "Var(f_B(x) | X)", colour = NULL,
         title = "Variance of the forest prediction vs number of trees",
         subtitle = sprintf("R = %d outcome draws at fixed X; averaged over %d test points (grey: individual points)",
                            cfg$R, cfg$n_test)) +
    theme_bw() + theme(legend.position = "bottom",
                       axis.text.x = element_text(angle = 45, hjust = 1)) +
    guides(colour = guide_legend(nrow = 2, override.aes = list(
      shape = c(16, NA, NA, NA),
      linetype = c("blank", "solid", "dashed", "solid"))))
  n_scen <- length(unique(res$scenario))
  ggsave(file.path(cfg$out_dir, "fig_variance_vs_B.pdf"), g1,
         width = 9, height = 4.5 * n_scen + 1)

  # Figure 2: pointwise agreement at small, medium and large B
  B_show <- unique(c(1, 10, cfg$B_max))
  pw <- subset(res, B %in% B_show)
  pw <- rbind(
    data.frame(pw[, c("scen_lab", "B", "emp_var")], pred = pw$pred_thm1,
               form = "Theorem 1"),
    data.frame(pw[, c("scen_lab", "B", "emp_var")], pred = pw$pred_eq8,
               form = "eq. (8)"))
  pw$B_lab <- factor(sprintf("B = %d", pw$B), levels = sprintf("B = %d", B_show))
  g2 <- ggplot(pw, aes(emp_var, pred, shape = form, colour = form)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
    geom_point(size = 1.8, alpha = 0.8) +
    scale_colour_manual(values = c("Theorem 1" = "#D55E00", "eq. (8)" = "#0072B2")) +
    scale_shape_manual(values = c("Theorem 1" = 16, "eq. (8)" = 2)) +
    facet_wrap(scen_lab ~ B_lab, scales = "free", ncol = length(B_show),
               labeller = label_wrap_gen(width = 45, multi_line = TRUE)) +
    labs(x = "Actual variance at test point", y = "Predicted variance",
         colour = NULL, shape = NULL,
         title = "Pointwise agreement: predicted vs actual variance",
         subtitle = "Each point is one test point; dotted line is y = x") +
    theme_bw() + theme(legend.position = "bottom",
                       strip.text = element_text(size = 7))
  ggsave(file.path(cfg$out_dir, "fig_pointwise.pdf"), g2,
         width = 10, height = 3.6 * n_scen + 1)
  message("\nWrote figures to ", cfg$out_dir)
} else {
  message("ggplot2 not installed; skipping figures")
}

message(sprintf("Done in %.1f min",
                as.numeric(difftime(Sys.time(), t_start, units = "mins"))))