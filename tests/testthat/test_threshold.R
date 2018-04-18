context("threshold methods")
result <- matrix(
  data = c(0.68, 0.03, 0.79, 0.2, 0.81,
           0.18, 0.01, 0.8, 0.8, 0.24,
           0.94, 0.34, 0.31, 0.68, 0.35),
  ncol = 3,
  dimnames = list(11:15, c("lbl1",  "lbl2", "lbl3"))
)
mlresult <- as.mlresult(result, probability = FALSE)

# result
#    lbl1  lbl2  lbl3
# 11 0.68  0.18  0.94
# 12 0.03  0.01  0.34
# 13 0.79  0.80  0.31
# 14 0.20  0.80  0.68
# 15 0.81  0.24  0.35

test_that("Fixed threshold", {
  crisp <- fixed_threshold(result)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(crisp, mlresult)

  newdata <- fixed_threshold(result, max(result))
  expect_is(newdata, "mlresult")
  expect_true(is.bipartition(newdata))
  expect_equal(rowSums(newdata), c(1, 1, 1, 1, 1), check.names = FALSE)
  newdata <- fixed_threshold(result, min(result))
  expect_is(newdata, "mlresult")
  expect_equal(rowSums(newdata), c(3, 3, 3, 3, 3), check.names = FALSE)

  newdata <- fixed_threshold(result, c(0.8, 0.2, 0.5))
  expect_is(newdata, "mlresult")
  expect_equal(newdata[, "lbl1"], c('11'=0, '12'=0, '13'=0, '14'=0, '15'=1))
  expect_equal(newdata[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=1))
  expect_equal(newdata[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=1, '15'=0))
})

test_that("Fixed with multiples thershold", {
  set.seed(1)
  pred <- matrix(runif(30, 0, 1), ncol = 3)
  colnames(pred) <- c("a","b","c")
  rownames(pred) <- seq(1, 30, 3)

  result <- fixed_threshold(pred, c(0, 0.5, 1))
  expect_equal(dimnames(result), dimnames(pred))

  expect_equal(result[,1], rep(1, 10), check.names=FALSE)
  expect_equal(result[,2], as.numeric(pred[,2] >= 0.5), check.names=FALSE)
  expect_equal(result[,3], rep(0, 10), check.names=FALSE)
})

test_that("Lcard threshold", {
  # result
  #    lbl1  lbl2  lbl3
  # 11 0.68  0.18  0.94
  # 12 0.03  0.01  0.34
  # 13 0.79  0.80  0.31
  # 14 0.20  0.80  0.68
  # 15 0.81  0.24  0.35
  crisp <- lcard_threshold(result, 1)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(crisp[, "lbl1"], c('11'=0, '12'=0, '13'=1, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=0, '15'=0))

  crisp <- lcard_threshold(result, 2)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(crisp[, "lbl1"], c('11'=1, '12'=0, '13'=1, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))

  crisp <- lcard_threshold(result, 3)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(crisp[, "lbl1"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
})

test_that("MCut threshold", {
  crisp <- mcut_threshold(result)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=1, '12'=0, '13'=1, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=1, '15'=0))

  bipartition <- mcut_threshold(mlresult)
  expect_is(bipartition, "mlresult")
  expect_true(is.bipartition(bipartition))
  expect_equal(as.probability(bipartition), as.probability(mlresult))
  expect_equal(as.bipartition(mcut_threshold(bipartition)),
               as.bipartition(bipartition))
})

test_that("PCut threshold", {
  crisp <- pcut_threshold(result, 0.20)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=0, '12'=0, '13'=0, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=0, '15'=0))

  crisp <- pcut_threshold(result, c(0.2, 0.3, 0.5))
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=0, '12'=0, '13'=0, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=1, '15'=1))

  bipartition <- pcut_threshold(mlresult, 0.3)
  expect_is(bipartition, "mlresult")
  expect_true(is.bipartition(bipartition))
  expect_equal(as.probability(bipartition), as.probability(mlresult))
  expect_equal(as.bipartition(pcut_threshold(bipartition, 0.3)),
               as.bipartition(bipartition))
})

test_that("RCut threshold", {
  crisp <- rcut_threshold(result, 1)
  expect_is(crisp, "mlresult")
  expect_true(is.bipartition(crisp))
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=0, '12'=0, '13'=0, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=0, '15'=0))

  crisp <- rcut_threshold(result, 2)
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=1, '12'=1, '13'=1, '14'=0, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=0))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=0, '14'=1, '15'=1))

  crisp <- rcut_threshold(result, 3)
  expect_equal(dimnames(crisp), dimnames(result))
  expect_equal(crisp[, "lbl1"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
  expect_equal(crisp[, "lbl2"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
  expect_equal(crisp[, "lbl3"], c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))

  bipartition <- rcut_threshold(mlresult, 2)
  expect_is(bipartition, "mlresult")
  expect_true(is.bipartition(bipartition))
  expect_equal(as.probability(bipartition), as.probability(mlresult))
  expect_equal(as.bipartition(rcut_threshold(bipartition, 2)),
               as.bipartition(bipartition))
})

test_that("SCut threshold", {
  thresholds <- scut_threshold(result, mlresult)
  expected <- c(lbl1 = 0.68, lbl2 = 0.8, lbl3 = 0.34)
  expect_equal(thresholds, expected)

  thresholds2 <- scut_threshold(mlresult, mlresult)
  expect_equal(thresholds, thresholds2)

  classes <- as.bipartition(mlresult)
  classes[, 3] <- 0
  thresholds <- scut_threshold(result, classes)
  expect_equal(thresholds[1], thresholds2[1])
  expect_equal(thresholds[2], thresholds2[2])
  expect_gt(thresholds[3], max(result[, 3]))

  expect_error(scut_threshold(result, mlresult, function (){}))
  expect_error(scut_threshold(result, mlresult, CORES = 0))
})

test_that("Subset correction", {
  prediction <- subset_correction(mlresult, as.bipartition(mlresult))
  expect_is(prediction, "mlresult")
  expect_equal(as.probability(prediction), as.probability(mlresult))
})
