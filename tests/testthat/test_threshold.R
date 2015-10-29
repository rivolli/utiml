context("threshold methods")
result <- matrix(
  data = c(0.68, 0.03, 0.79, 0.2, 0.81, 0.18, 0.01, 0.8, 0.8, 0.24, 0.94, 0.34, 0.31, 0.68, 0.35),
  ncol = 3,
  dimnames = list(11:15, c("lbl1",  "lb2", "lb3"))
)
predictions <- apply(result, 2, as.binaryPrediction)
mlresult <- as.multilabelPrediction(predictions, F)

# result
#    lbl1  lb2  lb3
# 11 0.68 0.18 0.94
# 12 0.03 0.01 0.34
# 13 0.79 0.80 0.31
# 14 0.20 0.80 0.68
# 15 0.81 0.24 0.35

test_that("Simple threshold", {
  expect_equal(simple.threshold(result), as.bipartition(mlresult))

  new.data <- simple.threshold(result, max(result))
  expect_equal(apply(new.data, 1, sum), c('11'=1, '12'=1, '13'=1, '14'=1, '15'=1))
  new.data <- simple.threshold(result, min(result))
  expect_equal(apply(new.data, 1, sum), c('11'=3, '12'=3, '13'=3, '14'=3, '15'=3))
  new.data <- simple.threshold(result, c(0.8, 0.2, 0.5))
  expect_equal(new.data[,"lbl1"], c('11'=0, '12'=0, '13'=0, '14'=0, '15'=1))
  expect_equal(new.data[,"lbl2"], c('11'=0, '12'=0, '13'=1, '14'=1, '15'=1))
  expect_equal(new.data[,"lbl3"], c('11'=1, '12'=1, '13'=0, '14'=1, '15'=0))
})
