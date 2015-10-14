context("Pre-process tests")

test_that("Sparce data", {
  df <- data.frame(
    X1 = factor(c("1", "2", rep(NA, 98))),
    X2 = c(1, 2, rep(NA, 98)),
    X3 = factor(c("a", "b", rep(NA, 98))),
    X4 = c("1", "2", rep(NA, 98)),
    X5 = c("a", "b", rep(NA, 98)),
    X6 = c("alfa", "beta", rep(NA, 98))
  )
  df$Label1 <- c(sample(c(0,1), 100, replace = TRUE))
  df$Label2 <- c(sample(c(0,1), 100, replace = TRUE))
  mdata <- mldr_from_dataframe(df, labelIndices = c(7, 8), name = "testMLDR")

  new.data <- mldr_fill_sparce_data(mdata)
  expect_equal(as.numeric(new.data$dataset[, 1]),  c(1, 2, rep(0, 98)))
  expect_equal(as.numeric(new.data$dataset[, 2]),  c(1, 2, rep(0, 98)))
  expect_equal(as.character(new.data$dataset[, 3]),  c("a", "b", rep("", 98)))
  expect_equal(as.numeric(new.data$dataset[, 4]),  c(1, 2, rep(0, 98)))
  expect_equal(as.character(new.data$dataset[, 5]),  c("a", "b", rep("", 98)))
  expect_equal(as.character(new.data$dataset[, 6]),  c("alfa", "beta", rep("", 98)))
})
