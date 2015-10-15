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

test_that("Normalize data", {
  df <- data.frame(
    X1 = seq(1, 100, by=2),
    X2 = rnorm(100),
    X3 = rnorm(100, 1000, 30),
    X4 = sample(c(runif(90, 0, 1000), rep(NA, 10))),
    X5 = runif(100, -50, 700),
    X6 = c("alfa", "beta", rep("gama", 98))
  )
  df$Label1 <- c(sample(c(0,1), 100, replace = TRUE))
  df$Label2 <- c(sample(c(0,1), 100, replace = TRUE))
  mdata <- mldr_from_dataframe(df, labelIndices = c(7, 8), name = "testMLDR")

  new.data <- mldr_normalize(mdata)
  for (i in 1:5) {
    new.col <- as.numeric(new.data$dataset[, i])
    expect_equal(max(new.col, na.rm = TRUE),  1)
    expect_equal(min(new.col, na.rm = TRUE),  0)
    expect_equal(which.max(new.col), which.max(df[, i]))
    expect_equal(which.min(new.col), which.min(df[, i]))
  }
  expect_equal(new.data$dataset[, 6], mdata$dataset[, 6])
})

test_that("Remove examples and attributes", {
  df <- data.frame(
    X1 = rep(1, 100),
    X2 = rep(c(1,2), 50),
    X3 = runif(100, 1, 3),
    X4 = rep("XYZ", 100),
    X5 = sample(c("abc", "bcd"), 100, replace = TRUE),
    X6 = c("alfa", "beta", rep("gama", 98)),
    X7 = sample(c(rep(1, 90), rep(NA, 10))),
    X8 = sample(c(rnorm(90), rep(NA, 10)))
  )
  df$Label1 <- rep(0, 100)
  df$Label2 <- sample(c(rep(1, 30), rep(0, 30), sample(c(0,1), 40, replace = TRUE)))
  mdata <- mldr_from_dataframe(df, labelIndices = c(9, 10), name = "testMLDR")

  new.data <- mldr_remove_unique_attributes(mdata)
  expect_equal(new.data$measures$num.attributes, 8)
  expect_named(new.data$dataset[new.data$attributesIndexes], c("X2", "X3", "X5", "X6", "X7", "X8"))

  new.data <- mldr_remove_unlabeled_instances(mdata)
  has.label <- mdata$dataset$Label2 == 1
  expect_equal(new.data$measures$num.instances, sum(has.label))
  expect_equal(new.data$dataset[mdata$attributesIndexes], mdata$dataset[has.label, mdata$attributesIndexes])

  df$Label3 <- c(c(1, 1), rep(0, 98))
  df$Label4 <- c(c(0, 0), rep(1, 98))
  df$Label5 <- rep(1, 100)
  df$Label6 <- c(rep(1, 11), rep(0, 89))
  mdata <- mldr_from_dataframe(df, labelIndices = 9:14, name = "testMLDR")

  new.data <- mldr_remove_labels(mdata)
  expect_equal(new.data$measures$num.labels, 4)
  expect_equal(rownames(new.data$labels), c("Label2", "Label3", "Label4", "Label6"))

  new.data <- mldr_remove_labels(mdata, 2)
  expect_equal(new.data$measures$num.labels, 2)
  expect_equal(rownames(new.data$labels), c("Label2", "Label6"))

  new.data <- mldr_remove_labels(mdata, 10)
  expect_equal(new.data$measures$num.labels, 2)
  expect_equal(rownames(new.data$labels), c("Label2", "Label6"))

  expect_error(mldr_remove_labels(mdata, 11))
})
