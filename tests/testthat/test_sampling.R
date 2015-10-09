context("Sampling tests")
set.seed(1)
df <- data.frame(matrix(rnorm(1000), ncol = 10))
df$Label1 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label4 <- as.numeric(df$Label1 == 0 | df$Label2 == 0 | df$Label3 == 0)
mdata <- mldr_from_dataframe(df, labelIndices = c(11, 12, 13, 14), name = "testMLDR")
set.seed(NULL)

test_that("random holdout", {
  folds <- mldr_random_holdout(mdata, 0.7)
  expect_equal(length(folds), 2)
  expect_is(folds[[1]], "mldr")
  expect_is(folds[[2]], "mldr")
  expect_equal(folds[[1]]$measures$num.instances, 70)
  expect_equal(folds[[2]]$measures$num.instances, 30)
  expect_equal(rownames(folds[[1]]$labels), rownames(folds[[2]]$labels))

  folds <- mldr_random_holdout(mdata, c(0.5, 0.5), c("train", "test"))
  expect_named(folds, c("train", "test"))
  expect_equal(folds$train$measures$num.instances, folds$test$measures$num.instances)
  names <- sort(c(rownames(folds$train$dataset), rownames(folds$test$dataset)))
  expect_equal(names, sort(rownames(mdata$dataset)))

  f1 <- mldr_random_holdout(mdata, c(0.5, 0.5), SEED = 1)
  f2 <- mldr_random_holdout(mdata, c(0.5, 0.5), SEED = 1)
  expect_equal(f1, f2)

  expect_error(mldr_random_holdout(mdata, NULL))
})

test_that("stratified holdout", {
  f <- mldr_stratified_holdout(mdata, c(0.4, 0.4, 0.2), c("a", "b", "c"))
  expect_equal(length(f), 3)
  expect_named(f, c("a", "b", "c"))
  expect_equal(f[[1]]$measures$num.instances, 40)
  expect_equal(f[[2]]$measures$num.instances, 40)
  expect_equal(f[[3]]$measures$num.instances, 20)
  expect_equal(rownames(f[[1]]$labels), rownames(f[[2]]$labels))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[3]]$labels))
  expect_true(all(abs(f[[1]]$labelsets - f[[2]]$labelsets) < 4))
  expect_true(all(abs(f[[1]]$labelsets - f[[3]]$labelsets * 2) < 4))
})

test_that("iterative holdout", {
  f <- mldr_iterative_stratification_holdout(mdata, c(0.4, 0.4, 0.1, 0.1), c("a", "b", "c", "d"))
  expect_equal(length(f), 4)
  expect_named(f, c("a", "b", "c", "d"))
  expect_less_than(abs(f[[1]]$measures$num.instances - f[[2]]$measures$num.instances), 5)
  expect_less_than(abs(f[[3]]$measures$num.instances - f[[4]]$measures$num.instances), 5)
  expect_equal(rownames(f[[1]]$labels), rownames(f[[2]]$labels))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[3]]$labels))
  expect_true(all(abs(f[[1]]$labels$count - f[[2]]$labels$count) < 4))
  expect_true(all(abs(f[[3]]$labels$count - f[[4]]$labels$count) < 4))
})

test_that("random kfold", {
})

test_that("stratified kfold", {
})

test_that("iterative kfold", {
})

test_that("subset and random subset", {
  rows <- 10:20
  cols <- 3:7

  data <- mldr_subset(mdata, rows, 1:10)
  expect_is(data, "mldr")
  expect_equal(data$measures$num.attributes, mdata$measures$num.attributes)

  data <- mldr_subset(mdata, 1:100, cols)
  expect_equal(data$measures$num.instances, mdata$measures$num.instances)
  expect_equal(data$dataset[data$labels$index], mdata$dataset[mdata$labels$index])

  data1 <- mldr_subset(mdata, rows, cols)
  data2 <- mldr_subset(mdata, rows, cols)
  expect_equal(data1, data2)

  data <- mldr_random_subset(mdata, 20, 5)
  expect_equal(data$measures$num.instances, 20)
  expect_equal(data$measures$num.attributes, 5 + data$measures$num.labels)
  expect_equal(data$dataset[,data$labels$index], mdata$dataset[rownames(data$dataset),mdata$labels$index])
})
