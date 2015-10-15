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
  #Test the stratified
})

test_that("iterative holdout", {
  f <- mldr_iterative_stratification_holdout(mdata, c(0.4, 0.4, 0.1, 0.1), c("a", "b", "c", "d"))
  expect_equal(length(f), 4)
  expect_named(f, c("a", "b", "c", "d"))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[2]]$labels))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[3]]$labels))
  #Test the stratified
})

test_that("random kfold", {
  f <- mldr_random_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_equal(length(f$fold[[i]]), 10)

  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  f1 <- mldr_random_kfold(mdata, 4, SEED = 1)
  f2 <- mldr_random_kfold(mdata, 4, SEED = 1)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))

  f3 <- mldr_random_kfold(mdata, 3)
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 32)
  expect_more_than(length(f3$fold[[2]]), 32)
  expect_more_than(length(f3$fold[[3]]), 32)
})

test_that("stratified kfold", {
  f <- mldr_stratified_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_equal(length(f$fold[[i]]), 10)

  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  f1 <- mldr_stratified_kfold(mdata, 4, SEED = 1)
  f2 <- mldr_stratified_kfold(mdata, 4, SEED = 1)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))

  f3 <- mldr_stratified_kfold(mdata, 3)
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 32)
  expect_more_than(length(f3$fold[[2]]), 32)
  expect_more_than(length(f3$fold[[3]]), 32)
})

test_that("iterative kfold", {
  f <- mldr_iterative_stratification_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_more_than(length(f$fold[[i]]), 7)

  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  f1 <- mldr_iterative_stratification_kfold(mdata, 4, SEED = 1)
  f2 <- mldr_iterative_stratification_kfold(mdata, 4, SEED = 1)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))

  f3 <- mldr_iterative_stratification_kfold(mdata, 3)
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 30)
  expect_more_than(length(f3$fold[[2]]), 30)
  expect_more_than(length(f3$fold[[3]]), 30)
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
