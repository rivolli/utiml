context("Sampling tests")
set.seed(1)
df <- data.frame(matrix(rnorm(1000), ncol = 10))
df$Label1 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 100, replace = TRUE))
df$Label4 <- as.numeric(df$Label1 == 0 | df$Label2 == 0 | df$Label3 == 0)
mdata <- mldr_from_dataframe(df, labelIndices = c(11, 12, 13, 14), name = "testMLDR")
set.seed(NULL)

testFolds <- function (kfold, original, msg) {
  real <- unlist(lapply(kfold$fold, names))
  expected <- unique(real)
  expect_true(all(expected == real), label=msg)
  expect_true(all(sort(unlist(kfold$fold)) == 1:original$measures$num.instances), label=msg)
  expect_true(all(sort(real) == sort(rownames(original$dataset))), label=msg)
}

testEmptyIntersectRows <- function (a, b) {
  expect_equal(length(intersect(rownames(a$dataset), rownames(b$dataset))),0)
}

testCompletude <- function (list, original) {
  names <- sort(unlist(lapply(list, function (fold)  rownames(fold$dataset))))
  expect_true(all(names == sort(rownames(original$dataset))))
}

test_that("random holdout", {
  folds <- mldr_random_holdout(mdata, 0.7)
  expect_equal(length(folds), 2)
  expect_is(folds[[1]], "mldr")
  expect_is(folds[[2]], "mldr")
  expect_equal(folds[[1]]$measures$num.instances, 70)
  expect_equal(folds[[2]]$measures$num.instances, 30)
  expect_equal(rownames(folds[[1]]$labels), rownames(folds[[2]]$labels))
  testEmptyIntersectRows(folds[[1]], folds[[2]])
  testCompletude(folds, mdata)

  subfolds <- mldr_random_holdout(folds[[1]], 0.5)
  testEmptyIntersectRows(subfolds[[1]], subfolds[[2]])
  testCompletude(subfolds, folds[[1]])

  folds <- mldr_random_holdout(mdata, c("train"=0.5, "test"=0.5))
  expect_named(folds, c("train", "test"))
  expect_equal(folds$train$measures$num.instances, folds$test$measures$num.instances)
  testCompletude(folds, mdata)

  set.seed(1)
  f1 <- mldr_random_holdout(mdata, c(0.5, 0.5))
  set.seed(1)
  f2 <- mldr_random_holdout(mdata, c(0.5, 0.5))
  expect_equal(f1, f2)
  set.seed(NULL)

  expect_error(mldr_random_holdout(mdata, NULL))
})

test_that("stratified holdout", {
  f <- mldr_stratified_holdout(mdata, c("a"=0.4, "b"=0.4, "c"=0.2))
  expect_equal(length(f), 3)
  expect_named(f, c("a", "b", "c"))
  expect_equal(f[[1]]$measures$num.instances, 40)
  expect_equal(f[[2]]$measures$num.instances, 40)
  expect_equal(f[[3]]$measures$num.instances, 20)
  expect_equal(rownames(f[[1]]$labels), rownames(f[[2]]$labels))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[3]]$labels))

  testEmptyIntersectRows(f$a, f$b)
  testEmptyIntersectRows(f$a, f$c)
  testEmptyIntersectRows(f$b, f$c)
  testCompletude(f, mdata)

  sf <- mldr_stratified_holdout(f$a, c("a"=0.5, "b"=0.5))
  expect_equal(length(sf), 2)
  testEmptyIntersectRows(sf$a, sf$b)
  testCompletude(sf, f$a)
})

test_that("iterative holdout", {
  f <- mldr_iterative_stratification_holdout(mdata, c("a"=0.4, "b"=0.4, "c"=0.1, "d"=0.1))
  expect_equal(length(f), 4)
  expect_named(f, c("a", "b", "c", "d"))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[2]]$labels))
  expect_equal(rownames(f[[1]]$labels), rownames(f[[3]]$labels))

  testEmptyIntersectRows(f$a, f$b)
  testEmptyIntersectRows(f$a, f$c)
  testEmptyIntersectRows(f$a, f$d)
  testEmptyIntersectRows(f$b, f$c)
  testEmptyIntersectRows(f$b, f$d)
  testEmptyIntersectRows(f$c, f$d)
  testCompletude(f, mdata)

  sf <- mldr_stratified_holdout(f$a, c("a"=0.5, "b"=0.5))
  expect_equal(length(sf), 2)
  testEmptyIntersectRows(sf$a, sf$b)
  testCompletude(sf, f$a)

  folds <- mldr_random_holdout(mdata, 0.6)
  sf <- mldr_stratified_holdout(folds[[2]], c("a"=0.6, "b"=0.4))
  testEmptyIntersectRows(sf$a, sf$b)
  testCompletude(sf, folds[[2]])
})

test_that("random kfold", {
  f <- mldr_random_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_equal(length(f$fold[[i]]), 10)
  testFolds(f, mdata, "f Random kfolds")

  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  set.seed(1)
  f1 <- mldr_random_kfold(mdata, 4)
  testFolds(f1, mdata, "f1 Random kfolds")
  set.seed(1)
  f2 <- mldr_random_kfold(mdata, 4)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))
  set.seed(NULL)

  f3 <- mldr_random_kfold(mdata, 3)
  testFolds(f3, mdata, "f3 Random kfolds")
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 32)
  expect_more_than(length(f3$fold[[2]]), 32)
  expect_more_than(length(f3$fold[[3]]), 32)

  ds <- mldr_random_holdout(mdata, c("train" = 0.9, "test" = 0.1))
  f4 <- mldr_random_kfold(ds$train, 9)
  testFolds(f4, ds$train, "f4 Random kfolds")
})

test_that("stratified kfold", {
  f <- mldr_stratified_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_equal(length(f$fold[[i]]), 10)

  testFolds(f, mdata, "f Stratified kfold")
  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  set.seed(1)
  f1 <- mldr_stratified_kfold(mdata, 4)
  testFolds(f1, mdata, "f1 Stratified kfold")
  set.seed(1)
  f2 <- mldr_stratified_kfold(mdata, 4)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))
  set.seed(NULL)

  f3 <- mldr_stratified_kfold(mdata, 3)
  testFolds(f3, mdata, "f3 Stratified kfold")
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 32)
  expect_more_than(length(f3$fold[[2]]), 32)
  expect_more_than(length(f3$fold[[3]]), 32)

  ds <- mldr_random_holdout(mdata, c("train" = 0.9, "test" = 0.1))
  f4 <- mldr_stratified_kfold(ds$train, 9)
  testFolds(f4, ds$train, "f4 Stratified kfold")
})

test_that("iterative kfold", {
  f <- mldr_iterative_stratification_kfold(mdata, 10)
  expect_is(f, "mldr_kfolds")
  expect_equal(f$k, 10)
  expect_equal(length(f$fold), 10)
  for (i in 1:10)
    expect_more_than(length(f$fold[[i]]), 7)

  testFolds(f, mdata, "f Iterative kfold")
  fdata1 <- mldr_getfold(mdata, f, 1)
  fdata2 <- mldr_getfold(mdata, f, 10)

  expect_equal(rownames(fdata1$labels), rownames(fdata2$labels))
  expect_equal(fdata1$measures$num.instances, fdata2$measures$num.instances)

  set.seed(1)
  f1 <- mldr_iterative_stratification_kfold(mdata, 4)
  testFolds(f1, mdata, "f1 Iterative kfold")
  set.seed(1)
  f2 <- mldr_iterative_stratification_kfold(mdata, 4)
  expect_equal(length(f1$fold), 4)
  expect_equal(length(f1$fold[[2]]), 25)
  expect_equal(f1, f2)
  expect_false(all(f$fold[[1]] == f1$fold[[1]]))
  set.seed(NULL)

  f3 <- mldr_iterative_stratification_kfold(mdata, 3)
  testFolds(f3, mdata, "f3 Iterative kfold")
  expect_equal(f3$k, 3)
  expect_equal(length(unlist(f3$fold)), 100)
  expect_more_than(length(f3$fold[[1]]), 30)
  expect_more_than(length(f3$fold[[2]]), 30)
  expect_more_than(length(f3$fold[[3]]), 30)

  ds <- mldr_random_holdout(mdata, c("train" = 0.9, "test" = 0.1))
  f4 <- mldr_iterative_stratification_kfold(ds$train, 9)
  testFolds(f4, ds$train, "f4 Iterative kfold")
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
