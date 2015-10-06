data("iris")
context("Base learner")
testdataset <- list(data = iris[1:100,], labelindex = 5, labelname = "Species")
testdataset$data[,5] <- as.factor(as.numeric(testdataset$data[,5] == "versicolor"))

test_that("as binary prediction", {
  probs <- runif(10, 0, 1)
  result <- as.binaryPrediction(probs)
  expect_is(result, "mlresult")
  expect_false(is.null(result$bipartition))
  expect_false(is.null(result$probability))
  expect_equal(result$probability, probs)

  names(probs) <- 11:20
  result <- as.binaryPrediction(probs)
  expect_named(result$bipartition, as.character(11:20))
  expect_named(result$probability, as.character(11:20))

  probs <- rep(0.5, 10)
  result <- as.binaryPrediction(probs, 0.4)
  expect_equal(result$bipartition, rep(1, 10))

  result <- as.binaryPrediction(probs, 0.5)
  expect_equal(result$bipartition, rep(1, 10))

  result <- as.binaryPrediction(probs, 0.6)
  expect_equal(result$bipartition, rep(0, 10))
})

test_that("test train/prediction base learner methods", {
  methods <- c("baseSVM", "baseJ48", "baseC4.5", "baseC5.0", "baseCART", "baseRF", "baseNB", "baseKNN")
  names(methods) <- c("svm", "J48", "J48", "C5.0", "rpart", "randomForest", "naiveBayes", "baseKNN")
  for (i in 1:length(methods)) {
    class(testdataset) <- methods[i]
    model <- mltrain(testdataset)
    expect_is(model, names(methods)[i])
    result <- mlpredict(model, iris[11:20, 1:4])
    expect_is(result, "mlresult")
    expect_named(result$bipartition, as.character(11:20))
    expect_named(result$bipartition, names(result$probability))
  }
  class(testdataset) <- NULL
  expect_error(mltrain(testdataset))
  expect_error(mlpredict(testdataset))
 })

test_that("Summary", {
  class(testdataset) <- "mltransformation"
  expect_equal(summary(testdataset), summary(testdataset$data))
})


