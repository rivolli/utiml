data("iris")
context("Base learner")
testdataset <- list(data = iris[1:100,], labelindex = 5, labelname = "Species")
testdataset$data[,5] <- as.factor(as.numeric(testdataset$data[,5] == "versicolor"))

test_that("test train/prediction base learner methods", {
  methods <- c("baseSVM", "baseJ48", "baseC5.0", "baseCART", "baseRF", "baseNB", "baseKNN")
  names(methods) <- c("svm", "J48", "C5.0", "rpart", "randomForest", "naiveBayes", "baseKNN")
  for (i in 1:length(methods)) {
    class(testdataset) <- methods[i]
    model <- mltrain(testdataset)
    expect_is(model, names(methods)[i])
    result <- mlpredict(model, iris[11:20, 1:4])
    expect_is(result, "matrix")
    expect_equal(rownames(result), as.character(11:20))
    expect_equal(colnames(result), c("0", "1"))
  }
  class(testdataset) <- NULL
  expect_error(mltrain(testdataset))
  expect_error(mlpredict(testdataset))
})

test_that("KNN", {
  #Train with a k and predict with other k
})

test_that("Summary", {
  class(testdataset) <- "mltransformation"
  expect_equal(summary(testdataset), summary(testdataset$data))
})


