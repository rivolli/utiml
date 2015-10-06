context("Internal tests")
mydata <- data.frame(
  class1 = runif(10, min = 0, max = 1),
  class2 = factor(as.numeric(runif(10, min = 0, max = 1) > 0.5), levels = c("0", "1"))
)

test_that("Result ML prediction", {
  set.seed(1)
  predictions <- list(
    class1 = as.binaryPrediction(runif(10, min = 0, max = 1)),
    class2 = as.binaryPrediction(runif(10, min = 0, max = 1))
  )
  result1 <- as.resultMLPrediction(predictions, TRUE)
  expect_null(rownames(result1))
  expect_equal(colnames(result1), c("class1", "class2"))

  expect_equal(predictions$class1$probability, result1[,"class1"])
  expect_equal(predictions$class2$probability, result1[,"class2"])

  result2 <- as.resultMLPrediction(predictions, FALSE)
  expect_equal(predictions$class1$bipartition, result2[,"class1"])
  expect_equal(predictions$class2$bipartition, result2[,"class2"])

  expect_equivalent(attr(result1, "classes"), result2)
  expect_equivalent(attr(result2, "probs"), result1)

  values <- runif(10, min = 0, max = 1)
  names(values) <- 6:15
  predictions <- list(
    class1 = as.binaryPrediction(values),
    class2 = as.binaryPrediction(values)
  )
  result <- as.resultMLPrediction(predictions, TRUE)
  expect_equal(rownames(result), as.character(6:15))
  expect_equal(result[,"class1"], result[,"class2"])
  result <- as.resultMLPrediction(predictions, FALSE)
  expect_equal(rownames(result), as.character(6:15))
  expect_equal(result[,"class1"], result[,"class2"])
  set.seed(NULL)
})

test_that("BR transformation", {
  dataset <- br.transformation(mydata, "testDataset", "SVM")
  expect_is(dataset, "testDataset")
  expect_is(dataset, "baseSVM")
  expect_is(dataset, "mltransformation")

  expect_equal(dataset$data, mydata)
  expect_equal(dataset$labelname, "class2")
  expect_equal(dataset$labelindex, 2)
  expect_equal(dataset$methodname, "SVM")

  dataset <- br.transformation(mydata, "onlytest", "XYZ", extra1="abc", extra2=1:10)
  expect_is(dataset, "onlytest")
  expect_is(dataset, "baseXYZ")
  expect_is(dataset, "mltransformation")

  expect_equal(dataset$data, mydata)
  expect_equal(dataset$labelname, "class2")
  expect_equal(dataset$labelindex, 2)
  expect_equal(dataset$methodname, "XYZ")
  expect_equal(dataset$extra1, "abc")
  expect_equal(dataset$extra2, 1:10)
})

test_that("br.create_model and br.predict_model", {
  dataset <- br.transformation(mydata, "testdata", "KNN")
  model <- br.create_model(dataset, k=3)
  expect_equal(attr(model, "labelname"), "class2")
  expect_equal(attr(model, "methodname"), "KNN")

  predict1 <- br.predict_model(model, mydata[,1, drop = FALSE])
  expect_is(predict1, "mlresult")

  model <- br.create_model(dataset)
  predict2 <- br.predict_model(model, mydata[,1, drop = FALSE], k=3)
  expect_equal(predict1$probability, predict2$probability)
  expect_true(all(predict1$probability == predict2$probability))

  predict3 <- br.predict_model(model, mydata[,1, drop = FALSE], k=1)
  expect_false(all(predict2$probability == predict3$probability))
})

test_that("New data", {
  test <- emotions$dataset[,emotions$attributesIndexes]
  expect_equal(utiml_newdata(test), test)
  expect_equal(utiml_newdata(emotions), test)
})
