context("Prediction utilities tests")
mydata <- data.frame(
  class1 = runif(10, min = 0, max = 1),
  class2 = factor(as.numeric(runif(10, min = 0, max = 1) > 0.5),
                  levels = c("0", "1"))
)

test_that("as binary prediction", {
  probs <- runif(10, 0, 1)
  result <- as.binaryPrediction(probs)
  expect_is(result, "binary.prediction")
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

test_that("Result ML prediction", {
  set.seed(1)
  predictions <- list(
    class1 = as.binaryPrediction(runif(10, min = 0, max = 1)),
    class2 = as.binaryPrediction(runif(10, min = 0, max = 1))
  )
  result1 <- as.multilabelPrediction(predictions, TRUE)
  mresult1 <- as.matrix(result1)
  expect_null(rownames(result1))
  expect_equal(colnames(result1), c("class1", "class2"))

  expect_equal(predictions$class1$probability, mresult1[,"class1"])
  expect_equal(predictions$class2$probability, mresult1[,"class2"])

  result2 <- as.multilabelPrediction(predictions, FALSE)
  mresult2 <- as.matrix(result2)
  TP1 <- predictions$class1$bipartition == 1
  TP2 <- predictions$class2$bipartition == 1
  expect_equal(predictions$class1$bipartition[TP1], mresult2[,"class1"][TP1])
  expect_equal(predictions$class2$bipartition[TP2], mresult2[,"class2"][TP2])
  expect_true(all(mresult2[,"class1"][!TP1] | mresult2[,"class2"][!TP1]))
  expect_true(all(mresult2[,"class1"][TP2] | mresult2[,"class2"][TP2]))
  filter <- !TP1 & !TP2
  expect_true(all(mresult2[,"class1"][filter] != mresult2[,"class2"][filter]))

  expect_true(all(attr(result1, "classes") == mresult2))
  expect_true(all(attr(result2, "probs") == mresult1))
  expect_equal(attr(result1, "type"), "probability")
  expect_equal(attr(result2, "type"), "bipartition")

  values <- runif(10, min = 0, max = 1)
  names(values) <- 6:15
  predictions <- list(
    class1 = as.binaryPrediction(values),
    class2 = as.binaryPrediction(values)
  )
  result <- as.multilabelPrediction(predictions, TRUE)
  mresult <- as.matrix(result)
  expect_equal(rownames(result), as.character(6:15))
  expect_equal(mresult[,"class1"], mresult[,"class2"])
  result <- as.multilabelPrediction(predictions, FALSE)
  expect_equal(rownames(result), as.character(6:15))
  set.seed(NULL)
})

test_that("Filter ML Result", {
  set.seed(1234)
  labels <- matrix(utiml_normalize(rnorm(150)), ncol = 10)
  bipartition <- fixed_threshold(labels, 0.5)
  colnames(labels) <- colnames(bipartition) <- paste("label", 1:10, sep='')
  mlresult1 <- get_multilabel_prediction(bipartition, labels, TRUE)
  mlresult2 <- get_multilabel_prediction(bipartition, labels, FALSE)

  expect_is(mlresult1[1:3, ], "mlresult")
  expect_is(mlresult1[1:3], "mlresult")
  expect_equal(mlresult1[1:3, ], mlresult1[1:3])
  expect_is(mlresult1[1:3, 1:3], "matrix")
  expect_is(mlresult1[, 1:5], "matrix")
  expect_is(mlresult1[, 1, drop = FALSE], "matrix")
  expect_is(mlresult1[, 1], "numeric")

  expect_true(is.probability(mlresult1[1:3, ]))
  expect_true(is.bipartition(mlresult2[1:3, ]))

  expect_equal(mlresult1[, c("label1", "label3")],
               labels[, c("label1", "label3")])
  expect_equal(mlresult2[, c("label1", "label3")],
               bipartition[, c("label1", "label3")])
  expect_equal(mlresult1[, 3:6], labels[, 3:6])
  expect_equal(mlresult2[, 2:8], bipartition[, 2:8])
  expect_equal(mlresult1[1:5, 1:5], labels[1:5, 1:5])
  expect_equal(mlresult2[2:8, 2:8], bipartition[2:8, 2:8])

  mlresult3 <- mlresult1[1:8]
  expect_equal(as.probability(mlresult3), labels[1:8, ])
  expect_equal(as.bipartition(mlresult3), bipartition[1:8, ])
})

test_that("BR prepare", {
  dataset <- prepare_br_data(mydata, "testDataset", "SVM")
  expect_is(dataset, "testDataset")
  expect_is(dataset, "baseSVM")
  expect_is(dataset, "mltransformation")

  expect_equal(dataset$data, mydata)
  expect_equal(dataset$labelname, "class2")
  expect_equal(dataset$labelindex, 2)
  expect_equal(dataset$methodname, "SVM")

  dataset <- prepare_br_data(mydata, "onlytest", "XYZ",
                               extra1="abc", extra2=1:10)
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
  dataset <- prepare_br_data(mydata, "testdata", "KNN")
  model <- create_br_model(dataset, k=3)
  expect_equal(attr(model, "labelname"), "class2")
  expect_equal(attr(model, "methodname"), "KNN")

  predict1 <- predict_br_model(model, mydata[,1, drop = FALSE])
  expect_is(predict1, "binary.prediction")

  model <- create_br_model(dataset)
  predict2 <- predict_br_model(model, mydata[,1, drop = FALSE], k=3)
  expect_equal(predict1$probability, predict2$probability)
  expect_true(all(predict1$probability == predict2$probability))

  predict3 <- predict_br_model(model, mydata[,1, drop = FALSE], k=1)
  expect_false(all(predict2$probability == predict3$probability))
})

test_that("create_br_data", {
  dataset <- create_br_data(toyml, "y1")
  expect_equal(ncol(dataset), toyml$measures$num.inputs + 1)
  expect_equal(dataset[seq(toyml$measures$num.inputs)],
               toyml$dataset[toyml$attributesIndexes])
  expect_equal(dataset["y1"], toyml$dataset["y1"])

  dataset <- create_br_data(toyml, "y2")
  expect_equal(ncol(dataset), toyml$measures$num.inputs + 1)
  expect_equal(dataset[seq(toyml$measures$num.inputs)],
               toyml$dataset[toyml$attributesIndexes])
  expect_equal(dataset["y2"], toyml$dataset["y2"])

  one.column <- rep(1, toyml$measures$num.instances)
  dataset <- create_br_data(toyml, "y3", one.column)
  expect_equal(ncol(dataset), toyml$measures$num.inputs + 2)
  expect_equal(dataset[, length(dataset)-1], one.column)
  expect_equal(dataset["y3"], toyml$dataset["y3"])

  extra.columns <- cbind(a=one.column, b=rnorm(toyml$measures$num.instances))
  dataset <- create_br_data(toyml, "y4", extra.columns)
  expect_equal(ncol(dataset), toyml$measures$num.inputs + 3)
  expect_equal(dataset[c("a", "b")], as.data.frame(extra.columns))
  expect_equal(dataset["y4"], toyml$dataset["y4"])
})
