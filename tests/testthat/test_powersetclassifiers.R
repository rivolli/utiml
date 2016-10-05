context("Powerset based classifiers")
train <- toyml
test <- toyml$dataset[10:40, toyml$attributesIndexes]

predictionTest <- function (model) {
  set.seed(123)
  pred <- predict(model, test)
  expect_is(pred, "mlresult")
  expect_equal(nrow(pred), nrow(test))
  expect_equal(ncol(pred), toyml$measures$num.labels)
  expect_equal(colnames(pred), rownames(toyml$labels))
  expect_equal(rownames(pred), rownames(test))

  set.seed(123)
  pred1 <- predict(model, test, prob = FALSE)
  expect_is(pred1, "mlresult")
  expect_equal(as.matrix(pred1), attr(pred, "classes"))
  expect_equal(as.matrix(pred), attr(pred1, "probs"))

  pred
}

baseTest <- function (model, expected.class) {
  expect_is(model, expected.class)
  predictionTest(model)
}

test_that("LP", {
  model <- lp(train, "RANDOM")
  baseTest(model, "LPmodel")
})

test_that("RAkEL", {
  model <- rakel(train, "RANDOM")
  baseTest(model, "RAkELmodel")
})
