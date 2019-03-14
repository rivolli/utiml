context("Powerset based classifiers")
train <- toyml
test <- toyml$dataset[10:40, toyml$attributesIndexes]

predictionTest <- function (model) {
  suppressWarnings(RNGversion("3.5.0"))
  set.seed(123)
  pred <- predict(model, test)
  expect_is(pred, "mlresult")
  expect_equal(nrow(pred), nrow(test))
  expect_equal(ncol(pred), toyml$measures$num.labels)
  expect_equal(colnames(pred), rownames(toyml$labels))
  expect_equal(rownames(pred), rownames(test))

  suppressWarnings(RNGversion("3.5.0"))
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

test_that("PPT", {
  model <- ppt(train, "RANDOM")
  baseTest(model, "PPTmodel")

  model <- ppt(train, "RANDOM", info.loss=TRUE)
  baseTest(model, "PPTmodel")

  expect_error(ppt(train, "RANDOM", p=0))
})

test_that("PS", {
  model <- ps(train, "RANDOM")
  baseTest(model, "PSmodel")

  model <- ps(train, "RANDOM", strategy="B")
  baseTest(model, "PSmodel")

  expect_error(ps(train, "RANDOM", p=0))
})

test_that("EPS", {
  model <- eps(train, "RANDOM")
  baseTest(model, "EPSmodel")

  expect_error(eps(train, "RANDOM", m=0))
})
