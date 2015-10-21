context("BR based classifiers")

df <- data.frame(matrix(rnorm(50), ncol = 5))
df$Label1 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 10, replace = TRUE))
train <- mldr_from_dataframe(df, labelIndices = c(6, 7, 8), name = "testMLDR")
test <- train$dataset[, train$attributesIndexes]

baseTest <- function (model, expected.class) {
  expect_is(model, expected.class)
  expect_equal(names(model$models), rownames(train$labels))

  pred <- predict(model, test)
  expect_is(pred, "mlresult")
  expect_equal(nrow(pred), nrow(test))
  expect_equal(ncol(pred), train$measures$num.labels)
  expect_equal(colnames(pred), rownames(train$labels))
  expect_equal(rownames(pred), rownames(test))

  pred1 <- predict(model, test, prob = FALSE)
  expect_is(pred1, "mlresult")
  expect_equal(as.matrix(pred1), attr(pred, "classes"))
  expect_equal(as.matrix(pred), attr(pred1, "probs"))

  pred
}

test_that("Binary Relevance", {
  model <- br(train, "test")
  baseTest(model, "BRmodel")
})

test_that("BR Plus", {
  model <- brplus(train, "test")
  pred1 <- baseTest(model, "BRPmodel")

  expect_is(model$initial, "BRmodel")
  pred0 <- predict(model$initial, test)
  expect_false(isTRUE(all.equal(pred0, pred1)))

  pred2 <- predict(model, test, strategy="NU")
  expect_equal(colnames(pred2), rownames(train$labels))
  expect_equal(pred1, pred2)

  pred3 <- predict(model, test, "Stat")
  expect_equal(colnames(pred3), rownames(train$labels))
  expect_equal(pred1, pred3)

  new.chain <- c("Label3", "Label2", "Label1")
  pred4 <- predict(model, test, "Ord", new.chain)
  expect_equal(pred1, pred4)

  expect_error(predict(model, test, "xay"))
  expect_error(predict(model, test, "Ord"))
  expect_error(predict(model, test, "Ord", new.chain[1:2]))
  expect_error(predict(model, test, "Ord", c(new.chain, "extra")))
  expect_error(predict(model, test, "Ord", c("a", "b", "c")))
})

test_that("CTRL", {
  model <- ctrl(train, "test")
  pred1 <- baseTest(model, "CTRLmodel")
  baseTest(ctrl(train, "test", validation.threshold = 1), "CTRLmodel")

  model2 <- ctrl(train, "test", m = 2, validation.size = 0.2, validation.threshold = 0)
  pred2 <- baseTest(model2, "CTRLmodel")
  expect_equal(model2$rounds, 2)

  expect_error(ctrl(train, "test", 0))
  expect_error(ctrl(train, "test", validation.size=0))
  expect_error(ctrl(train, "test", validation.size=1))
  expect_error(ctrl(train, "test", validation.threshold=1.1))
  expect_error(predict(model, test, "ABC"))
})

test_that("DBR", {
  model <- dbr(train, "test")
  pred <- baseTest(model, "DBRmodel")
  expect_is(model$estimation, "BRmodel")

  estimative <- predict(model$estimation, test, prob = FALSE)
  pred1 <- predict(model, test, estimative)
  expect_equal(pred1, pred)

  model <- dbr(train, "test", estimate = FALSE)
  expect_error(predict(model, test))
})
