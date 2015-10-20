context("BR based classifiers")

df <- data.frame(matrix(rnorm(50), ncol = 5))
df$Label1 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 10, replace = TRUE))
train <- mldr_from_dataframe(df, labelIndices = c(6, 7, 8), name = "testMLDR")
test <- train$dataset[, train$attributesIndexes]

test_that("Binary Relevance", {
  model <- br(train, "test")
  expect_is(model, "BRmodel")
  expect_equal(length(model$models), train$measures$num.labels)
  expect_equal(model$labels, rownames(train$labels))

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
  expect_equal(pred[,1], model$models[[1]]$predictions, check.names=FALSE)
})

test_that("BR Plus", {
  model <- brplus(train, "test")
  expect_is(model, "BRPmodel")
  expect_is(model$initial, "BRmodel")
  expect_equal(names(model$models), rownames(train$labels))

  pred0 <- predict(model$initial, test)

  pred1 <- predict(model, test)
  expect_is(pred1, "mlresult")
  expect_equal(nrow(pred1), nrow(test))
  expect_equal(ncol(pred1), train$measures$num.labels)
  expect_equal(colnames(pred1), rownames(train$labels))
  expect_equal(rownames(pred1), rownames(test))
  expect_false(isTRUE(all.equal(pred0, pred1)))

  pred2 <- predict(model, test, prob = FALSE)
  expect_is(pred2, "mlresult")
  expect_equal(as.matrix(pred2), attr(pred1, "classes"))
  expect_equal(as.matrix(pred1), attr(pred2, "probs"))
  expect_equal(pred1[,1], model$models[[1]]$predictions, check.names=FALSE)

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


