context("CC based classifiers")

df <- data.frame(matrix(rnorm(50), ncol = 5))
df$Label1 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 10, replace = TRUE))
train <- mldr_from_dataframe(df, labelIndices = c(6, 7, 8), name = "testMLDR")
test <- train$dataset[, train$attributesIndexes]

test_that("Classifier Chain", {
  model <- cc(train, "test")
  expect_is(model, "CCmodel")
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

  new.chain <- c("Label3", "Label2", "Label1")
  model2 <- cc(train, "test", new.chain)
  expect_equal(model2$chain, new.chain)

  pred2 <- predict(model2, test)
  expect_equal(colnames(pred2), rownames(train$labels))

  pred3 <- predict(model2, test)
  expect_false(isTRUE(all.equal(pred3, pred1)))
  expect_equal(pred3, pred2)

  expect_error(cc(train, "test", chain=c("a", "b", "c")))
  expect_error(cc(train, "test", chain=c(new.chain, "extra")))
})
