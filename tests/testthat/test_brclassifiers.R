context("BR classifiers")

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
  pred <- predict(model, test, prob = FALSE)
  expect_is(pred, "mlresult")
})

