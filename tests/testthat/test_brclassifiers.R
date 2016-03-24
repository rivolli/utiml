context("BR based classifiers")

df <- data.frame(matrix(rnorm(50), ncol = 5))
df$Label1 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label2 <- c(sample(c(0,1), 10, replace = TRUE))
df$Label3 <- c(sample(c(0,1), 10, replace = TRUE))
train <- mldr::mldr_from_dataframe(df, labelIndices = c(6, 7, 8), name = "testMLDR")
test <- train$dataset[, train$attributesIndexes]

predictionTest <- function (model) {
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

baseTest <- function (model, expected.class) {
  expect_is(model, expected.class)
  expect_equal(names(model$models), rownames(train$labels))

  predictionTest(model)
}

ensembleTest <- function (model, expected.class) {
  expect_is(model, expected.class)
  expect_equal(length(model$models), model$rounds)

  predictionTest(model)
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

test_that("Classifier Chain", {
  model <- cc(train, "test")
  pred <- baseTest(model, "CCmodel")
  mpred <- as.matrix(pred)

  pred1 <- predict(model, test, prob = FALSE)
  expect_is(pred1, "mlresult")
  expect_equal(as.matrix(pred1), attr(pred, "classes"))
  expect_equal(as.matrix(pred), attr(pred1, "probs"))
  expect_equal(mpred[,1], model$models[[1]]$predictions, check.names=FALSE)

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
  expect_error(predict(model, test, NULL))
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

test_that("EBR", {
  model1 <- ebr(train, "test")
  pred1 <- ensembleTest(model1, "EBRmodel")

  model2 <- ebr(train, "test", m=3, subsample=0.5, attr.space=0.40)
  expect_equal(model2$nrow, 5)
  expect_equal(model2$ncol, 2)
  pred2<- ensembleTest(model2, "EBRmodel")

  expect_false(isTRUE(all.equal(pred1, pred2)))

  predictions <- predict(model1, test, vote.schema = NULL)
  expect_equal(length(predictions), 10)
  predictions <- predict(model2, test, vote.schema = NULL)
  expect_equal(length(predictions), 3)

  expect_error(ebr(train, "test", subsample=0))
  expect_error(ebr(train, "test", attr.space=0))
  expect_error(ebr(train, "test", m=0))
  expect_error(predict(model1, test, "ABC"))
})

test_that("ECC", {
  model1 <- ecc(train, "test")
  pred1 <- ensembleTest(model1, "ECCmodel")

  model2 <- ecc(train, "test", m=3, subsample=0.5, attr.space=0.40)
  expect_equal(model2$nrow, 5)
  expect_equal(model2$ncol, 2)
  pred2<- ensembleTest(model2, "ECCmodel")

  expect_false(isTRUE(all.equal(pred1, pred2)))

  predictions <- predict(model1, test, vote.schema = NULL)
  expect_equal(length(predictions), 10)
  predictions <- predict(model2, test, vote.schema = NULL)
  expect_equal(length(predictions), 3)

  expect_error(ecc(train, "test", subsample=0))
  expect_error(ecc(train, "test", attr.space=0))
  expect_error(ecc(train, "test", m=0))
  expect_error(predict(model1, test, "ABC"))
})

test_that("MBR", {
  model <- mbr(train, "test")
  pred <- baseTest(model, "MBRmodel")
  expect_is(model$basemodel, "BRmodel")

  model <- mbr(train, "test", folds=2, phi=0.3)
  pred <- baseTest(model, "MBRmodel")

  expect_error(mbr(train, "test", folds=0))
  expect_error(mbr(train, "test", phi=1.1))
})
