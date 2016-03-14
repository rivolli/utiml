context("Ensemble tests")

test_that("Majority votes", {
  predictions <- list(
    as.binaryPrediction(c(1  ,1  ,1  ,1  )),
    as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
    as.binaryPrediction(c(0.8,0.3,0.4,0.1))
  )
  result <- majority_ensemble_votes(predictions)
  expect_is(result, "binary.prediction")
  expect_equal(result$bipartition, c(1,0,1,0))
  expect_equal(result$probability, c(0.8,0.2,0.9,0.15))

  result <- majority_ensemble_votes(predictions[2])
  expect_equal(result, predictions[[2]])

  expect_error(majority_ensemble_votes(list()), "Predictions can not be empty")

  predictions[[4]] <- as.binaryPrediction(c(0.6,0.5,0.6,1))
  for (i in 1:4) names(predictions[[i]]$bipartition) <- c(1,3,5,6)
  result <- majority_ensemble_votes(predictions)
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(1,0,1,1))
  expect_equivalent(result$probability, c(0.75,0.2,0.8,1))
})

test_that("Other votes", {
  predictions <- list(
    as.binaryPrediction(c(1  ,1  ,1  ,1  )),
    as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
    as.binaryPrediction(c(0.8,0.3,0.4,0.1))
  )
  for (i in 1:3) names(predictions[[i]]$bipartition) <- c(1,3,5,6)
  for (i in 1:3) names(predictions[[i]]$probability) <- c(1,3,5,6)

  result <- maximum_ensemble_votes(predictions)
  expect_is(result, "binary.prediction")
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(1,1,1,1))
  expect_equivalent(result$probability, c(1,1,1,1))
  result <- maximum_ensemble_votes(predictions[3])
  expect_equal(result, predictions[[3]])
  expect_error(maximum_ensemble_votes(list()), "Predictions can not be empty")

  result <- minimum_ensemble_votes(predictions)
  expect_is(result, "binary.prediction")
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(1,0,0,0))
  expect_equivalent(result$probability, c(0.6,0.1,0.4,0.1))
  result <- minimum_ensemble_votes(predictions[3])
  expect_equal(result, predictions[[3]])
  expect_error(minimum_ensemble_votes(list()), "Predictions can not be empty")

  result <- average_ensemble_votes(predictions)
  expect_is(result, "binary.prediction")
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(1,0,1,0))
  expect_equivalent(result$probability, c(0.8,1.4/3,2.2/3,1.3/3))
  result <- average_ensemble_votes(predictions[3])
  expect_equal(result, predictions[[3]])
  expect_error(average_ensemble_votes(list()), "Predictions can not be empty")

  result <- product_ensemble_votes(predictions)
  expect_is(result, "binary.prediction")
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(0,0,0,0))
  expect_equivalent(result$probability, c(0.48,0.03,0.32,0.02))
  result <- product_ensemble_votes(predictions[3])
  expect_equal(result, predictions[[3]])
  expect_error(product_ensemble_votes(list()), "Predictions can not be empty")
})

test_that("Multilabel ensemble", {
  pred1 <- as.multilabelPrediction(list(
    class1 = as.binaryPrediction(c(1  ,1  ,1  ,1  )),
    class2 = as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
    class3 = as.binaryPrediction(c(0.8,0.3,0.4,0.1))
  ), TRUE)
  pred2 <- as.multilabelPrediction(list(
    class1 = as.binaryPrediction(c(1  ,1  ,1  ,1  )),
    class2 = as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
    class3 = as.binaryPrediction(c(0.8,0.3,0.4,0.1))
  ), TRUE)
  pred3 <- as.multilabelPrediction(list(
    class1 = as.binaryPrediction(c(0.5,0.5,0.5,0.5)),
    class2 = as.binaryPrediction(c(0.6,0.6,0.6,0.6)),
    class3 = as.binaryPrediction(c(0.7,0.7,0.7,0.7))
  ), TRUE)

  result1 <- compute_multilabel_ensemble_votes(list(pred1, pred2, pred3), "maj")
  result2 <- compute_multilabel_ensemble_votes(list(pred1, pred2, pred3),
                                               "maj", FALSE)
  expect_true(all(result1 == attr(result2, "probs")))
  expect_true(all(result2 == attr(result1, "classes")))
  result2 <- as.matrix(result2)
  expect_equal(result2[,1], c(1,1,1,1))
  expect_equal(result2[,2], c(1,0,1,0))
  expect_equal(result2[,3], c(1,0,0,0))

  rownames(pred1) <- rownames(pred2) <- rownames(pred3) <- c(11:14)
  result <- compute_multilabel_ensemble_votes(list(pred1, pred2, pred3), "max")
  result <- as.matrix(result)
  expect_equal(dimnames(result), dimnames(pred1))
  expected <- matrix(c(1, 0.6, 0.8, 1, 0.6, 0.7, 1, 0.8, 0.7, 1, 0.6, 0.7),
                     ncol = 3, byrow = T)
  dimnames(expected) <- dimnames(pred1)
  expect_equal(result[,1:3], expected)
})
