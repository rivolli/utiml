context("Ensemble tests")

test_that("Majority votes", {
  predictions <- list(
    as.binaryPrediction(c(1  ,1  ,1  ,1  )),
    as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
    as.binaryPrediction(c(0.8,0.3,0.4,0.1))
  )
  result <- utiml_ensemble_majority_votes(predictions)
  expect_is(result, "mlresult")
  expect_equal(result$bipartition, c(1,0,1,0))
  expect_equal(result$probability, c(0.8,0.2,0.9,0.15))

  predictions[[4]] <- as.binaryPrediction(c(0.6,0.5,0.6,1))
  for (i in 1:4) names(predictions[[i]]$bipartition) <- c(1,3,5,6)
  result <- utiml_ensemble_majority_votes(predictions)
  expect_named(result$bipartition, as.character(c(1,3,5,6)))
  expect_equivalent(result$bipartition, c(1,0,1,1))
  expect_equivalent(result$probability, c(0.75,0.2,0.8,1))
})
