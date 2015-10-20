mltrain.basetest <- function (dataset, ...) {
  model <- list(predictions <- runif(10, 0, 1))
  class(model) <- "test"
}

mltrain.test <- function (model, newdata, ...) {
  matrix(
    1 - model$predictions,
    model$predictions,
    ncol=2,
    dimnames=list(rownames(newdata), c("0","1"))
  )
}
