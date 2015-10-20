mltrain.basetest <- function (dataset, ...) {
  model <- list(predictions <- utiml_normalize(rowMeans(dataset$data)) < 0.05)
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
