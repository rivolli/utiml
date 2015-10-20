mltrain.basetest <- function (dataset, ...) {
  model <- list(predictions = round(utiml_normalize(rowMeans(dataset$data[,-dataset$labelindex])), 3))
  class(model) <- "test"
  model
}

mlpredict.test <- function (model, newdata, ...) {
  matrix(
    c(1 - model$predictions, model$predictions),
    ncol=2,
    dimnames=list(rownames(newdata), c("0","1"))
  )
}
