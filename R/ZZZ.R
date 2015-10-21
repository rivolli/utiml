mltrain.basetest <- function (dataset, ...) {
  model <- list(predictions = round(utiml_normalize(rowMeans(dataset$data[,-dataset$labelindex])), 3))
  class(model) <- "test"
  model
}

mlpredict.test <- function (model, newdata, ...) {
  predictions <- model$predictions
  if (nrow(newdata) < length(predictions))
    predictions <- predictions[1:nrow(newdata)]
  else if (nrow(newdata) > length(predictions))
    predictions <- rep(predictions, nrow(newdata))[1:nrow(newdata)]

  matrix(
    c(1 - predictions, predictions),
    ncol=2,
    dimnames=list(rownames(newdata), c("0","1"))
  )
}
