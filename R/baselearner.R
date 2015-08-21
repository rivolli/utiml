getResultPrediction <- function (bipartition = NULL, probability = NULL) {
  emptyBipartition <- is.null(bipartition)
  emptyProbability <- is.null(probability)
  if (emptyBipartition && emptyProbability)
    stop("Bipartion and probability results are NULL")
  else if (emptyProbability)
    probability <- bipartition
  else {
    bipartition <- probability
    active <- bipartition >= 0.5
    bipartition[active] <- 1
    bipartition[!active] <- 0
  }
  res <- list(bipartition = bipartition, probability = probability)
  class(res) <- "mlresult"

  res
}

#SVM classifier
mltrain.SVM <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::svm(traindata, labeldata, probability = TRUE, ...)
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  model
}

mlpredict.SVM <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, probability = TRUE, ...)
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  pscore <- attr(result, "probabilities")[,"1"]
  attr(result, "probabilities") <- NULL

  getResultPrediction(result, pscore)
}

#J48 classifier
mltrain.J48 <- function (dataset, ...) {
  if (requireNamespace("RWeka", quietly = TRUE)) {
    classname <- colnames(dataset$data)[ncol(dataset$data)]
    formula <- as.formula(paste("`", classname, "` ~ .", sep=""))
    model <- RWeka::J48(formula, dataset$data)
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  model
}

mlpredict.J48 <- function (model, newdata, ...) {
  if (requireNamespace("RWeka", quietly = TRUE)) {
    result <- predict(model, newdata, "probability")
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  getResultPrediction(probability = result[,2])
}
