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

#Suport Vector Machines
mltrain.SVM <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::svm(traindata, labeldata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  model
}

mlpredict.SVM <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  pscore <- attr(result, "probabilities")[,"1"]
  attr(result, "probabilities") <- NULL

  getResultPrediction(result, pscore)
}

#Decision Tree - J48
mltrain.J48 <- function (dataset, ...) {
  if (requireNamespace("RWeka", quietly = TRUE)) {
    classname <- colnames(dataset$data)[ncol(dataset$data)]
    formula <- as.formula(paste("`", classname, "` ~ .", sep=""))
    model <- RWeka::J48(formula, dataset$data, ...)
  } else
    stop('There are no installed package "RWeka" to use J48 classifier as base method')

  model
}

mlpredict.J48 <- function (model, newdata, ...) {
  if (requireNamespace("RWeka", quietly = TRUE)) {
    result <- predict(model, newdata, "probability", ...)
  } else
    stop('There are no installed package "RWeka" to use J48 classifier as base method')

  getResultPrediction(probability = result[,"1"])
}

#Decision Tress - C5.0
mltrain.C5.0 <- function (dataset, ...) {
  if (requireNamespace("C50", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- C50::C5.0(traindata, labeldata, ...)
  } else
    stop('There are no installed package "C50" to use C5.0 classifier as base method')

  model
}

mlpredict.C5.0 <- function (model, newdata, ...) {
  if (requireNamespace("C50", quietly = TRUE)) {
    result <- predict(model, newdata, type = "prob", ...)
  } else
    stop('There are no installed package "C50" to use C5.0 classifier as base method')

  getResultPrediction(probability = result[,"1"])
}

#CART
mltrain.CART <- function (dataset, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    classname <- colnames(dataset$data)[ncol(dataset$data)]
    formula <- as.formula(paste("`", classname, "` ~ .", sep=""))
    model <- rpart::rpart(formula, dataset$data, ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  model
}

mlpredict.CART <- function (model, newdata, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    result <- predict(model, newdata, type = "prob", ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  getResultPrediction(probability = result[,"1"])
}

#Random Forest
mltrain.RF <- function (dataset, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- randomForest::randomForest(traindata, labeldata, ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  model
}

mlpredict.RF <- function (model, newdata, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    result <- predict(model, newdata, type="prob", ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  getResultPrediction(probability = result[,"1"])
}

#Naive Bayes
mltrain.NB <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::naiveBayes(traindata, labeldata, type="raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')

  model
}

mlpredict.NB <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, type = "raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')
  rownames(result) <- rownames(newdata)
  getResultPrediction(probability = result[,"1"])
}

#Knn - consider others packages (FNN and KKNN)
mltrain.KNN <- function (dataset, ...) {
  if (!requireNamespace("class", quietly = TRUE))
    stop('There are no installed package "class" to use kNN classifier as base method')
  dataset$extrakNN <- list(...)
  dataset
}

mlpredict.KNN <- function (model, newdata, ...) {
  if (requireNamespace("class", quietly = TRUE)) {
    traindata <- model$data[, -ncol(model$data)]
    labeldata <- model$data[, model$labelname]
    args <- list(...)
    result <- if (is.null(model$extrakNN[["k"]]) || !is.null(args[["k"]]))
        class::knn(traindata, newdata, labeldata, prob=T, ...)
      else
        class::knn(traindata, newdata, labeldata, k=model$extrakNN[["k"]], prob=T, ...)
    result <- ifelse(result == 0, 1-attr(result, "prob"), attr(result, "prob"))
  } else
    stop('There are no installed package "class" to use kNN classifier as base method')
  names(result) <- rownames(newdata)
  getResultPrediction(probability = result)
}
