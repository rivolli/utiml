#' @title Create a predictive result object
#'
#' @description The transformation methods require a specific data format from base
#'  classifiers prediction. If you implement a new base method then use this method
#'  to return the final result of your \code{mlpredict} method.
#'
#' @param probability A vector with probabilities predictions or with bipartitions
#'  prediction for a binary prediction.
#' @param threshold A numeric value between 0 and 1 to create the bipartitions.
#'
#' @return An object of type "\code{mlresult}" used by problem transformation
#'  methods that use binary classifiers. It has only two attributes:
#'  \code{bipartition} and \code{probability}, that respectively have the
#'  bipartition and probabilities results.
#' @export
#'
#' @examples
#' # This method is used to implement a mlpredict based method
#' # In this example we create a random predict method
#' mlpredicti.random <- function (model, newdata, ...) {
#'    probs <- sample(seq(0,1,.025), nrow(newdata), replace = TRUE)
#'    as.resultPrediction(probs)
#' }
#'
#' # Define a different threshold for a specific subproblem use
#' ...
#' result <- as.resultPrediction(probs, 0.6)
#' ...
as.resultPrediction <- function (probability, threshold = 0.5) {
  bipartition <- probability
  active <- bipartition >= threshold
  bipartition[active] <- 1
  bipartition[!active] <- 0

  res <- list(bipartition = bipartition, probability = probability)
  class(res) <- "mlresult"
  res
}

#' @title Train function to extend base classifiers
#'
#' @description
#'  To extend a base classifier, two steps are necessary:
#'  \enumerate{
#'    \item Create a train method
#'    \item Create a prediction method
#'  }
#'  This section is about how to create a train method. To create a new predict
#'  model see \code{\link{mlpredict}} documentation.
#'
#' @section How to create a new train base method:
#' Fist is necessary to define a name of your classifier because this name
#' determine the method name, that must start with mltrain.base followed by your
#' name, e.g. a "foo" classify must be \code{mltrain.basefoo}.
#'
#' After defined the name, you need to implement your base method. The dataset
#' is available on \code{dataset$base}
#'
#'
#'
#' @param dataset An object of \code{mltransformation} class, that has at least
#'  three attributes: \strong{data}, \strong{labelname} and \strong{methodname}.
#'  The \code{data} is the dataframe with the predictive attributes and the
#'  class column. The \code{labelname} is the name of the class column. Finally,
#'  the \code{methodname} is the name of the implemented method.
#' @param ... Others arguments passed to the base method.
#'
#' @return A model object. The class of this model can be of any type, however,
#'  this object will be passed to the respective mlpredict method.
#'
#' @export
#'
#' @examples
#' # Create a empty model of type foo
#' mltrain.basefoo <- function (dataset, ...) {
#'    mymodel <- list()
#'    class(mymodel) <- "foomodel"
#'    mymodel
#' }
#'
#' #Use this base method with Binary Relevance
#' brmodel <- br(emotions, "foo")
#'
#' # Create a SVM method using the e1071 package
#' library(e1071)
#' mltrain.baseSVM <- function (dataset, ...) {
#'    traindata <- dataset$data[, -ncol(dataset$data)]
#'    labeldata <- dataset$data[, dataset$labelname]
#'    model <- svm(traindata, labeldata, probability = TRUE, ...)
#'    model
#' }
mltrain <- function (dataset, ...) UseMethod("mltrain")

mltrain.default <- function (dataset, ...) {
  funcname <- paste("mltrain.base", dataset$methodname, sep='')
  stop(paste("The function '", funcname, "(dataset, ...)' is not implemented", sep=''))
}

mlpredict <- function (model, newdata, ...) UseMethod("mlpredict")

mlpredict.default <- function (model, newdata, ...) {
  funcname <- paste("mlpredict.", class(model), sep='')
  stop(paste("The function '", funcname, "(dataset, newdata, ...)' is not implemented", sep=''))
}

#Suport Vector Machines
mltrain.baseSVM <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::svm(traindata, labeldata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  model
}

mlpredict.svm <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  as.resultPrediction(attr(result, "probabilities")[,"1"])
}

#Decision Tree - J48
mltrain.baseJ48 <- function (dataset, ...) {
  classname <- colnames(dataset$data)[ncol(dataset$data)]
  formula <- as.formula(paste("`", classname, "` ~ .", sep=""))
  if (requireNamespace("RWeka", quietly = TRUE))
    model <- RWeka::J48(formula, dataset$data, ...)
  else
    stop('There are no installed package "RWeka" to use C4.5/J48 classifier as base method')

  model
}

mlpredict.J48 <- function (model, newdata, ...) {
  if (requireNamespace("RWeka", quietly = TRUE))
    result <- predict(model, newdata, "probability", ...)
  else
    stop('There are no installed package "RWeka" to use C4.5/J48 classifier as base method')

  as.resultPrediction(result[,"1"])
}

#Decision Tree - C4.5
mltrain.baseC4.5 <- mltrain.baseJ48

#Decision Tress - C5.0
mltrain.baseC5.0 <- function (dataset, ...) {
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

  as.resultPrediction(result[,"1"])
}

#CART
mltrain.baseCART <- function (dataset, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    classname <- colnames(dataset$data)[ncol(dataset$data)]
    formula <- as.formula(paste("`", classname, "` ~ .", sep=""))
    model <- rpart::rpart(formula, dataset$data, ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  model
}

mlpredict.rpart <- function (model, newdata, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    result <- predict(model, newdata, type = "prob", ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  as.resultPrediction(result[,"1"])
}

#Random Forest
mltrain.baseRF <- function (dataset, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- randomForest::randomForest(traindata, labeldata, ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  model
}

mlpredict.randomForest <- function (model, newdata, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    result <- predict(model, newdata, type="prob", ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  as.resultPrediction(result[,"1"])
}

#Naive Bayes
mltrain.baseNB <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::naiveBayes(traindata, labeldata, type="raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')

  model
}

mlpredict.naiveBayes <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, type = "raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')
  rownames(result) <- rownames(newdata)
  as.resultPrediction(result[,"1"])
}

#Knn - consider others packages (FNN and KKNN)
mltrain.baseKNN <- function (dataset, ...) {
  if (!requireNamespace("class", quietly = TRUE))
    stop('There are no installed package "class" to use kNN classifier as base method')
  dataset$extrakNN <- list(...)
  dataset
}

mlpredict.baseKNN <- function (model, newdata, ...) {
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
  as.resultPrediction(result)
}
