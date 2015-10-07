#' @title Create a predictive binary result object
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
#' mlpredict.random <- function (model, newdata, ...) {
#'    probs <- runif(nrow(newdata), 0, 1)
#'    as.binaryPrediction(probs)
#' }
#'
#' # Use different threshold value
#' probs <- runif(10, 0, 1)
#' result <- as.binaryPrediction(probs, 0.6)
as.binaryPrediction <- function (probability, threshold = 0.5) {
  bipartition <- probability
  active <- bipartition >= threshold
  bipartition[active] <- 1
  bipartition[!active] <- 0

  mlresult(bipartition, probability)
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
#' is available on \code{dataset$base}. In the examples there are some ways to
#' implement this method.
#'
#' @param dataset An object of \code{mltransformation} class, that has at least
#'  four attributes: \strong{data}, \strong{labelname}, \strong{labelindex} and
#'  \strong{methodname}. The \code{data} is the dataframe with the predictive
#'  attributes and the class column. The \code{labelname} is the name of the
#'  class column. The \code{labelindex} is the column number of the class.
#'   Finally, the \code{methodname} is the name of the implemented method.
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
#'    traindata <- dataset$data[, -dataset$labelindex]
#'    labeldata <- dataset$data[, dataset$labelindex]
#'    model <- svm(traindata, labeldata, probability = TRUE, ...)
#'    model
#' }
mltrain <- function (dataset, ...) UseMethod("mltrain")

#' @describeIn mltrain Default S3 method
mltrain.default <- function (dataset, ...) {
  funcname <- paste("mltrain.base", dataset$methodname, sep='')
  stop(paste("The function '", funcname, "(dataset, ...)' is not implemented", sep=''))
}


#' @title Prediction function to extend base classifiers
#'
#' @description
#'  To extend a base classifier, two steps are necessary:
#'  \enumerate{
#'    \item Create a train method
#'    \item Create a prediction method
#'  }
#'  This section is about how to create a prediction method. To create a new train
#'  method see \code{\link{mltrain}} documentation.
#'
#' @section How to create a new prediction base method:
#' Fist is necessary to know the class of model generate by respective train method
#' because this name determine the method name, that must start with mlpredict.
#' followed by the model class name, e.g. a model with class "foomodel" must be
#' \code{mlpredict.foomodel}.
#'
#' After defined the name, you need to implement your prediction base method. The
#' model is available on \code{model} parameter and the new data to predict
#' \code{newdata}. In the examples there are some ways to implement this method.
#'
#' The return of this method must be provided by the
#' \code{\link{as.binaryPrediction}} method.
#'
#' @param model An object model returned by some mltrain method, its class
#'  determine the name of this method.
#' @param newdata A dataframe with the new data to be predicted
#' @param ... Others arguments passed to the predict method.
#'
#' @return An object of the type "\code{mlresult}". Use the
#'  \code{\link{as.binaryPrediction}} to return the prediction result
#'
#' @export
#'
#' @examples
#' # Create a method that predict always the negative class (The model must be the class "negativemodel")
#' mlpredict.negativemodel <- function (model, newdata, ...) {
#'    preds <- rep(0, nrow(newdata))
#'    as.binaryPrediction(preds)
#' }
#'
#' # Create a SVM predict method using the e1071 package (the class of SVM model from e1071 package is "svm")
#' library(e1071)
#' mlpredict.svm <- function (dataset, ...) {
#'    result <- predict(model, newdata, probability = TRUE, ...)
#'    as.biinaryPrediction(attr(result, "probabilities")[,"1"])
#' }
mlpredict <- function (model, newdata, ...) UseMethod("mlpredict")

#' @describeIn mlpredict Default S3 method
mlpredict.default <- function (model, newdata, ...) {
  funcname <- paste("mlpredict.", class(model), sep='')
  stop(paste("The function '", funcname, "(dataset, newdata, ...)' is not implemented", sep=''))
}

#' @describeIn mltrain SVM implementation (require \pkg{e1071} package to use)
mltrain.baseSVM <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -dataset$labelindex]
    labeldata <- dataset$data[, dataset$labelindex]
    model <- e1071::svm(traindata, labeldata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  model
}

#' @describeIn mlpredict SVM implementation (require \pkg{e1071} package to use)
mlpredict.svm <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, probability = TRUE, ...)
  } else
    stop('There are no installed package "e1071" to use SVM classifier as base method')

  as.binaryPrediction(attr(result, "probabilities")[,"1"])
}

#' @describeIn mltrain J48 implementation (require \pkg{RWeka} package to use)
mltrain.baseJ48 <- function (dataset, ...) {
  if (requireNamespace("RWeka", quietly = TRUE)) {
    formula <- as.formula(paste("`", dataset$labelname, "` ~ .", sep=""))
    model <- RWeka::J48(formula, dataset$data, ...)
  } else
    stop('There are no installed package "RWeka" to use C4.5/J48 classifier as base method')

  rJava::.jcache(model$classifier)
  model
}

#' @describeIn mlpredict C4.5/J48 implementation (require \pkg{RWeka} package to use)
mlpredict.J48 <- function (model, newdata, ...) {
  if (requireNamespace("RWeka", quietly = TRUE))
    result <- predict(model, newdata, "probability", ...)
  else
    stop('There are no installed package "RWeka" to use C4.5/J48 classifier as base method')

  as.binaryPrediction(result[,"1"])
}

#' @describeIn mltrain C4.5 implementation (require \pkg{RWeka} package to use)
mltrain.baseC4.5 <- mltrain.baseJ48

#' @describeIn mltrain C5.0 implementation (require \pkg{C50} package to use)
mltrain.baseC5.0 <- function (dataset, ...) {
  if (requireNamespace("C50", quietly = TRUE)) {
    traindata <- dataset$data[, -dataset$labelindex]
    labeldata <- dataset$data[, dataset$labelindex]
    model <- C50::C5.0(traindata, labeldata, ...)
  } else
    stop('There are no installed package "C50" to use C5.0 classifier as base method')

  model
}

#' @describeIn mlpredict C5.0 implementation (require \pkg{C50} package to use)
mlpredict.C5.0 <- function (model, newdata, ...) {
  if (requireNamespace("C50", quietly = TRUE)) {
    result <- predict(model, newdata, type = "prob", ...)
  } else
    stop('There are no installed package "C50" to use C5.0 classifier as base method')

  as.binaryPrediction(result[,"1"])
}

#' @describeIn mltrain CART implementation (require \pkg{rpart} package to use)
mltrain.baseCART <- function (dataset, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    formula <- as.formula(paste("`", dataset$labelname, "` ~ .", sep=""))
    model <- rpart::rpart(formula, dataset$data, ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  model
}

#' @describeIn mlpredict CART implementation (require \pkg{rpart} package to use)
mlpredict.rpart <- function (model, newdata, ...) {
  if (requireNamespace("rpart", quietly = TRUE)) {
    result <- predict(model, newdata, type = "prob", ...)
  } else
    stop('There are no installed package "rpart" to use Cart classifier as base method')

  as.binaryPrediction(result[,"1"])
}

#' @describeIn mltrain Random Forest (RF) implementation (require \pkg{randomForest} package to use)
mltrain.baseRF <- function (dataset, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    traindata <- dataset$data[, -dataset$labelindex]
    labeldata <- dataset$data[, dataset$labelindex]
    model <- randomForest::randomForest(traindata, labeldata, ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  model
}

#' @describeIn mlpredict Random Forest (RF) implementation (require \pkg{randomForest} package to use)
mlpredict.randomForest <- function (model, newdata, ...) {
  if (requireNamespace("randomForest", quietly = TRUE)) {
    result <- predict(model, newdata, type="prob", ...)
  } else
    stop('There are no installed package "randomForest" to use randomFores classifier as base method')

  as.binaryPrediction(result[,"1"])
}

#' @describeIn mltrain Naive Bayes (NB) implementation (require \pkg{e1071} package to use)
mltrain.baseNB <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -dataset$labelindex]
    labeldata <- dataset$data[, dataset$labelindex]
    model <- e1071::naiveBayes(traindata, labeldata, type="raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')

  model
}

#' @describeIn mlpredict Naive Bayes (NB) implementation (require \pkg{e1071} package to use)
mlpredict.naiveBayes <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, type = "raw", ...)
  } else
    stop('There are no installed package "e1071" to use naiveBayes classifier as base method')
  rownames(result) <- rownames(newdata)
  as.binaryPrediction(result[,"1"])
}

#' @describeIn mltrain kNN implementation (require \pkg{class} package to use)
mltrain.baseKNN <- function (dataset, ...) {
  if (!requireNamespace("class", quietly = TRUE))
    stop('There are no installed package "class" to use kNN classifier as base method')
  dataset$extrakNN <- list(...)
  dataset
}

#' @describeIn mlpredict kNN implementation (require \pkg{class} package to use)
mlpredict.baseKNN <- function (model, newdata, ...) {
  if (requireNamespace("class", quietly = TRUE)) {
    traindata <- model$data[, -model$labelindex]
    labeldata <- model$data[, model$labelindex]
    args <- list(...)
    result <- if (is.null(model$extrakNN[["k"]]) || !is.null(args[["k"]]))
        class::knn(traindata, newdata, labeldata, prob= T, ...)
      else
        class::knn(traindata, newdata, labeldata, k = model$extrakNN[["k"]], prob = T, ...)
    result <- ifelse(result == 0, 1-attr(result, "prob"), attr(result, "prob"))
  } else
    stop('There are no installed package "class" to use kNN classifier as base method')
  names(result) <- rownames(newdata)
  as.binaryPrediction(result)
}

summary.mltransformation <- function (x, ...) {
  summary(x$data)
}
