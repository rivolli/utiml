#' @title Binary Relevance for multi-label Classification
#' @family Transformation methods
#' @description Create a Binary Relevance model for multilabel classification.
#'
#'   Binary Relevance is a simple and effective transformation method to predict
#'   multi-label data. This is based on the one-versus-all approach to build a
#'   specific model for each label.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset.
#' @param base.method A string with the name of the base method.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'})
#' @param ... Others arguments passed to the base method for all subproblems
#' @param save.datasets Logical value indicating whether the binary datasets must be
#'   saved in the model or not. (default: \code{FALSE})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{BRmodel} containing the set of fitted
#'   models, including: \describe{
#'   \item{labels}{A vector with the label names}
#'   \item{models}{A list of the generated models, named by the label names.}
#'   \item{datasets}{A list of \code{mldBR} named by the label names.
#'   Only when the \code{save.datasets = TRUE}.} }
#'
#' @references
#'  Boutell, M. R., Luo, J., Shen, X., & Brown, C. M. (2004). Learning
#'    multi-label scene classification. Pattern Recognition, 37(9), 1757–1771.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Binary Relevance
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- br(emotions)
#' pred <- predict(model, testdata)
#'
#' # Change the default base method and use 4 CORES
#' model <- br(emotions, "C4.5", CORES = 4)
#' pred <- predict(model, testdata)
#'
#' # Set a parameters for all subproblems
#' model <- br(emotions, "KNN", k=5)
#' pred <- predict(model, testdata)
br <- function (mdata,
                base.method = "SVM",
                ...,
                save.datasets = FALSE,
                CORES = 1
              ) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BR Model class
  brmodel <- list()
  brmodel$labels <- rownames(mdata$labels)

  #Transformation
  datasets <- lapply(mldr_transform(mdata), br.transformation, classname = "mldBR", base.method = base.method)
  names(datasets) <- brmodel$labels
  if (save.datasets)
    brmodel$datasets <- datasets

  #Create models
  brmodel$models <- utiml_lapply(datasets, br.create_model, CORES, ...)

  brmodel$call <- match.call()
  class(brmodel) <- "BRmodel"

  brmodel
}

#' @title Predict Method for Binary Relevance
#' @description This function predicts values based upon a model trained by
#'  \code{\link{br}}.
#'
#' @param object Object of class "\code{BRmodel}", created by \code{\link{br}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param CORES The number of cores to parallelize the prediction. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @seealso \code{\link[=br]{Binary Relevance (BR)}}
#'
#' @export
#'
#' @examples
#' library(utiml)
#'
#' # Emotion multi-label dataset using Binary Relevance
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- br(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, testdata, probability = FALSE, CORES = 6)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
predict.BRmodel <- function (object,
                             newdata,
                             ...,
                             probability = TRUE,
                             CORES = 1
                             ) {
  #Validations
  if(class(object) != 'BRmodel')
    stop('First argument must be an BRmodel object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #Create models
  predictions <- utiml_lapply(object$models, br.predict_model, CORES, newdata = utiml_newdata(newdata), ...)
  as.multilabelPrediction(predictions, probability)
}

print.BRmodel <- function (x, ...) {
  cat("Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$labels), "Models (labels):\n")
  print(x$labels)
}

print.mldBR <- function (x, ...) {
  cat("Binary Relevance Transformation Dataset\n\n")
  cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep="")
  cat("Dataset info:\n")
  cat(" ", ncol(x$data) - 1, "Predictive attributes\n")
  cat(" ", nrow(x$data), "Examples\n")
  cat("  ", round((sum(x$data[,ncol(x$data)] == 1) / nrow(x$data)) * 100, 1), "% of positive examples\n", sep="")
}
