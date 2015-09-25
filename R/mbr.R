#' @title Meta-BR or 2BR for multi-label Classification
#' @family Transformation methods
#' @description Create a Meta-BR (MBR) classifier to predic multi-label
#'  data. To this, two round of Binary Relevance is executed, such that,
#'  the first iteraction generates new attributes to enrich the second
#'  prediction.
#'
#'  This implementation use complete training set for both training and
#'  prediction steps of 2BR. However, the \code{phi} parameter may be used
#'  to remove low label correlations on the second step. Furthermore, we
#'  remove the \code{F} parameter because its specification is not clear.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string with the name of base method. The same base method
#'   will be used for train all subproblems in the two rounds. This is slightly
#'   different from the original paper that uses two distinct values from this
#'   parameter.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'})
#' @param phi A value between 0 and 1 to determine the correlation coefficient,
#'    The value 0 include all labels in the second phase and the 1 only the
#'    predicted label. A good value for this argument is 0.3 as suggest in
#'    original paper. (default: 0)
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param predict.params A list of default arguments passed to the predictor a mldr object
#'  method. (default: \code{list()})
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: \code{FALSE})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{MBRmodel} containing the set of fitted
#'   models, including: \describe{
#'      \item{labels}{A vector with the label names.}
#'      \item{phi}{The value of \code{phi} parameter.}
#'      \item{correlation}{The matrix of label correlations used in combination
#'        with \code{phi} parameter to define the labels used in the second step.
#'      }
#'      \item{basemodel}{The BRModel used in the first iteration.}
#'      \item{metamodels}{A list of models named by the label names used in the
#'        second iteration.
#'      }
#'      \item{datasets}{A list with \code{base} and \code{meta} datasets of
#'        type \code{mldBR} named by the label names. Only when the
#'        \code{save.datasets = TRUE}.
#'      }
#'  }
#'
#' @references
#'  Tsoumakas, G., Dimou, A., Spyromitros, E., Mezaris, V., Kompatsiaris, I., &
#'    Vlahavas, I. (2009). Correlation-based pruning of stacked binary relevance models
#'    for multi-label learning. In Proceedings of the Workshop on Learning from
#'    Multi-Label Data (MLD’09) (pp. 22–30).
#'  Godbole, S., & Sarawagi, S. (2004). Discriminative Methods for Multi-labeled
#'    Classification. In Data Mining and Knowledge Discovery (pp. 1–26).
#' @seealso \code{\link{labels_correlation_coefficient}}
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Meta-BR
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- mbr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use different phi correlation with C4.5 classifier
#' model <- mbr(emotions, "C4.5", 0.3)
#' pred <- predict(model, testdata)
#'
#' # Set a specific parameter
#' model <- mbr(emotions, "KNN", k=5)
#' pred <- predict(model, testdata)
mbr <- function (mdata,
                  base.method = "SVM",
                  phi = 0,
                  ...,
                  predict.params = list(),
                  save.datasets = FALSE,
                  CORES = 1
) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #MBR Model class
  mbrmodel <- list()
  mbrmodel$labels <- rownames(mdata$labels)
  mbrmodel$phi <- phi

  #1 Iteration - Base Level
  mbrmodel$basemodel <- br(mdata, base.method, ..., save.datasets = TRUE, CORES = CORES)
  params <- list(object = mbrmodel$basemodel,
                 newdata = mdata$dataset[mdata$attributesIndexes],
                 probability = FALSE, CORES = CORES)
  base.preds <- do.call(predict, c(params, predict.params))

  #2 Iteration - Meta level
  corr <- mbrmodel$correlation <- labels_correlation_coefficient(mdata)
  datasets <- utiml_lapply(mbrmodel$basemodel$datasets, function (dataset) {
    extracolumns <- base.preds[,colnames(corr)[corr[dataset$labelname,] > phi]]
    colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
    base <- cbind(dataset$data[-dataset$labelindex], extracolumns, dataset$data[dataset$labelindex])
    br.transformation(base, "mldMBR", base.method, new.features = colnames(extracolumns))
  }, CORES)
  mbrmodel$metamodels <- utiml_lapply(datasets, br.create_model, CORES, ...)

  if (save.datasets)
    mbrmodel$datasets <- list(base = mbrmodel$basemodel$datasets, meta = datasets)

  mbrmodel$basemodel$datasets <- NULL

  mbrmodel$call <- match.call()
  class(mbrmodel) <- "MBRmodel"

  mbrmodel
}

#' @title Predict Method for Meta-BR/2BR
#' @description This function predicts values based upon a model trained by \code{mbr}.
#'
#' @param object Object of class "\code{MBRmodel}", created by \code{\link{mbr}} method.
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
#' @seealso \code{\link[=mbr]{Meta-BR (MBR or 2BR)}}
#' @export
#'
#' @examples
#' #' library(utiml)
#'
#' # Emotion multi-label dataset using Meta-BR or 2BR
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- mbr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, testdata, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
predict.MBRmodel <- function (object,
                              newdata,
                              ...,
                              probability = TRUE,
                              CORES = 1
) {
  #Validations
  if(class(object) != 'MBRmodel')
    stop('First argument must be an MBRmodel object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  newdata <- utiml_newdata(newdata)

  #1 Iteration - Base level
  base.preds <- predict(object$basemodel, newdata, ..., probability = FALSE, CORES = CORES)

  #2 Iteration - Meta level
  corr <- object$correlation
  predictions <- utiml_lapply(object$labels, function (labelname) {
    extracolumns <- base.preds[,colnames(corr)[corr[labelname,] > object$phi]]
    colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
    br.predict_model(object$metamodels[[labelname]], cbind(newdata, extracolumns), ...)
  }, CORES)
  names(predictions) <- object$labels

  as.resultMLPrediction(predictions, probability)
}

print.MBRmodel <- function (x, ...) {
  cat("Classifier Meta-BR (also called 2BR)\n\nCall:\n")
  print(x$call)
  cat("\nPhi:", x$phi, "\n")
  cat("\nCorrelation Table Overview:\n")
  corr <- x$correlation
  diag(corr) <- NA
  tbl <- data.frame(
    min = apply(corr, 1, min, na.rm = TRUE),
    mean = apply(corr, 1, mean, na.rm = TRUE),
    median = apply(corr, 1, median, na.rm = TRUE),
    max = apply(corr, 1, max, na.rm = TRUE),
    extra = apply(x$correlation, 1, function (row) sum(row > x$phi))
  )
  print(tbl)
}

print.mldMBR <- function (x, ...) {
  cat("Meta Binary Relevance Transformation Dataset\n\n")
  cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep="")
  cat("Dataset info:\n")
  cat(" ", ncol(x$data) - 1 - length(x$new.features), "Predictive attributes\n")
  cat(" ", length(x$new.features), " meta features\n")
  cat(" ", nrow(x$data), "Examples\n")
  cat("  ", round((sum(x$data[,ncol(x$data)] == 1) / nrow(x$data)) * 100, 1), "% of positive examples\n", sep="")
}

