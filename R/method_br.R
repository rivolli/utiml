#' Binary Relevance for multi-label Classification
#'
#' Create a Binary Relevance model for multilabel classification.
#'
#' Binary Relevance is a simple and effective transformation method to predict
#' multi-label data. This is based on the one-versus-all approach to build a
#' specific model for each label.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param ... Others arguments passed to the base method for all subproblems
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{BRmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{models}{A list of the generated models, named by the label names.}
#'   }
#' @references
#'  Boutell, M. R., Luo, J., Shen, X., & Brown, C. M. (2004). Learning
#'    multi-label scene classification. Pattern Recognition, 37(9), 1757–1771.
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- br(toyml)
#' pred <- predict(model, toyml)
#'
#' # Change the default base method and use 4 CORES
#' model <- br(toyml[1:50], 'RF', CORES = 4)
#'
#' # Set a parameters for all subproblems
#' model <- br(toyml, 'KNN', k=5)
#' }
br <- function(mdata, base.method = getOption("utiml.base.method", "SVM"), ...,
               CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # BR Model class
  brmodel <- list(labels = rownames(mdata$labels), call = match.call())

  # Create models
  labels <- utiml_renames(brmodel$labels)
  brmodel$models <- utiml_lapply(labels, function (label, ...) {
    cat(label, paste(round(mem_used() / 1024 / 1024, 3), "MB"), "\n")
    brdata  <- create_br_data(mdata, label)
    dataset <- prepare_br_data(brdata,
                               classname = "mldBR",
                               base.method = base.method)
    cat(label, paste(round(mem_change(model <- create_br_model(dataset, ...)) / 1024 / 1024, 3), "MB"), "\n")
    model
  }, CORES, ...)

  class(brmodel) <- "BRmodel"
  brmodel
}

#' Predict Method for Binary Relevance
#'
#' This function predicts values based upon a model trained by \code{\link{br}}.
#'
#' @param object Object of class '\code{BRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=br]{Binary Relevance (BR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- br(toyml, "SVM")
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions running in 4 cores
#' pred <- predict(model, toyml, probability = FALSE, CORES = 4)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, dataset$test, na.action = na.fail)
#' }
predict.BRmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "BRmodel") {
    stop("First argument must be an BRmodel object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # Create models
  predictions <- utiml_lapply(object$models,
                              predict_br_model,
                              CORES,
                              newdata = utiml_newdata(newdata),
                              ...)
  as.multilabelPrediction(predictions, probability)
}

#' Print BR model
#' @export
print.BRmodel <- function(x, ...) {
  cat("Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$labels), "Models (labels):\n")
  print(x$labels)
}
