#' Dependent Binary Relevance (DBR) for multi-label Classification
#'
#' Create a DBR classifier to predic multi-label data. This is a simple approach
#' that enables the binary classifiers to discover existing label dependency by
#' themselves. The idea of DBR is exactly the same used in BR+ (the training
#' method is the same, excepted by the argument \code{estimate.models} that
#' indicate if the estimated models must be created).
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param estimate.models Logical value indicatind whether is necessary build
#'  Binary Relevance classifier for estimate process. The default implementaion
#'  use BR as estimators, however when other classifier is desirable then use
#'  the value \code{FALSE} to skip this process. (Default: \code{TRUE}).
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{DBRmodel} containing the set of fitted
#'  models, including:
#'  \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{estimation}{The BR model to estimate the values for the labels.
#'      Only when the \code{estimate.models = TRUE}.}
#'    \item{models}{A list of final models named by the label names.}
#'  }
#' @references
#'  Montanes, E., Senge, R., Barranquero, J., Ramon Quevedo, J., Jose Del Coz,
#'    J., & Hullermeier, E. (2014). Dependent binary relevance models for
#'    multi-label classification. Pattern Recognition, 47(3), 1494-1508.
#' @seealso \code{\link{rdbr}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- dbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- dbr(toyml, 'RF', CORES = 4)
#' }
dbr <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                estimate.models = TRUE, ...,
                CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # DBR Model class
  dbrmodel <- list(labels = rownames(mdata$labels), call = match.call())
  if (estimate.models) {
    dbrmodel$estimation <- br(mdata, base.method, ..., CORES = CORES)
  }

  # Create models
  labeldata <- mdata$dataset[mdata$labels$index]
  labels <- utiml_renames(seq(dbrmodel$labels), dbrmodel$labels)
  dbrmodel$models <- utiml_lapply(labels, function(li) {
    dbrdata  <- create_br_data(mdata, dbrmodel$labels[li], labeldata[-li])
    dataset <- prepare_br_data(dbrdata, "mldDBR", base.method)
    create_br_model(dataset, ...)
  }, CORES)

  class(dbrmodel) <- "DBRmodel"
  dbrmodel
}

#' Predict Method for DBR
#'
#' This function predicts values based upon a model trained by \code{dbr}.
#' In general this method is a restricted version of
#' \code{\link{predict.BRPmodel}} using the 'NU' strategy.
#'
#' As new feature is possible to use other multi-label classifier to predict the
#' estimate values of each label. To this use the prediction argument to inform
#' a result of other multi-label algorithm.
#'
#' @param object Object of class '\code{DBRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param estimative A matrix containing the bipartition result of other
#'  multi-label classification algorithm or an mlresult object with the
#'  predictions.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @references
#'  Montanes, E., Senge, R., Barranquero, J., Ramon Quevedo, J., Jose Del Coz,
#'    J., & Hullermeier, E. (2014). Dependent binary relevance models for
#'    multi-label classification. Pattern Recognition, 47(3), 1494-1508.
#' @seealso \code{\link[=dbr]{Dependent Binary Relevance (DBR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- dbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#'
#' # Using other classifier (EBR) to made the labels estimatives
#' estimative <- predict(ebr(toyml), toyml)
#' model <- dbr(toyml, estimate.models = FALSE)
#' pred <- predict(model, toyml, estimative = estimative)
#' }
predict.DBRmodel <- function(object, newdata, estimative = NULL,
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "DBRmodel") {
    stop("First argument must be an DBRmodel object")
  }

  if (is.null(object$estimation) && is.null(estimative)) {
    stop("The model requires an estimative matrix")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  newdata <- utiml_newdata(newdata)
  if (is.null(estimative)) {
    estimative <- predict(object$estimation, newdata, probability = FALSE, ...,
                          CORES = CORES)
  }
  else if ('mlresult' %in% class(estimative)) {
    estimative <- as.bipartition(estimative)
  }

  estimative <- as.matrix(estimative)
  labels <- utiml_renames(seq(object$labels), object$labels)
  predictions <- utiml_lapply(labels, function(li) {
    predict_br_model(object$models[[li]], cbind(newdata, estimative[, -li]),
                     ...)
  }, CORES)

  as.multilabelPrediction(predictions, probability)
}

#' Print DBR model
#' @param x The dbr model
#' @param ... ignored
#' @export
print.DBRmodel <- function(x, ...) {
  cat("Classifier DBR\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$models), "Models (labels):\n")
  print(names(x$models))
}
