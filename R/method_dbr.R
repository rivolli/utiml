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
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param estimate.models Logical value indicatind whether is necessary build
#'  Binary Relevance classifier for estimate process. The default implementaion
#'  use BR as estimators, however when other classifier is desirable then use
#'  the value \code{FALSE} to skip this process. (Default: \code{TRUE}).
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
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
#' @seealso \code{\link[=rdbr]{Recursive Dependent Binary Relevance}}
#' @export
#'
#' @examples
#' model <- dbr(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' # Use Random Forest as base algorithm and 4 cores
#' model <- dbr(toyml, 'RF', cores = 4)
#' }
dbr <- function(mdata,
                base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                estimate.models = TRUE, ...,
                cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # DBR Model class
  dbrmodel <- list(labels = rownames(mdata$labels), call = match.call())
  if (estimate.models) {
    dbrmodel$estimation <- br(mdata, base.algorithm, ...,
                              cores=cores, seed=seed)
  }

  # Create models
  labeldata <- as.data.frame(mdata$dataset[mdata$labels$index])
  for (i in seq(ncol(labeldata))) {
    labeldata[, i] <- factor(labeldata[, i], levels=c(0, 1))
  }

  labels <- utiml_rename(seq(dbrmodel$labels), dbrmodel$labels)
  dbrmodel$models <- utiml_lapply(labels, function(li) {
    utiml_create_model(
      utiml_prepare_data(
        utiml_create_binary_data(mdata, dbrmodel$labels[li], labeldata[-li]),
        "mldDBR", mdata$name, "dbr", base.algorithm
      ), ...
    )
  }, cores, seed)

  utiml_restore_seed()

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
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
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
#' # Passing a specif parameter for SVM predict algorithm
#' pred <- predict(model, toyml, na.action = na.fail)
#'
#' # Using other classifier (EBR) to made the labels estimatives
#' estimative <- predict(ebr(toyml), toyml)
#' model <- dbr(toyml, estimate.models = FALSE)
#' pred <- predict(model, toyml, estimative = estimative)
#' }
predict.DBRmodel <- function(object, newdata, estimative = NULL,
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., cores = getOption("utiml.cores", 1),
                             seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "DBRmodel") {
    stop("First argument must be an DBRmodel object")
  }

  if (is.null(object$estimation) && is.null(estimative)) {
    stop("The model requires an estimative matrix")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  newdata <- utiml_newdata(newdata)
  if (is.null(estimative)) {
    estimative <- predict.BRmodel(object$estimation, newdata,
                                  probability = FALSE, ...,
                                  cores = cores, seed = seed)
  }

  if ('mlresult' %in% class(estimative)) {
    estimative <- as.bipartition(estimative)
  }

  estimative <- as.data.frame(estimative)
  for (i in seq(ncol(estimative))) {
    estimative[,i] <- factor(estimative[,i], levels=c(0, 1))
  }

  labels <- utiml_rename(seq(object$labels), object$labels)
  predictions <- utiml_lapply(labels, function(li) {
    utiml_predict_binary_model(object$models[[li]],
                               cbind(newdata, estimative[, -li]),
                               ...)
  }, cores, seed)

  utiml_restore_seed()
  utiml_predict(predictions, probability)
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
