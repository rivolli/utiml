#' Recursive Dependent Binary Relevance (RDBR) for multi-label Classification
#'
#' Create a RDBR classifier to predic multi-label data. This is a recursive
#' approach that enables the binary classifiers to discover existing label
#' dependency by themselves. The idea of RDBR is running DBR recursivelly until
#' the results stabilization of the result.
#'
#' The train method is exactly the same of DBR the recursivity is in the predict
#' method.
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
#' @return An object of class \code{RDBRmodel} containing the set of fitted
#'  models, including:
#'  \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{estimation}{The BR model to estimate the values for the labels.
#'      Only when the \code{estimate.models = TRUE}.}
#'    \item{models}{A list of final models named by the label names.}
#'  }
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejão, F. M.
#'   (2014). Recursive Dependent Binary Relevance Model for Multi-label
#'   Classification. In Advances in Artificial Intelligence - IBERAMIA, 206–217.
#' @seealso \code{\link[=dbr]{Dependent Binary Relevance (DBR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- rdbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- rdbr(toyml, 'RF', CORES = 4)
#' }
rdbr <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                 estimate.models = TRUE, ...,
                 CORES = getOption("utiml.cores", 1)) {
  rdbrmodel <- dbr(mdata, base.method, estimate.models, ..., CORES = CORES)
  class(rdbrmodel) <- "RDBRmodel"
  rdbrmodel
}

#' Predict Method for RDBR
#'
#' This function predicts values based upon a model trained by \code{rdbr}.
#' In general this method is a recursive version of
#' \code{\link{predict.DBRmodel}}.
#'
#' Two versions of the update strategy of the estimated labels are implemented.
#' The batch re-estimates the labels only when a complete current label vector
#' is available. The stochastic uses re-estimated labels as soon as they become
#' available. This second does not support parallelize the prediction, however
#' stabilizes earlier than batch mode.
#'
#' @param object Object of class '\code{DBRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param estimative A matrix containing the bipartition result of other
#'  multi-label classification algorithm or an mlresult object with the
#'  predictions.
#' @param max.iterations The maximum allowed iterations of the RDBR technique.
#'   (Default: 5)
#' @param batch.mode Logical value to determine if use the batch re-estimation.
#'  If \code{FALSE} then use the stochastic re-estimation strategy.
#'  (Default: \code{FALSE})
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejão, F. M.
#'   (2014). Recursive Dependent Binary Relevance Model for Multi-label
#'   Classification. In Advances in Artificial Intelligence - IBERAMIA, 206–217.
#' @seealso \code{\link[=rdbr]{Recursive Dependent Binary Relevance (RDBR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- rdbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#'
#' # Use the batch mode and increase the max number of iteration to 10
#' pred <- predict(model, toyml, max.iterations = 10, batch.mode = TRUE)
#'
#' # Using other classifier (EBR) to made the labels estimatives
#' estimative <- predict(ebr(toyml), toyml, probability = FALSE)
#' model <- rdbr(toyml, estimate.models = FALSE)
#' pred <- predict(model, toyml, estimative = estimative)
#' }
predict.RDBRmodel <- function(object, newdata, estimative = NULL,
                              max.iterations = 5, batch.mode = FALSE,
                              probability = getOption("utiml.use.probs", TRUE),
                              ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "RDBRmodel") {
    stop("First argument must be an RDDBRmodel object")
  }
  if (is.null(object$estimation) && is.null(estimative)) {
    stop("The model requires an estimative matrix")
  }
  if (max.iterations < 1) {
    stop("The number of iteractions must be positive")
  }
  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  newdata <- utiml_newdata(newdata)
  if (is.null(estimative)) {
    estimative <- predict(object$estimation, newdata, FALSE, ..., CORES = CORES)
  }

  labels <- names(object$models)
  modelsindex <- utiml_renames(seq(labels), labels)
  if (batch.mode) {
    for (i in seq(max.iterations)) {
      predictions <- utiml_lapply(modelsindex, function(li) {
        predict_br_model(object$models[[li]],
                         cbind(newdata, estimative[, -li]), ...)
      }, CORES)

      new.estimative <- do.call(cbind, lapply(predictions,
                                              function(lbl) lbl$bipartition))
      if (all(new.estimative == estimative)) {
        break
      }
      estimative <- new.estimative
    }
  }
  else {
    for (i in seq(max.iterations)) {
      old.estimative <- estimative
      predictions <- list()

      # the labels needs to be shuffled in each iteraction
      for (li in modelsindex) {
        predictions[[li]] <- predict_br_model(object$models[[li]],
                                              cbind(newdata, estimative[, -li]),
                                              ...)
        estimative[, li] <- predictions[[li]]$bipartition
      }
      names(predictions) <- labels
      if (all(old.estimative == estimative)) {
        break
      }
    }
  }

  as.multilabelPrediction(predictions, probability)
}

#' Print RDBR model
#' @export
print.RDBRmodel <- function(x, ...) {
    cat("Classifier RDBR\n\nCall:\n")
    print(x$call)
    cat("\n", length(x$models), "Models (labels):\n")
    print(names(x$models))
}
