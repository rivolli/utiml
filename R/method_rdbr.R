#' Recursive Dependent Binary Relevance (RDBR) for multi-label Classification
#'
#' Create a RDBR classifier to predict multi-label data. This is a recursive
#' approach that enables the binary classifiers to discover existing label
#' dependency by themselves. The idea of RDBR is running DBR recursively until
#' the results stabilization of the result.
#'
#' The train method is exactly the same of DBR the recursion is in the predict
#' method.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param estimate.models Logical value indicating whether is necessary build
#'  Binary Relevance classifier for estimate process. The default implementation
#'  use BR as estimators, however when other classifier is desirable then use
#'  the value \code{FALSE} to skip this process. (Default: \code{TRUE}).
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{RDBRmodel} containing the set of fitted
#'  models, including:
#'  \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{estimation}{The BR model to estimate the values for the labels.
#'      Only when the \code{estimate.models = TRUE}.}
#'    \item{models}{A list of final models named by the label names.}
#'  }
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejao, F. M.
#'   (2014). Recursive Dependent Binary Relevance Model for Multi-label
#'   Classification. In Advances in Artificial Intelligence - IBERAMIA, 206-217.
#' @seealso \code{\link[=dbr]{Dependent Binary Relevance (DBR)}}
#' @export
#'
#' @examples
#' model <- rdbr(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \donttest{
#' # Use Random Forest as base algorithm and 2 cores
#' model <- rdbr(toyml, 'RF', cores = 2, seed = 123)
#' }
rdbr <- function(mdata,
                 base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                 estimate.models = TRUE, ...,
                 cores = getOption("utiml.cores", 1),
                 seed = getOption("utiml.seed", NA)) {
  rdbrmodel <- dbr(mdata, base.algorithm, estimate.models, ...,
                   cores=cores, seed=seed)
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
#' @param object Object of class '\code{RDBRmodel}'.
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
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejao, F. M.
#'   (2014). Recursive Dependent Binary Relevance Model for Multi-label
#'   Classification. In Advances in Artificial Intelligence - IBERAMIA, 206-217.
#' @seealso \code{\link[=rdbr]{Recursive Dependent Binary Relevance (RDBR)}}
#' @export
#'
#' @examples
#' \donttest{
#' # Predict SVM scores
#' model <- rdbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Passing a specif parameter for SVM predict algorithm
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
                              ..., cores = getOption("utiml.cores", 1),
                              seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "RDBRmodel")) {
    stop("First argument must be an RDDBRmodel object")
  }
  if (is.null(object$estimation) && is.null(estimative)) {
    stop("The model requires an estimative matrix")
  }
  if (max.iterations < 1) {
    stop("The number of iteractions must be positive")
  }
  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  if (!anyNA(seed)) {
    set.seed(seed)
  }

  newdata <- utiml_newdata(newdata)

  if (is.null(estimative)) {
    estimative <- predict.BRmodel(object$estimation, newdata, FALSE, ...,
                                  cores=cores, seed=seed)
  }

  if (is(estimative, 'mlresult')) {
    estimative <- as.bipartition(estimative)
  }

  estimative <- as.data.frame(estimative)
  for (i in seq(ncol(estimative))) {
    estimative[,i] <- factor(estimative[,i], levels=c(0, 1))
  }

  labels <- names(object$models)
  modelsindex <- utiml_rename(seq(labels), labels)
  if (batch.mode) {
    for (i in seq(max.iterations)) {
      old.estimative <- estimative
      predictions <- utiml_lapply(modelsindex, function(li) {
        utiml_predict_binary_model(object$models[[li]],
                                   cbind(newdata, estimative[, -li]), ...)
      }, cores, seed)

      for (j in seq(predictions)) {
        classes <- predictions[[j]]$bipartition
        estimative[, j] <- factor(classes, levels=c(0, 1))
      }

      if (all(old.estimative == estimative)) {
        break
      }
    }
  }
  else {
    for (i in seq(max.iterations)) {
      old.estimative <- estimative
      predictions <- list()

      # the labels needs to be shuffled in each iteraction
      for (li in sample(modelsindex)) {
        predictions[[li]] <- utiml_predict_binary_model(object$models[[li]],
                                              cbind(newdata, estimative[, -li]),
                                              ...)
        estimative[, li] <- factor(predictions[[li]]$bipartition, levels=c(0,1))
      }

      names(predictions) <- labels
      if (all(old.estimative == estimative)) {
        break
      }
    }
  }

  utiml_predict(predictions, probability)
}

#' Print RDBR model
#' @param x The rdbr model
#' @param ... ignored
#'
#' @return No return value, called for print model's detail
#'
#' @export
print.RDBRmodel <- function(x, ...) {
    cat("Classifier RDBR\n\nCall:\n")
    print(x$call)
    cat("\n", length(x$models), "Models (labels):\n")
    print(names(x$models))
}
