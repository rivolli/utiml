#' Calibrated Label Ranking (CLR) for multi-label Classification
#'
#' Create a CLR model for multilabel classification.
#'
#' CLR is an extension of label ranking that incorporates the calibrated
#' scenario. The introduction of an artificial calibration label,
#' separates the relevant from the irrelevant labels.
#'
#' @family Transformation methods
#' @family Pairwise methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param ... Others arguments passed to the base algorithm for all subproblems
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{RPCmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{rpcmodel}{A RPC model.}
#'    \item{brmodel}{A BR model used to calibrated the labels.}
#'   }
#' @references
#'  Brinker, K., Furnkranz, J., & Hullermeier, E. (2006). A unified model for
#'    multilabel classification and ranking. In Proceeding of the ECAI 2006:
#'    17th European Conference on Artificial Intelligence. p. 489-493.
#'  Furnkranz, J., Hullermeier, E., Loza Mencia, E., & Brinker, K. (2008).
#'    Multilabel classification via calibrated label ranking.
#'    Machine Learning, 73(2), 133-153.
#' @export
#'
#' @examples
#' model <- clr(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' }
clr <- function(mdata,
                base.algorithm = getOption("utiml.base.algorithm", "SVM"), ...,
                cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  # CLR Model class
  clrmodel <- list(labels = rownames(mdata$labels), call = match.call())

  # Create pairwise models
  clrmodel$rpcmodel <- rpc(mdata, base.algorithm, ..., cores=cores, seed=seed)

  # Create calibrated models
  clrmodel$brmodel <- br(mdata, base.algorithm, ..., cores=cores, seed=seed)

  class(clrmodel) <- "CLRmodel"
  clrmodel
}

#' Predict Method for CLR
#'
#' This function predicts values based upon a model trained by
#' \code{\link{clr}}.
#'
#' @param object Object of class '\code{CLRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
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
#' @seealso \code{\link[=br]{Binary Relevance (BR)}}
#' @export
#'
#' @examples
#' model <- clr(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' }
predict.CLRmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "CLRmodel") {
    stop("First argument must be an CLRmodel object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # Predict RPC models
  predictions <- as.matrix(predict.RPCmodel(object$rpcmodel, newdata, TRUE,
                                            ..., cores=cores, seed=seed))

  previous.value <- getOption("utiml.empty.prediction")
  options(utiml.empty.prediction = TRUE)
  calibrated <- as.matrix(predict.BRmodel(object$brmodel, newdata, FALSE, ...,
                                          cores=cores, seed=seed))
  options(utiml.empty.prediction = previous.value)

  utiml_restore_seed()

  # Compute votes
  l0 <- (length(object$labels) - rowSums(calibrated)) / length(object$labels)
  bipartitions <- apply(predictions >= l0, 2, as.numeric)

  multilabel_prediction(bipartitions, predictions, probability)
}

#' Print CLR model
#' @param x The br model
#' @param ... ignored
#' @export
print.CLRmodel <- function(x, ...) {
  cat("CLR Model\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$rpcmodel$models) + length(x$labels), " pairwise models\n", sep='')}
