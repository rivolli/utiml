#' Ensemble of Pruned Set for multi-label Classification
#'
#' Create an Ensemble of Pruned Set model for multilabel classification.
#'
#' Pruned Set (PS) is a multi-class transformation that remove the less common
#' classes to predict multi-label data. The ensemble is created with different
#' subsets of the original multi-label data.
#'
#' @family Transformation methods
#' @family Powerset
#' @family Ensemble methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param m The number of Pruned Set models used in the ensemble.
#' @param subsample A value between 0.1 and 1 to determine the percentage of
#'    training instances that must be used for each classifier. (Default: 0.63)
#' @param p Number of instances to prune. All labelsets that occurs p times or
#'  less in the training data is removed. (Default: 3)
#' @param strategy The strategy  (A or B) for processing infrequent labelsets.
#'    (Default: A).
#' @param b The number used by the strategy for processing infrequent labelsets.
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#' @return An object of class \code{EPSmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{rounds}{The number of interactions}
#'    \item{models}{A list of PS models.}
#'   }
#' @references
#'  Read, J. (2008). A pruned problem transformation method for multi-label
#'  classification. In Proceedings of the New Zealand Computer Science Research
#'  Student Conference (pp. 143-150).
#' @export
#'
#' @examples
#' model <- eps(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \donttest{
#' ##Change default configurations
#' model <- eps(toyml, "RF", m=15, subsample=0.4, p=4, strategy="B", b=4)
#' }
eps <- function (mdata,
                 base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                m = 10, subsample = 0.75, p = 3, strategy = c("A", "B"), b = 2,
                ..., cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(mdata, "mldr")) {
    stop("First argument must be an mldr object")
  }

  if (m <= 1) {
    stop("The number of iterations (m) must be greater than 1")
  }

  if (subsample < 0.1 || subsample > 1) {
    stop("The subset of training instances must be between 0.1 and 1 inclusive")
  }

  if (p < 1) {
    stop("The prunning value must be greater than 0")
  }

  strategy <- match.arg(strategy)

  if (b < 0) {
    stop("The parameter b must be greater or equal than 0")
  }

  # EPS Model class
  epsmodel <- list(rounds = m, p = p, strategy = strategy, b = b,
                   nrow = ceiling(mdata$measures$num.instances * subsample),
                   call = match.call())

  utiml_preserve_seed()
  if (!anyNA(seed)) {
    set.seed(seed)
  }
  idxs <- lapply(seq(m), function(iteration) {
    sample(mdata$measures$num.instances, epsmodel$nrow)
  })

  epsmodel$models <- utiml_lapply(idxs, function(idx) {
    ps(create_subset(mdata, idx), base.algorithm = base.algorithm, p = p,
       strategy = strategy, b = b, ..., seed = seed)
  }, cores, seed)

  utiml_restore_seed()
  class(epsmodel) <- "EPSmodel"
  epsmodel
}

#' Predict Method for Ensemble of Pruned Set Transformation
#'
#' This function predicts values based upon a model trained by
#'  \code{\link{eps}}. Different from the others methods the probability value,
#'  is actually, the sum of all probability predictions such as it is described
#'  in the original paper.
#'
#' @param object Object of class '\code{EPSmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param threshold A threshold value for producing bipartitions. (Default: 0.5)
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the prediction. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. (Default:
#'   \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=eps]{Ensemble of Pruned Set (EPS)}}
#' @export
#'
#' @examples
#' model <- eps(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.EPSmodel <- function(object, newdata, threshold = 0.5,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "EPSmodel")) {
    stop("First argument must be a EPSmodel object")
  }

  previous.value <- getOption("utiml.empty.prediction")
  options(utiml.empty.prediction = TRUE)

  newdata <- utiml_newdata(newdata)
  utiml_preserve_seed()

  results <- utiml_lapply(object$models, function (psmodel){
    res <- predict.PSmodel(psmodel, newdata)
    as.probability(res) * as.bipartition(res)
  }, cores, seed)

  utiml_restore_seed()
  options(utiml.empty.prediction = previous.value)

  as.mlresult(Reduce('+', results), probability = probability,
              threshold = threshold)
}

#' Print EPS model
#' @param x The ps model
#' @param ... ignored
#'
#' @return No return value, called for print model's detail
#'
#' @export
print.EPSmodel <- function(x, ...) {
  cat("Ensemble of Pruned Set Model\n\nCall:\n")
  print(x$call)

  cat("\nModels:", x$rounds, "\n")
  cat("Instance by models: ", x$nrow, "\n")
  cat("Prune:", x$p, "\n")
  cat("Strategy:", x$strategy, "\n")
  cat("B value:", x$b, "\n")
}
