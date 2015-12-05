#' Ensemble of Binary Relevance for multi-label Classification
#'
#' Create an Ensemble of Binary Relevance model for multilabel classification.
#'
#' This model is composed by a set of Binary Relevance models. Binary Relevance
#' is a simple and effective transformation method to predict multi-label data.
#'
#' @family Transformation methods
#' @family Ensemble methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param m The number of Binary Relevance models used in the ensemble.
#' @param subsample A value between 0.1 and 1 to determine the percentage of
#'    training instances that must be used for each classifier. (Default: 0.75)
#' @param attr.space A value between 0.1 and 1 to determine the percentage of
#'    attributes that must be used for each classifier. (Default: 0.50)
#' @param ... Others arguments passed to the base method for all subproblems
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{EBRmodel} containing the set of fitted
#'   BR models, including:
#' \describe{
#'   \item{models}{A list of BR models.}
#'   \item{nrow}{The number of instances used in each training dataset.}
#'   \item{ncol}{The number of attributes used in each training dataset.}
#'   \item{rounds}{The number of interations.}
#' }
#' @references
#'    Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier
#'    chains for multi-label classification. Machine Learning, 85(3), 333–359.
#'
#'    Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2009).
#'    Classifier Chains for Multi-label Classification. Machine Learning and
#'    Knowledge Discovery in Databases, Lecture Notes in Computer Science,
#'    5782, 254–269.
#' @note If you want to reproduce the same classification and obtain the same
#'  result will be necessary set a flag utiml.mc.set.seed to FALSE.
#' @export
#'
#' @examples
#' \dontrun{
#' # Use all default values
#' model <- ebr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use J48 with 90% of instances and only 5 rounds
#' model <- ebr(toyml, 'J48', m = 5, subsample = 0.9)
#'
#' # Use 75% of attributes
#' model <- ebr(dataset$train, attr.space = 0.75)
#'
#' # Running in 4 cores and define a specific seed
#' options(utiml.mc.set.seed = FALSE)
#' set.seed(91179631)
#' model1 <- ebr(toyml, CORES=4)
#'
#' set.seed(91179631)
#' model2 <- ebr(toyml, CORES=4)
#' }
ebr <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                m = 10, subsample = 0.75, attr.space = 0.5, ...,
                CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (m < 2) {
    stop("The number of iterations (m) must be greater than 1")
  }

  if (subsample < 0.1 || subsample > 1) {
    stop("The subset of training instances must be between 0.1 and 1 inclusive")
  }

  if (attr.space <= 0.1 || attr.space > 1) {
    stop("The attribbute space of training instances must be between 0.1 and 1 inclusive")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # EBR Model class
  ebrmodel <- list(rounds = m, call = match.call())
  ebrmodel$nrow <- ceiling(mdata$measures$num.instances * subsample)
  ebrmodel$ncol <- ceiling(length(mdata$attributesIndexes) * attr.space)

  ebrmodel$models <- lapply(seq(m), function(iteration) {
    ndata <- create_random_subset(mdata, ebrmodel$nrow, ebrmodel$ncol)
    brmodel <- br(ndata, base.method, ..., CORES = CORES)
    brmodel$attrs <- colnames(ndata$dataset[, ndata$attributesIndexes])
    brmodel
  })

  class(ebrmodel) <- "EBRmodel"
  ebrmodel
}

#' Predict Method for Ensemble of Binary Relevance
#'
#' This method predicts values based upon a model trained by \code{\link{ebr}}.
#'
#' @param object Object of class '\code{EBRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param vote.schema Define the way that ensemble must compute the predictions.
#'  The default valid options are:
#'  \describe{
#'    \code{'avg'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'maj'}{Compute the averages of probabilities},
#'    \code{'max'}{Compute the votes scaled between 0 and \code{m}
#'      (number of interations)},
#'    \code{'min'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'prod'}{Compute the product of all votes for each instance}
#'  }
#'  If \code{NULL} then all predictions are returned. (Default: \code{'maj'})
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=ebr]{Ensemble of Binary Relevance (EBR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- ebr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, toyml, prob = FALSE, CORES = 6)
#'
#' # Return the classes with the highest score
#' pred <- predict(model, toyml, vote = 'max')
#' }
predict.EBRmodel <- function(object, newdata, vote.schema = "maj",
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "EBRmodel") {
    stop("First argument must be an EBRmodel object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  newdata <- utiml_newdata(newdata)
  allpreds <- lapply(object$models, function(brmodel) {
    predict(brmodel, newdata[, brmodel$attrs], ..., CORES = CORES)
  })

  compute_multilabel_ensemble_votes(allpreds, vote.schema, probability)
}

#' Print EBR model
#' @export
print.EBRmodel <- function(x, ...) {
  cat("Ensemble of Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", x$nrow, "Instances")
  cat("\n ", x$ncol, "Attributes\n")
}
