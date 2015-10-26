#' @title Ensemble of Classifier Chains for multi-label Classification
#' @family Transformation methods
#' @family Ensemble methods
#' @description Create an Ensemble of Classifier Chains model for
#'   multilabel classification.
#'
#'   This model is composed by a set of Classifier Chains models.
#'   Classifier Chains is a Binary Relevance transformation method based
#'   to predict multi-label data. It is different from BR method due the strategy
#'   of extended the attribute space with the 0/1 label relevances of all previous
#'   classifiers, forming a classifier chain.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string with the name of base method. The same base method
#'   will be used for train all subproblems.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'})
#' @param m The number of Binary Relevance models used in the ensemble.
#' @param subsample A value between 0.1 and 1 to determine the percentage of
#'    training instances must be used for each interation. (default: 0.75)
#' @param attr.space A value between 0.1 and 1 to determine the percentage of
#'    attributes must be used for each interation. (default: 0.50)
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param SEED A single value, interpreted as an integer to allow obtain the
#'   same results again. (default: \code{NULL})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{ECCmodel} containing the set of fitted
#'   CC models, including: \describe{
#'   \item{rounds}{The number of interations}
#'   \item{models}{A list of BR models.}
#'   \item{nrow}{The number of instances used in each training dataset}
#'   \item{ncol}{The number of attributes used in each training dataset}
#' }
#'
#' @references
#'    Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier
#'    chains for multi-label classification. Machine Learning, 85(3), 333–359.
#'
#'    Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2009).
#'    Classifier Chains for Multi-label Classification. Machine Learning and
#'    Knowledge Discovery in Databases, Lecture Notes in Computer Science,
#'    5782, 254–269.
#'
#' @seealso \code{\link[=ecc]{Ensemble of classifier Chains (ECC)}}
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Ensemble of Classifier Chains
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Use all default values
#' model <- ecc(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Use C4.5 with 100% of instances and only 5 rounds
#' model <- ecc(dataset$train, "C4.5", m = 5, subsample = 1)
#' pred <- predict(model, dataset$test)
#'
#' # Use 75% of attributes and use a specific seed
#' model <- ecc(dataset$train, attr.space = 0.75, SEED = 1)
#' pred <- predict(model, dataset$test)
#'
#' # Running in 4 cores
#' model <- ecc(dataset$train, CORES=4)
#' pred <- predict(model, dataset$test, CORES=4)
ecc <- function (mdata,
                 base.method = "SVM",
                 m = 10,
                 subsample = 0.75,
                 attr.space = 0.5,
                 ...,
                 SEED = NULL,
                 CORES = 1
) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if(m <= 1)
    stop('The number of iterations (m) must be greater than 1')

  if (subsample < 0.1 || subsample > 1)
    stop("The subset of training instances must be between 0.1 and 1 inclusive")

  if (attr.space <= 0.1 || attr.space > 1)
    stop("The attribbute space of training instances must be between 0.1 and 1 inclusive")

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BR Model class
  eccmodel <- list()
  eccmodel$rounds <- m
  eccmodel$nrow <- ceiling(mdata$measures$num.instances * subsample)
  eccmodel$ncol <- ceiling(length(mdata$attributesIndexes) * attr.space)

  if (!is.null(SEED)) {
    eccmodel$seed <- SEED
    set.seed(SEED)
  }

  eccmodel$models <- lapply(1:m, function (iteration){
    ndata <- mldr_random_subset(mdata, eccmodel$nrow, eccmodel$ncol)
    chain <- sample(rownames(ndata$labels))
    ccmodel <- cc(ndata, base.method, chain, ..., CORES = CORES)
    ccmodel$attrs <- colnames(ndata$dataset[,ndata$attributesIndexes])
    ccmodel
  })

  eccmodel$call <- match.call()
  class(eccmodel) <- "ECCmodel"

  if (!is.null(SEED))
    set.seed(NULL)

  eccmodel
}

#' @title Predict Method for Ensemble of Classifier Chains
#' @description This function predicts values based upon a model trained
#'  by \code{\link{ecc}}.
#'
#' @param object Object of class "\code{ECCmodel}", created by \code{\link{ecc}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param vote.schema Define the way that ensemble must compute the predictions.
#'  The valid options are describe in \link{utiml_vote.schema_method}. (default: "MAJ")
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
#' @seealso \code{\link[=ecc]{Ensemble of Classifier Chains (ECC)}}
#' @export
#'
#' @examples
#' # Emotion multi-label dataset using Ensemble of Binary Relevance
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Predict SVM scores
#' model <- ecc(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, dataset$test, probability = FALSE, CORES = 6)
#'
#' # Return the classes with the highest score
#' pred <- predict(model, dataset$test, vote.schema = "MAX")
predict.ECCmodel <- function (object,
                              newdata,
                              vote.schema = "MAJ",
                              probability = TRUE,
                              ...,
                              CORES = 1
) {
  #Validations
  if(class(object) != 'ECCmodel')
    stop('First argument must be an ECCmodel object')

  if (is.null(utiml_vote.schema_method(vote.schema)))
    stop("Invalid vote schema")

  if (CORES < 1)
    stop('Cores must be a positive value')

  newdata <- utiml_newdata(newdata)
  allpreds <- utiml_lapply(object$models, function (ccmodel) {
    predict(ccmodel, newdata[,ccmodel$attrs], ...)
  }, CORES)

  utiml_compute_multilabel_ensemble(allpreds, vote.schema, probability)
}

print.ECCmodel <- function (x, ...) {
  cat("Ensemble of Classifier Chains Model\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", x$nrow, "Instances")
  cat("\n ", x$ncol, "Attributes\n")
  if (!is.null(x$seed))
    cat("\nSeed value:", x$seed)
}
