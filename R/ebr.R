#' @title Ensemble of Binary Relevance for multi-label Classification
#' @family Transformation methods
#' @family Ensemble
#' @description Create an Ensemble of Binary Relevance model for
#'   multilabel classification.
#'
#'   This model is composed by a set of Binary Relevance models.
#'   Binary Relevance is a simple and effective transformation method
#'   to predict multi-label data.
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
#' @param m The number of interations or Binary Relevance models to use in the
#'    ensemble.
#' @param subsample A value between 0.1 and 1 to determine the percentage of
#'    training instances must be used for each interation. (default: 0.75)
#' @param attr.space A value between 0.1 and 1 to determine the percentage of
#'    attributes must be used for each interation. (default: 0.50)
#' @param ... Others arguments passed to the base method for all subproblems
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: \code{FALSE})
#' @param SEED A single value, interpreted as an integer to allow obtain the
#'   same results again. (default: \code{NULL})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{EBRmodel} containing the set of fitted
#'   BR models, including: \describe{ \item{rounds}{The number of interations}
#'   \item{models}{A list of BR models.} \item{nrow}{The number of instances
#'   used in each training dataset} \item{ncol}{The number of attributes used
#'   in each training dataset} \item{seed}{The value of the seed, present only
#'   when the \code{SEED} is defined.}}
#'
#' @section Warning:
#'    RWeka package does not permit use \code{'C4.5'} in parallel mode, use
#'    \code{'C5.0'} or \code{'CART'} instead of it.
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
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Ensemble of Binary Relevance
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use all default values
#' model <- ebr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use C4.5 with 100% of instances and only 5 rounds
#' model <- ebr(emotions, "C4.5", m = 5, subsample = 1)
#' pred <- predict(model, testdata)
#'
#' # Use 75% of attributes and use a specific seed
#' model <- ebr(emotions, attr.space = 0.75, SEED = 1)
#' pred <- predict(model, testdata)
ebr <- function (mdata,
                base.method = "SVM",
                m = 10,
                subsample = 0.75,
                attr.space = 0.5,
                ...,
                save.datasets = FALSE,
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
  ebrmodel <- list()
  ebrmodel$rounds <- m
  ebrmodel$nrow <- ceiling(mdata$measures$num.instances * subsample)
  ebrmodel$ncol <- ceiling(length(mdata$attributesIndexes) * attr.space)

  if (!is.null(SEED)) {
    ebrmodel$seed <- SEED
    set.seed(SEED)
  }

  ebrmodel$models <- lapply(1:m, function (iteration){
    ndata <- mldr_random_subset(mdata, ebrmodel$nrow, ebrmodel$ncol)
    brmodel <- br(ndata, base.method, ..., save.datasets = save.datasets, CORES = CORES)
    brmodel$attrs <- colnames(ndata$dataset[,ndata$attributesIndexes])
    brmodel
  })

  ebrmodel$call <- match.call()
  class(ebrmodel) <- "EBRmodel"

  if (!is.null(SEED))
    set.seed(NULL)

  ebrmodel
}

#' @title Predict Method for Ensemble of Binary Relevance
#' @description This function predicts values based upon a model trained
#'  by \code{\link{ebr}}.
#'
#' @param object Object of class "\code{EBRmodel}", created by \code{\link{ebr}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param vote.schema Define the way that ensemble must compute the predictions.
#' The valid options are: \describe{
#'  \code{'score'}{Compute the averages of probabilities},
#'  \code{'majority'}{Compute the votes scaled between 0 and \code{m} (number of interations)},
#'  \code{'prop'}{Compute the proportion of votes, scale data between min and max of votes} }
#'  (default: \code{'score'})
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
#' @section Warning:
#'    RWeka package does not permit use \code{'C4.5'} in parallel mode, use
#'    \code{'C5.0'} or \code{'CART'} instead of it
#'
#' @seealso \code{\link[=ebr]{Ensemble of Binary Relevance (EBR)}}
#' @export
#'
#' @examples
#' library(utiml)
#'
#' # Emotion multi-label dataset using Ensemble of Binary Relevance
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- ebr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, testdata, probability = FALSE, CORES = 6)
#'
#' # Return the classes with have at least half of votes
#' pred <- predict(model, testdata, vote.schema = "majority", probability = FALSE)
predict.EBRmodel <- function (object,
                             newdata,
                             vote.schema = c("score", "majority", "prop"),
                             ...,
                             probability = TRUE,
                             CORES = 1
) {
  #Validations
  if(class(object) != 'EBRmodel')
    stop('First argument must be an EBRmodel object')

  schemas <- c("score", "majority", "prop")
  if(!vote.schema[1] %in% schemas)
    stop(paste("Vote schema value must be '", paste(schemas, collapse = "' or '"), "'", sep=""))

  if (CORES < 1)
    stop('Cores must be a positive value')

  newdata <- utiml_newdata(newdata)
  allpreds <- lapply(model$models, function (brmodel) {
    predict(brmodel, newdata[,brmodel$attrs], ..., probability = vote.schema[1] == "score", CORES = CORES)
  })

  predictions <- utiml_compute_ensemble_predictions(allpreds, vote.schema[1])
  as.resultMLPrediction(predictions, probability)
}

print.EBRmodel <- function (x, ...) {
  cat("Ensemble of Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", x$nrow, "Instances")
  cat("\n ", x$ncol, "Attributes\n")
  if (!is.null(x$seed))
    cat("\nSeed value:", x$seed)
}
