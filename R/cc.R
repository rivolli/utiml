#' @title Classifier Chains for multi-label Classification
#' @family Transformation methods
#' @family Chain methods
#' @description Create a Classifier Chains model for multilabel classification.
#'
#'   Classifier Chains is a Binary Relevance transformation method based to predict
#'   multi-label data. This is based on the one-versus-all approach to build a
#'   specific model for each label. It is different from BR method due the strategy
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
#' @param chain A vector with the label names to define the chain order. If
#'   empty the chain is the default label sequence of the dataset. (default:
#'   \code{list()})
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param CORES he number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{CCmodel} containing the set of fitted
#'   models, including: \describe{
#'   \item{chain}{A vector with the chain order}
#'   \item{labels}{A vector with the label names in expected order}
#'   \item{models}{A list of models named by the label names.}
#' }
#'
#' @references
#'  Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains
#'    for multi-label classification. Machine Learning, 85(3), 333–359.
#'
#'  Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2009). Classifier Chains
#'    for Multi-label Classification. Machine Learning and Knowledge Discovery
#'    in Databases, Lecture Notes in Computer Science, 5782, 254–269.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Classifier Chains
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Use SVM as base method
#' model <- cc(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Use a specific chain with C4.5 classifier
#' mychain <- sample(rownames(dataset$train$labels))
#' model <- cc(dataset$train, 'C4.5', mychain)
#' pred <- predict(model, dataset$test)
#'
#' # Set a specific parameter
#' model <- cc(dataset$train, 'KNN', k=5)
#' pred <- predict(model, dataset$test)
cc <- function(mdata, base.method = "SVM", chain = c(), ..., CORES = 1) {
    # Validations
    if (class(mdata) != "mldr")
        stop("First argument must be an mldr object")

    labels <- rownames(mdata$labels)
    if (length(chain) == 0)
        chain <- rownames(mdata$labels) else {
        if (length(chain) != mdata$measures$num.labels || length(setdiff(union(chain, labels), intersect(chain, labels))) > 0) {
            stop("Invalid chain (all labels must be on the chain)")
        }
    }

    # CC Model class
    ccmodel <- list()
    ccmodel$labels <- labels
    ccmodel$chain <- chain
    ccmodel$models <- list()

    basedata <- mdata$dataset[mdata$attributesIndexes]
    labeldata <- mdata$dataset[mdata$labels$index][chain]
    datasets <- utiml_lapply(1:mdata$measures$num.labels, function(labelIndex) {
        data <- cbind(basedata, labeldata[1:labelIndex])
        prepare_br_data(data, "mldCC", base.method, chain.order = labelIndex)
    }, CORES)
    names(datasets) <- chain
    ccmodel$models <- utiml_lapply(datasets, create_br_model, CORES, ...)

    ccmodel$call <- match.call()
    class(ccmodel) <- "CCmodel"

    ccmodel
}

#' @title Predict Method for Classifier Chains
#' @description This function predicts values based upon a model trained by \code{cc}.
#'
#' @param object Object of class '\code{CCmodel}', created by \code{\link{cc}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @seealso \code{\link[=cc]{Classifier Chains (CC)}}
#'
#' @export
#'
#' @examples
#' # Emotion multi-label dataset using Classifier Chains
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Predict SVM scores
#' model <- cc(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, dataset$test, prob = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, dataset$test, na.action = na.fail)
predict.CCmodel <- function(object, newdata, probability = TRUE, ...) {
    # Validations
    if (class(object) != "CCmodel")
        stop("First argument must be an CCmodel object")

    newdata <- utiml_newdata(newdata)
    predictions <- list()
    for (label in object$chain) {
        predictions[[label]] <- predict_br_model(object$models[[label]], newdata, ...)
        newdata <- cbind(newdata, predictions[[label]]$bipartition)
        names(newdata)[ncol(newdata)] <- label
    }

    as.multilabelPrediction(predictions[object$labels], probability)
}

print.CCmodel <- function(x, ...) {
    cat("Classifier Chains Model\n\nCall:\n")
    print(x$call)
    cat("\n Chain: (", length(x$chain), "labels )\n")
    print(x$chain)
}

print.mldCC <- function(x, ...) {
    cat("Classifier Chains Transformation Dataset\n\n")
    cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep = "")
    cat("Chain Order: ", x$chain.order, "\n\n", sep = "")
    cat("Dataset info:\n")
    cat(" ", ncol(x$data) - 1, "Predictive attributes\n")
    cat(" ", nrow(x$data), "Examples\n")
    cat("  ", round((sum(x$data[, ncol(x$data)] == 1)/nrow(x$data)) * 100, 1), "% of positive examples\n", sep = "")
}
