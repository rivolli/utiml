#' @title Nested Stacking for multi-label Classification
#' @family Transformation methods
#' @description Create a Nested Stacking model for multilabel classification.
#'
#'   Nested Stacking is based on Classifier Chains transformation method to predict
#'   multi-label data. It differs from CC to predict the labels values in the
#'   training step and to regularize the output based on the labelsets available
#'   on training data.
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
#' @param predict.params A list of default arguments passed to the predict
#'  method. (default: \code{list()})
#'
#' @return An object of class \code{NSmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{chain}{A vector with the chain order}
#'    \item{labels}{A vector with the label names in expected order}
#'    \item{labelset}{The matrix containing only labels values}
#'    \item{models}{A list of models named by the label names.}
#'   }
#'
#' @references
#'  Senge, R., Coz, J. J. del, & Hüllermeier, E. (2013). Rectifying classifier
#'    chains for multi-label classification. In Workshop of Lernen, Wissen &
#'    Adaptivität (LWA 2013) (pp. 162–169). Bamberg, Germany.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Nested Stacking
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Use SVM as base method
#' model <- ns(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Use a specific chain with C4.5 classifier
#' mychain <- sample(rownames(dataset$train$labels))
#' model <- ns(dataset$train, 'C4.5', mychain)
#' pred <- predict(model, dataset$test)
#'
#' # Set a specific parameter
#' model <- ns(dataset$train, 'KNN', k=5)
#' pred <- predict(model, dataset$test)
ns <- function(mdata, base.method = "SVM", chain = c(), ..., predict.params = list()) {
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

    # NS Model class
    nsmodel <- list()
    nsmodel$labels <- labels
    nsmodel$chain <- chain
    nsmodel$models <- list()
    nsmodel$labelsets <- as.matrix(mdata$dataset[, mdata$labels$index])
    if (save.datasets)
        nsmodel$datasets <- list()

    basedata <- mdata$dataset[mdata$attributesIndexes]
    newattrs <- matrix(nrow = mdata$measures$num.instances, ncol = 0)
    for (labelIndex in 1:length(chain)) {
        label <- chain[labelIndex]

        # Create data
        dataset <- cbind(basedata, mdata$dataset[label])
        mldCC <- prepare_br_data(dataset, "mldCC", base.method, chain.order = labelIndex)

        # Call dynamic multilabel model with merged parameters
        model <- do.call(mltrain, c(list(dataset = mldCC), ...))

        result <- do.call(mlpredict, c(list(model = model, newdata = basedata), predict.params))
        basedata <- cbind(basedata, result$bipartition)
        names(basedata)[ncol(basedata)] <- label

        nsmodel$models[[label]] <- model
    }

    nsmodel$call <- match.call()
    class(nsmodel) <- "NSmodel"

    nsmodel
}

#' @title Predict Method for Nested Stacking
#' @description This function predicts values based upon a model trained by \code{ns}.
#'  The scores of the prediction was adapted once this method uses a correction of
#'  labelsets to predict only classes present on training data. To more information
#'  about this implementation see \code{\link{ns.subsetcorrection.score}}.
#'
#' @param object Object of class '\code{NSmodel}', created by \code{\link{ns}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @seealso \code{\link[=ns]{Nested Stacking (NS)}}
#'
#' @export
#'
#' @examples
#' # Emotion multi-label dataset using Nested Stacking
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Predict SVM scores
#' model <- ns(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, dataset$test, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, dataset$test, na.action = na.fail)
predict.NSmodel <- function(object, newdata, probability = TRUE, ...) {
    # Validations
    if (class(object) != "NSmodel")
        stop("First argument must be an NSmodel object")

    newdata <- utiml_newdata(newdata)
    predictions <- list()
    for (label in object$chain) {
        params <- c(list(model = object$models[[label]], newdata = newdata), ...)
        predictions[[label]] <- do.call(mlpredict, params)
        newdata <- cbind(newdata, predictions[[label]]$bipartition)
        names(newdata)[ncol(newdata)] <- label
    }

    result <- as.multilabelPrediction(predictions[object$labels], probability)
    compute_subset_correction(result, object$labelsets)
}

print.NSmodel <- function(x, ...) {
    cat("Nested Stacking Model\n\nCall:\n")
    print(x$call)
    cat("\n Chain: (", length(x$chain), "labels )\n")
    print(x$chain)
}
