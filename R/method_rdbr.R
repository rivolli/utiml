#' @title Recursive Dependent Binary Relevance (RDBR) for multi-label Classification
#' @family Transformation methods
#' @description Create a RDBR classifier to predic multi-label data.
#'  This is a recursive approach that enables the binary classifiers to
#'  discover existing label dependency by themselves. The idea of RDBR
#'  is running DBR recursivelly until the results stabilization of the
#'  result.
#'  The train method is exactly the same of DBR the recursivity is in the
#'  predict method.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string with the name of base method. The same base method
#'   will be used for train all subproblems and the BR classifers
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'}).
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param estimate.models Logical value indicatind whether is necessary build
#'   Binary Relevance classifier for estimate process. The default implementaion
#'   use BR as estimators, however when other classifier is desirable then use
#'   the value \code{FALSE} to skip this process. (default: \code{TRUE}).
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: FALSE)
#' @param CORES he number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{DBRmodel} containing the set of fitted
#'   models, including: \describe{
#'    \item{estimation}{The BR model to estimate the values for the labels.
#'      Only when the \code{estimate.models = TRUE}.}
#'    \item{models}{A list of final models named by the label names.}
#'    \item{datasets}{A list with \code{estimation} and \code{final} datasets of
#'      type \code{mldDBR} named by the label names. Only when the
#'      \code{save.datasets = TRUE}.
#'    }
#' }
#'
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejão, F. M. (2014).
#'    Recursive Dependent Binary Relevance Model for Multi-label Classification.
#'    In Advances in Artificial Intelligence - IBERAMIA 2014 (pp. 206–217).
#'
#' @seealso \code{\link{dbr}}
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using RDBR
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- rdbr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- rdbr(emotions, 'RF', CORES = 4)
#' pred <- predict(model, testdata)
rdbr <- function(mdata, base.method = "SVM", ..., estimate.models = TRUE, save.datasets = FALSE, CORES = 1) {
    rdbrmodel <- dbr(mdata, base.method, ..., estimate.models = estimate.models, save.datasets = save.datasets, CORES = CORES)
    class(rdbrmodel) <- "RDBRmodel"
    rdbrmodel
}

#' @title Predict Method for RDBR
#' @description This function predicts values based upon a model trained by \code{rdbr}.
#' In general this method is a recursive version of \code{\link{predict.DBRmodel}}.
#'
#' Two versions of the update strategy of the estimated labels are implemented. The batch
#' re-estimates the labels only when a complete current label vector is available. The
#' stochastic uses re-estimated labels as soon as they become available. This second
#' does not support parallelize the prediction, however stabilizes earlier than batch
#' mode.
#'
#' @param object Object of class '\code{DBRmodel}', created by \code{\link{dbr}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param estimative A matrix containing the result of other multi-label classification
#'   algorithm. This table must contain only 0 or 1 predictions and it must be a
#'   multi-label prediction result.
#' @param max.iterations The maximum allowed iterations of the RDBR technique.
#'   (default: 5)
#' @param batch.mode Logical value to determine if use the batch re-estimation. If
#'   \code{FALSE} then use the stochastic re-estimation strategy. (default: \code{FALSE})
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param CORES The number of cores to parallelize the prediction. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @references
#'  Rauber, T. W., Mello, L. H., Rocha, V. F., Luchi, D., & Varejão, F. M. (2014).
#'    Recursive Dependent Binary Relevance Model for Multi-label Classification.
#'    In Advances in Artificial Intelligence - IBERAMIA 2014 (pp. 206–217).
#'
#' @seealso \code{\link[=rdbr]{Recursive Dependent Binary Relevance (DBR)}}
#' @export
#'
#' @examples
#' #' library(utiml)
#'
#' # Emotion multi-label dataset using DBR
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- rdbr(emotions)
#' pred <- predict(model, testdata)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
#'
#' # Use the batch mode and increase the max number of iteration to 10
#' pred <- predict(model, testdata, max.iterations = 10, batch.mode = TRUE)
#'
#' # Using other classifier (EBR) to made the labels estimatives
#' estimative <- predict(ebr(emotions), testdata, probability = FALSE)
#' model <- rdbr(emotions, estimate.models = FALSE)
#' pred <- predict(model, testdata, estimative = estimative)
predict.RDBRmodel <- function(object, newdata, ..., max.iterations = 5, batch.mode = FALSE, estimative = NULL, probability = TRUE, CORES = 1) {
    # Validations
    if (class(object) != "RDBRmodel")
        stop("First argument must be an RDDBRmodel object")

    if (is.null(object$estimation) && is.null(estimative))
        stop("The model requires an estimative matrix")

    if (max.iterations < 1)
        stop("The number of iteractions must be positive")

    if (CORES < 1)
        stop("Cores must be a positive value")

    newdata <- utiml_newdata(newdata)
    if (is.null(estimative))
        estimative <- predict(object$estimation, newdata, ..., probability = FALSE, CORES = CORES)

    labels <- names(object$models)
    if (batch.mode) {
        for (i in 1:max.iterations) {
            predictions <- utiml_lapply(1:length(labels), function(li) {
                predict_br_model(object$models[[li]], cbind(newdata, estimative[, -li]), ...)
            }, CORES)
            names(predictions) <- labels
            new.estimative <- do.call(cbind, lapply(predictions, function(lbl) lbl$bipartition))
            if (all(new.estimative == estimative))
                break
            estimative <- new.estimative
        }
    } else {
        for (i in 1:max.iterations) {
            old.estimative <- estimative
            predictions <- list()
            # the labels needs to be shuffled in each iteraction
            for (li in 1:length(labels)) {
                predictions[[li]] <- predict_br_model(object$models[[li]], cbind(newdata, estimative[, -li]), ...)
                estimative[, li] <- predictions[[li]]$bipartition
            }
            names(predictions) <- labels
            if (all(old.estimative == estimative))
                break
        }
    }

    as.multilabelPrediction(predictions, probability)
}

print.RDBRmodel <- function(x, ...) {
    cat("Classifier RDBR\n\nCall:\n")
    print(x$call)
    cat("\n", length(x$models), "Models (labels):\n")
    print(names(x$models))
}
