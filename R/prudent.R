#' @title PruDent classifier for multi-label Classification
#' @family Transformation methods
#' @description Create a PruDent (MBR) classifier to predic multi-label
#'  data. To this, two round of Binary Relevance is executed, such that,
#'  the first iteraction generates new attributes to enrich the second
#'  prediction.
#'
#'  In the second phase only labels whose information gain is greater than
#'  a specific phi value is added.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string with the name of base method. The same base method
#'   will be used for train all subproblems in the two rounds. This is slightly
#'   different from the original paper that uses two distinct values from this
#'   parameter.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'})
#' @param phi A value between 0 and 1 to determine the information gain,
#'    The value 0 include all labels in the second phase and the none.
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: \code{FALSE})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{PruDentmodel} containing the set of fitted
#'   models, including: \describe{
#'      \item{labels}{A vector with the label names.}
#'      \item{phi}{The value of \code{phi} parameter.}
#'      \item{IG}{The matrix of Information Gain used in combination
#'        with \code{phi} parameter to define the labels used in the second step.
#'      }
#'      \item{basemodel}{The BRModel used in the first iteration.}
#'      \item{metamodels}{A list of models named by the label names used in the
#'        second iteration.
#'      }
#'      \item{datasets}{A list with \code{base} and \code{meta} datasets of
#'        type \code{mldBR} named by the label names. Only when the
#'        \code{save.datasets = TRUE}.
#'      }
#'  }
#'
#' @references
#'  Alali, A., & Kubat, M. (2015). PruDent: A Pruned and Confident Stacking
#'    Approach for Multi-Label Classification. IEEE Transactions on Knowledge
#'    and Data Engineering, 27(9), 2480–2493.
#' @seealso \code{\link{labels_information_gain}}
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Meta-BR
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- prudent(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use different phi correlation with C4.5 classifier
#' model <- prudent(emotions, 'C4.5', 0.3)
#' pred <- predict(model, testdata)
#'
#' # Set a specific parameter
#' model <- prudent(emotions, 'KNN', k=5)
#' pred <- predict(model, testdata)
prudent <- function(mdata, base.method = "SVM", phi = 0, ..., save.datasets = FALSE, CORES = 1) {
    # Validations
    if (class(mdata) != "mldr")
        stop("First argument must be an mldr object")

    if (phi < 0)
        stop("The phi threshold must be between 0 and 1, inclusive")

    if (CORES < 1)
        stop("Cores must be a positive value")

    # PruDent Model class
    pdmodel <- list()
    pdmodel$labels <- rownames(mdata$labels)
    pdmodel$phi <- phi

    # 1 Iteration - Base Level
    pdmodel$basemodel <- br(mdata, base.method, ..., save.datasets = TRUE, CORES = CORES)
    base.preds <- as.matrix(mdata$dataset[mdata$labels$index])

    # 2 Iteration - Meta level
    IG <- pdmodel$IG <- labels_information_gain(mdata)
    datasets <- utiml_lapply(pdmodel$basemodel$datasets, function(dataset) {
        extracolumns <- base.preds[, colnames(IG)[IG[dataset$labelname, ] > phi], drop = FALSE]
        if (ncol(extracolumns) > 0) {
            colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
            base <- cbind(dataset$data[-dataset$labelindex], extracolumns, dataset$data[dataset$labelindex])
            transform_br_data(base, "mldPruDent", base.method, new.features = colnames(extracolumns))
        }
    }, CORES)
    pdmodel$metamodels <- utiml_lapply(datasets[!unlist(lapply(datasets, is.null))], create_br_model, CORES, ...)

    if (save.datasets)
        pdmodel$datasets <- list(base = pdmodel$basemodel$datasets, meta = datasets)
    pdmodel$basemodel$datasets <- NULL

    pdmodel$call <- match.call()
    class(pdmodel) <- "PruDentmodel"

    pdmodel
}

#' @title Predict Method for PruDent
#' @description This function predicts values based upon a model trained by \code{prudent}.
#'
#' @param object Object of class '\code{PruDentmodel}', created by \code{\link{prudent}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
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
#' @seealso \code{\link[=prudent]{PruDent}}
#' @export
#'
#' @examples
#' #' library(utiml)
#'
#' # Emotion multi-label dataset using PruDent
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- prudent(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, testdata, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
predict.PruDentmodel <- function(object, newdata, ..., probability = TRUE, CORES = 1) {
    # Validations
    if (class(object) != "PruDentmodel")
        stop("First argument must be an PruDentmodel object")

    if (CORES < 1)
        stop("Cores must be a positive value")

    newdata <- utiml_newdata(newdata)

    # 1 Iteration - Base level
    base.scores <- predict(object$basemodel, newdata, ..., probability = TRUE, CORES = CORES)
    base.preds <- as.bipartition(base.scores, 0.5)

    # 2 Iteration - Meta level
    corr <- object$IG
    predictions <- utiml_lapply(object$labels, function(labelname) {
        extracolumns <- base.preds[, colnames(corr)[corr[labelname, ] > object$phi], drop = FALSE]
        if (ncol(extracolumns) > 0) {
            colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
            predict_br_model(object$metamodels[[labelname]], cbind(newdata, extracolumns), ...)
        } else {
            as.binaryPrediction(base.scores[, labelname])
        }
    }, CORES)
    names(predictions) <- object$labels

    original <- predictions
    # Choosing the Final Classification
    for (i in 1:length(predictions)) {
        indexes <- predictions[[i]]$probability >= 0.5

        # Positive scores
        predictions[[i]]$probability[indexes] <- unlist(lapply(which(indexes), function(j) {
            max(predictions[[i]]$probability[j], base.scores[j, i])
        }))

        # Negative scores
        predictions[[i]]$probability[!indexes] <- unlist(lapply(which(!indexes), function(j) {
            min(predictions[[i]]$probability[j], base.scores[j, i])
        }))

        predictions[[i]]$bipartition <- as.numeric(predictions[[i]]$probability >= 0.5)
        names(predictions[[i]]$bipartition) <- names(predictions[[i]]$probability)
    }

    as.multilabelPrediction(predictions, probability)
}

print.PruDentmodel <- function(x, ...) {
    cat("Classifier PruDent\n\nCall:\n")
    print(x$call)
    cat("\nMeta models:", length(x$metamodels), "\n")
    cat("\nPhi:", x$phi, "\n")
    cat("\nInformation Gain Table Overview:\n")
    corr <- x$IG
    diag(corr) <- NA
    tbl <- data.frame(min = apply(corr, 1, min, na.rm = TRUE), mean = apply(corr, 1, mean, na.rm = TRUE), median = apply(corr, 1, median, na.rm = TRUE), max = apply(corr,
        1, max, na.rm = TRUE), extra = apply(x$IG, 1, function(row) sum(row > x$phi)))
    print(tbl)
}

print.mldPruDent <- function(x, ...) {
    cat("PruDent Transformation Dataset\n\n")
    cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep = "")
    cat("Dataset info:\n")
    cat(" ", ncol(x$data) - 1 - length(x$new.features), "Predictive attributes\n")
    cat(" ", length(x$new.features), " meta features\n")
    cat(" ", nrow(x$data), "Examples\n")
    cat("  ", round((sum(x$data[, ncol(x$data)] == 1)/nrow(x$data)) * 100, 1), "% of positive examples\n", sep = "")
}
