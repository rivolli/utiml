#' @title Dependent Binary Relevance (DBR) for multi-label Classification
#' @family Transformation methods
#' @family Stacking methods
#' @description Create a DBR classifier to predic multi-label data.
#'  This is a simple approach that enables the binary classifiers to
#'  discover existing label dependency by themselves. The idea of DBR
#'  is exactly the same used in BR+ (the training method is the same,
#'  excepted by the argument \code{estimate.models} that indicate if the
#'  estimated models must be created).
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
#' @param estimate.models Logical value indicatind whether is necessary build
#'   Binary Relevance classifier for estimate process. The default implementaion
#'   use BR as estimators, however when other classifier is desirable then use
#'   the value \code{FALSE} to skip this process. (default: \code{TRUE}).
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param CORES he number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{DBRmodel} containing the set of fitted
#'   models, including: \describe{
#'    \item{estimation}{The BR model to estimate the values for the labels.
#'      Only when the \code{estimate.models = TRUE}.}
#'    \item{models}{A list of final models named by the label names.}
#' }
#'
#' @references
#'  Montañes, E., Senge, R., Barranquero, J., Ramón Quevedo, J., José Del Coz,
#'    J., & Hüllermeier, E. (2014). Dependent binary relevance models for
#'    multi-label classification. Pattern Recognition, 47(3), 1494–1508.
#' @seealso \code{\link{rdbr}}
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using DBR
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Use SVM as base method
#' model <- dbr(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- dbr(dataset$train, 'RF', CORES = 4)
#' pred <- predict(model, dataset$test)
dbr <- function(mdata, base.method = "SVM", estimate.models = TRUE, ..., CORES = 1) {
    # Validations
    if (class(mdata) != "mldr")
        stop("First argument must be an mldr object")

    if (CORES < 1)
        stop("Cores must be a positive value")

    # DBR Model class
    dbrmodel <- list()
    if (estimate.models)
        dbrmodel$estimation <- br(mdata, base.method, ..., CORES = CORES)

    basedata <- mdata$dataset[mdata$attributesIndexes]
    labeldata <- mdata$dataset[mdata$labels$index]
    datasets <- utiml_lapply(1:mdata$measures$num.labels, function(li) {
      prepare_br_data(cbind(basedata, labeldata[-li], labeldata[li]), "mldDBR", base.method)
    }, CORES)
    names(datasets) <- rownames(mdata$labels)
    dbrmodel$models <- utiml_lapply(datasets, create_br_model, CORES, ...)

    dbrmodel$call <- match.call()
    class(dbrmodel) <- "DBRmodel"

    dbrmodel
}

#' @title Predict Method for DBR
#' @description This function predicts values based upon a model trained by \code{dbr}.
#' In general this method is a restricted version of \code{\link{predict.BRPmodel}}
#' using the 'NU' strategy.
#'
#' As new feature is possible to use other multi-label classifier to predict the estimate
#' values of each label. To this use the prediction argument to inform a result of other
#' multi-label algorithm.
#'
#' @param object Object of class '\code{DBRmodel}', created by \code{\link{dbr}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param estimative A matrix containing the result of other multi-label classification
#'   algorithm. This table must contain only 0 or 1 predictions and it must be a
#'   multi-label prediction result.
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the prediction. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @references
#'  Montañes, E., Senge, R., Barranquero, J., Ramón Quevedo, J., José Del Coz,
#'    J., & Hüllermeier, E. (2014). Dependent binary relevance models for
#'    multi-label classification. Pattern Recognition, 47(3), 1494–1508.
#'
#' @seealso \code{\link[=dbr]{Dependent Binary Relevance (DBR)}}
#' @export
#'
#' @examples
#' # Emotion multi-label dataset using DBR
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Predict SVM scores
#' model <- dbr(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, dataset$test, na.action = na.fail)
#'
#' # Using other classifier (EBR) to made the labels estimatives
#' estimative <- predict(ebr(dataset$train), dataset$test, prob = FALSE)
#' model <- dbr(dataset$train, estimate.models = FALSE)
#' pred <- predict(model, dataset$test, estimative = estimative)
predict.DBRmodel <- function(object, newdata, estimative = NULL, probability = TRUE, ..., CORES = 1) {
    # Validations
    if (class(object) != "DBRmodel")
        stop("First argument must be an DBRmodel object")

    if (is.null(object$estimation) && is.null(estimative))
        stop("The model requires an estimative matrix")

    if (CORES < 1)
        stop("Cores must be a positive value")

    newdata <- utiml_newdata(newdata)
    if (is.null(estimative))
        estimative <- predict(object$estimation, newdata, ..., probability = FALSE, CORES = CORES)

    labels <- names(object$models)
    predictions <- utiml_lapply(1:length(labels), function(li) {
        predict_br_model(object$models[[li]], cbind(newdata, estimative[, -li]), ...)
    }, CORES)
    names(predictions) <- labels

    as.multilabelPrediction(predictions, probability)
}

print.DBRmodel <- function(x, ...) {
    cat("Classifier DBR\n\nCall:\n")
    print(x$call)
    cat("\n", length(x$models), "Models (labels):\n")
    print(names(x$models))
}
