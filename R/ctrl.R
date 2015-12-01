#' @title CTRL model for multi-label Classification
#' @family Transformation methods
#' @family Ensemble methods
#' @description Create a binary relevance with ConTRolled Label correlation
#'   exploitation (CTRL) model for multilabel classification.
#'
#'   CTRL employs a two-stage filtering procedure to exploit label correlations
#'   in a controlled manner. In the first stage, error-prone class labels are
#'   pruned from Y to generate the candidate label set for correlation exploitation.
#'   In the second stage, classification models are built for each class label by
#'   exploiting its closely-related labels in the candidate label set.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset.
#' @param base.method A string with the name of the base method.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'})
#' @param m The max number of Binary Relevance models used in the ensemble.
#'    (default: 5)
#' @param validation.size The size of validation set, used internally to
#'    prunes error-prone class labels. The value must be between 0.1 and 0.6.
#'    (default: 0.3)
#' @param validation.threshold Thresholding parameter determining whether any
#'    class label in Y is regarded as error-prone or not. (default: 0.3)
#' @param ... Others arguments passed to the base method for all subproblems
#' @param predict.params A list of default arguments passed to the predictor
#'  method. (default: \code{list()})
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{CTRLmodel} containing the set of fitted
#'   models, including: \describe{
#'   \item{rounds}{The value passed in the m parameter}
#'   \item{validation.size}{The value passed in the validation.size parameter}
#'   \item{validation.threshold}{The value passed in the validation.threshold
#'      parameter}
#'   \item{Y}{Name of labels less susceptible to error, according to the
#'      validation process}
#'   \item{R}{List of close-related labels related with Y obtained by using
#'      feature selection technique}
#'   \item{models}{A list of the generated models, for each label a list of
#'      models was built based on close-related labels.}
#' }
#'
#' @details Dependencies:
#'  The degree of label correlations are estimated via supervised feature
#'    selection techniques. Thus, this implementation use the \link{relief}
#'    method available in \pkg{FSelector} package.
#'
#' @references
#'  Li, Y., & Zhang, M. (2014). Enhancing Binary Relevance for Multi-label
#'    Learning with Controlled Label Correlations Exploitation. In 13th Pacific
#'    Rim International Conference on Artificial Intelligence (pp. 91–103).
#'    Gold Coast, Australia.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Binary Relevance
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Use SVM as base method
#' model <- ctrl(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Change default values and use 4 CORES
#' model <- ctrl(dataset$train, 'C4.5', m = 10, validation.size = 0.4, validation.threshold = 0.5, CORES = 4)
#' pred <- predict(model, dataset$test)
#'
#' # Set a parameters for all subproblems
#' model <- ctrl(dataset$train, 'KNN', k=5)
#' pred <- predict(model, dataset$test)
ctrl <- function(mdata, base.method = "SVM", m = 5, validation.size = 0.3, validation.threshold = 0.3, ..., predict.params = list(), CORES = 1) {
    # Validations
    if (!requireNamespace("FSelector", quietly = TRUE))
        stop("There are no installed package \"FSelector\" to use CTRL multi-label classifier")

    if (class(mdata) != "mldr")
        stop("First argument must be an mldr object")

    if (m <= 1)
        stop("The number of iterations (m) must be greater than 1")

    if (validation.size < 0.1 || validation.size > 0.6)
        stop("The validation size must be between 0.1 and 0.6")

    if (validation.threshold < 0 || validation.threshold > 1)
        stop("The validation size must be between 0 and 1")

    if (CORES < 1)
        stop("Cores must be a positive value")

    # BR Model class
    ctrlmodel <- list()
    ctrlmodel$rounds <- m
    ctrlmodel$validation.size <- validation.size
    ctrlmodel$validation.threshold <- validation.threshold

    # Step1 - Split validation data, train and evaluation using F1 measure (1-5)
    validation.set <- create_holdout_partition(mdata, c(1 - validation.size, validation.size), "iterative")
    validation.model <- br(validation.set[[1]], base.method = base.method, ..., CORES = CORES)
    params <- list(object = validation.model, newdata = validation.set[[2]], probability = FALSE, CORES = CORES)
    validation.prediction <- do.call(predict, c(params, predict.params))
    validation.result <- utiml_measure_labels(validation.set[[2]], validation.prediction, utiml_measure_recall)
    Yc <- names(which(validation.result >= validation.threshold))
    ctrlmodel$Y <- Yc

    # Step2 - Identify close-related labels within Yc using feature selection technique (6-10)
    classes <- mdata$dataset[mdata$labels$index][, Yc]
    Rj <- utiml_lapply(rownames(mdata$labels), function(labelname) {
        formula <- as.formula(paste("`", labelname, "` ~ .", sep = ""))
        Aj <- mdata$dataset[, mdata$labels$index, drop = FALSE][, unique(c(Yc, labelname)), drop = FALSE]
        if (ncol(Aj) > 1) {
            weights <- FSelector::relief(formula, Aj)
            FSelector::cutoff.k(weights, m)
        }
    }, CORES)
    names(Rj) <- rownames(mdata$labels)
    ctrlmodel$R <- Rj

    # Build models (11-17)
    D <- mdata$dataset[mdata$attributesIndexes]
    ctrlmodel$models <- utiml_lapply(rownames(mdata$labels), function(labelname) {
        Di <- transform_br_data(cbind(D, mdata$dataset[labelname]), "mldBR", base.method)
        fi <- list(create_br_model(Di, ...))
        for (k in Rj[[labelname]]) {
            Di <- transform_br_data(cbind(D, mdata$dataset[k], mdata$dataset[labelname]), "mldBR", base.method)
            fi <- c(fi, list(create_br_model(Di, ...)))
        }
        names(fi) <- c(labelname, Rj[[labelname]])
        fi
    }, CORES)
    names(ctrlmodel$models) <- rownames(mdata$labels)

    ctrlmodel$call <- match.call()
    class(ctrlmodel) <- "CTRLmodel"

    ctrlmodel
}


#' @title Predict Method for CTRL
#' @description This function predicts values based upon a model trained by
#'  \code{\link{ctrl}}.
#'
#' @param object Object of class '\code{CTRLmodel}', created by \code{\link{ctrl}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param vote.schema The way that the method will compute the binary predictions
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
#' @seealso \code{\link[=ctrl]{CTRL}}
#'
#' @export
#'
#' @examples
#' # Emotion multi-label dataset using Binary Relevance
#' dataset <- mldr_random_holdout(emotions, c(train=0.9, test=0.1))
#'
#' # Predict SVM scores
#' model <- ctrl(dataset$train)
#' pred <- predict(model, dataset$test)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, dataset$test, probability = FALSE, CORES = 6)
#'
#' # Using the Maximum vote schema
#' pred <- predict(model, dataset$test, vote.schema = 'max')
predict.CTRLmodel <- function(object, newdata, vote.schema = "maj", probability = TRUE, ..., CORES = 1) {
    # Validations
    if (class(object) != "CTRLmodel")
        stop("First argument must be an CTRLmodel object")

    if (CORES < 1)
        stop("Cores must be a positive value")

    newdata <- utiml_newdata(newdata)

    # Predict initial values
    initial.prediction <- utiml_lapply(object$models, function(models) {
      predict_br_model(models[[1]], newdata, ...)
    }, CORES)
    fjk <- as.matrix(as.multilabelPrediction(initial.prediction, FALSE))

    # Predict binary ensemble values
    predictions <- utiml_lapply(names(object$models), function(labelname) {
        models <- object$models[[labelname]]
        preds <- list()
        for (labels in names(models)[-1]) preds[[labels]] <- predict_br_model(models[[labels]], cbind(newdata, fjk[, labels]), ...)

        if (length(preds) < 1) {
            initial.prediction[[labelname]]  #No models are found, only first prediction
        }
        else {
          compute_binary_ensemble_votes(preds, vote.schema)
        }
    }, CORES)

    names(predictions) <- names(object$models)
    as.multilabelPrediction(predictions, probability)
}

print.CTRLmodel <- function(x, ...) {
    cat("BR with ConTRolled Label correlation Model (CTRL)\n\nCall:\n")
    print(x$call)
    cat("\nDetails:")
    cat("\n ", x$rounds, "Iterations")
    cat("\n ", 1 - x$validation.size, "/", x$validation.size, "train/validation size")
    cat("\n ", x$validation.threshold, "Threshold value")
    cat("\n\nPruned Labels:", length(x$Y), "\n  ")
    cat(x$Y, sep = ", ")
}
