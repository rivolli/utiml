#' Meta-BR or 2BR for multi-label Classification
#'
#' Create a Meta-BR (MBR) classifier to predic multi-label data. To this, two
#' round of Binary Relevance is executed, such that, the first step generates
#' new attributes to enrich the second prediction.
#'
#' This implementation use complete training set for both training and
#' prediction steps of 2BR. However, the \code{phi} parameter may be used to
#' remove labels with low correlations on the second step.
#'
#' @family Transformation methods
#' @family Stacking methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param folds The number of folds used in internal prediction. If this value
#'  is 1 all dataset will be used in the first prediction. (Default: 1)
#' @param phi A value between 0 and 1 to determine the correlation coefficient,
#'  The value 0 include all labels in the second phase and the 1 only the
#'  predicted label. (Default: 0)
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param predict.params A list of default arguments passed to the predictor
#'  method. (Default: \code{list()})
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{MBRmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'      \item{labels}{A vector with the label names.}
#'      \item{phi}{The value of \code{phi} parameter.}
#'      \item{correlation}{The matrix of label correlations used in combination
#'        with \code{phi} parameter to define the labels used in the second
#'        step. }
#'      \item{basemodel}{The BRModel used in the first iteration.}
#'      \item{models}{A list of models named by the label names used in the
#'        second iteration. }
#'   }
#' @references
#'  Tsoumakas, G., Dimou, A., Spyromitros, E., Mezaris, V., Kompatsiaris, I., &
#'    Vlahavas, I. (2009). Correlation-based pruning of stacked binary relevance
#'    models for multi-label learning. In Proceedings of the Workshop on
#'    Learning from Multi-Label Data (MLD’09) (pp. 22–30).
#'  Godbole, S., & Sarawagi, S. (2004). Discriminative Methods for Multi-labeled
#'    Classification. In Data Mining and Knowledge Discovery (pp. 1–26).
#' @seealso \code{\link{calculate_labels_correlation}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- mbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use 10 folds and different phi correlation with C4.5 classifier
#' model <- mbr(toyml, 'C4.5', 10, 0.2)
#'
#' # Set a specific parameter
#' model <- mbr(toyml, 'KNN', k=5)
#' }
mbr <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                folds = 1, phi = 0, ..., predict.params = list(),
                CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (folds < 1) {
    stop("The number of folds must be positive")
  }

  if (phi < 0 || phi > 1) {
    stop("The phi threshold must be between 0 and 1, inclusive")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # MBR Model class
  mbrmodel <- list(labels = rownames(mdata$labels), phi = phi,
                   call = match.call())

  # 1 Iteration - Base Level -------------------------------------------------
  mbrmodel$basemodel <- br(mdata, base.method, ..., CORES = CORES)
  if (folds == 1) {
    params <- list(object = mbrmodel$basemodel,
                   newdata = mdata$dataset[mdata$attributesIndexes],
                   probability = FALSE, CORES = CORES)
    base.preds <- do.call(predict, c(params, predict.params))
  }
  else {
    kf <- create_kfold_partition(mdata, folds, "iterative")
    base.preds <- do.call(rbind, lapply(seq(folds), function(f) {
      dataset <- partition_fold(kf, f)
      classifier <- br(dataset$train, base.method, ..., CORES = CORES)
      params <- list(object = classifier, newdata = dataset$test,
                     probability = FALSE, CORES = CORES)
      do.call(predict, c(params, predict.params))
    }))
  }

  # 2 Iteration - Meta level -------------------------------------------------
  corr <- mbrmodel$correlation <- calculate_labels_correlation(mdata)
  labels <- utiml_renames(mbrmodel$labels)
  mbrmodel$models <- utiml_lapply(labels, function (label) {
    new.col <- colnames(corr)[corr[label, ] > phi]
    new.data <- base.preds[, new.col, drop = FALSE]
    colnames(new.data) <- paste("extra", new.col, sep = ".")
    brdata  <- create_br_data(mdata, label, new.data)
    dataset <- prepare_br_data(brdata, "mldMBR", base.method,
                               new.features = new.col)
    create_br_model(dataset, ...)
  }, CORES)

  class(mbrmodel) <- "MBRmodel"
  mbrmodel
}

#' Predict Method for Meta-BR/2BR
#'
#' This function predicts values based upon a model trained by \code{mbr}.
#'
#' @param object Object of class '\code{MBRmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=mbr]{Meta-BR (MBR or 2BR)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- mbr(toyml)
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, toyml, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#' }
predict.MBRmodel <- function(object, newdata,
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "MBRmodel") {
    stop("First argument must be an MBRmodel object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }
  newdata <- utiml_newdata(newdata)

  # 1 Iteration - Base level -------------------------------------------------
  base.preds <- predict(object$basemodel, newdata, probability = FALSE, ...,
                        CORES = CORES)

  # 2 Iteration - Meta level -------------------------------------------------
  corr <- object$correlation
  labels <- utiml_renames(object$labels)
  predictions <- utiml_lapply(labels, function(labelname) {
    new.col <- colnames(corr)[corr[labelname, ] > object$phi]
    extra.col <- base.preds[, new.col, drop = FALSE]
    colnames(extra.col) <- paste("extra", new.col, sep = ".")
    predict_br_model(object$models[[labelname]], cbind(newdata, extra.col), ...)
  }, CORES)

  as.multilabelPrediction(predictions, probability)
}

#' Print MBR model
#' @export
print.MBRmodel <- function(x, ...) {
  cat("Classifier Meta-BR (also called 2BR)\n\nCall:\n")
  print(x$call)
  cat("\nPhi:", x$phi, "\n")
  cat("\nCorrelation Table Overview:\n")
  corr <- x$correlation
  diag(corr) <- NA
  tbl <- data.frame(
    min = apply(corr, 1, min, na.rm = TRUE),
    mean = apply(corr, 1, mean, na.rm = TRUE),
    median = apply(corr, 1, median, na.rm = TRUE),
    max = apply(corr, 1, max, na.rm = TRUE),
    extra = apply(x$correlation, 1, function(row) sum(row > x$phi))
  )
  print(tbl)
}
