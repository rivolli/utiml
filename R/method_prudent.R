#' PruDent classifier for multi-label Classification
#'
#' Create a PruDent (MBR) classifier to predic multi-label data. To this, two
#' round of Binary Relevance is executed, such that, the first iteraction
#' generates new attributes to enrich the second prediction.
#'
#' In the second phase only labels whose information gain is greater than a
#' specific phi value is added.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param phi A value between 0 and 1 to determine the information gain. The
#'  value 0 include all labels in the second phase and the 1 none.
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{PruDentmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{phi}{The value of \code{phi} parameter.}
#'    \item{IG}{The matrix of Information Gain used in combination
#'      with \code{phi} parameter to define the labels used in the second step.
#'    }
#'    \item{basemodel}{The BRModel used in the first iteration.}
#'    \item{metamodels}{A list of models named by the label names used in the
#'      second iteration.
#'    }
#' }
#' @references
#'  Alali, A., & Kubat, M. (2015). PruDent: A Pruned and Confident Stacking
#'    Approach for Multi-Label Classification. IEEE Transactions on Knowledge
#'    and Data Engineering, 27(9), 2480–2493.
#' @seealso \code{\link{calculate_labels_information_gain}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- prudent(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use different phi correlation with J48 classifier
#' model <- prudent(toyml, 'J48', 0.3)
#'
#' # Set a specific parameter
#' model <- prudent(toyml, 'KNN', k=5)
#' }
prudent <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                    phi = 0, ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (phi < 0 || phi > 1) {
    stop("The phi threshold must be between 0 and 1, inclusive")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # PruDent Model class
  pdmodel <- list(labels = rownames(mdata$labels), call = match.call(),
                  IG = calculate_labels_information_gain(mdata), phi = phi)

  # 1 Iteration - Base Level
  pdmodel$basemodel <- br(mdata, base.method, ..., CORES = CORES)
  base.preds <- as.matrix(mdata$dataset[mdata$labels$index])

  # 2 Iteration - Meta level
  IG <- pdmodel$IG
  labels <- utiml_renames(pdmodel$labels)
  pdmodel$metamodels <- utiml_lapply(labels, function(label) {
    extracols <- base.preds[, colnames(IG)[IG[label, ] > phi], drop = FALSE]
    if (ncol(extracols) > 0) {
      colnames(extracols) <- paste("extra", colnames(extracols), sep = ".")
      base <- create_br_data(mdata, label, extracols)
      dataset <- prepare_br_data(base, "mldPruDent", base.method,
                                 new.features = colnames(extracols))
      create_br_model(dataset, ...)
    }
  }, CORES)

  class(pdmodel) <- "PruDentmodel"
  pdmodel
}

#' Predict Method for PruDent
#'
#' This function predicts values based upon a model trained by \code{prudent}.
#'
#' @param object Object of class '\code{PruDentmodel}'.
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
#' @seealso \code{\link[=prudent]{PruDent}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- prudent(toyml)
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, toyml, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#' }
predict.PruDentmodel <- function(object, newdata,
                                 probability = getOption("utiml.use.probs",
                                                         TRUE),
                                 ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "PruDentmodel") {
    stop("First argument must be an PruDentmodel object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  newdata <- utiml_newdata(newdata)

  # 1 Iteration - Base level
  base.scores <- predict(object$basemodel, newdata, TRUE, ..., CORES = CORES)
  base.preds <- as.bipartition(base.scores)

  # 2 Iteration - Meta level
  corr <- object$IG
  labels <- utiml_renames(object$labels)
  predictions <- utiml_lapply(labels, function(labelname) {
    corr.lbs <- colnames(corr)[corr[labelname, ] > object$phi]
    extracols <- base.preds[, corr.lbs, drop = FALSE]
    if (ncol(extracols) > 0) {
      colnames(extracols) <- paste("extra", colnames(extracols), sep = ".")
      predict_br_model(object$metamodels[[labelname]],
                       cbind(newdata, extracols), ...)
    }
    else {
      as.binaryPrediction(base.scores[, labelname])
    }
  }, CORES)

  original <- predictions
  # Choosing the Final Classification
  for (i in seq(predictions)) {
    indexes <- predictions[[i]]$bipartition == 1 | base.preds[, i] == 1

    # Positive scores
    predictions[[i]]$probability[indexes] <- unlist(lapply(which(indexes),
                                                           function(j) {
      max(predictions[[i]]$probability[j], base.scores[j, i])
    }))

    # Negative scores
    predictions[[i]]$probability[!indexes] <- unlist(lapply(which(!indexes),
                                                            function(j) {
      min(predictions[[i]]$probability[j], base.scores[j, i])
    }))

    predictions[[i]]$bipartition <- as.numeric(predictions[[i]]$probability >=
                                                 0.5)
    names(predictions[[i]]$bipartition) <- names(predictions[[i]]$probability)
  }

  as.multilabelPrediction(predictions, probability)
}

#' Print PruDent model
#' @export
print.PruDentmodel <- function(x, ...) {
  cat("Classifier PruDent\n\nCall:\n")
  print(x$call)
  cat("\nMeta models:", length(x$metamodels), "\n")
  cat("\nPhi:", x$phi, "\n")
  cat("\nInformation Gain Table Overview:\n")
  corr <- x$IG
  diag(corr) <- NA
  tbl <- data.frame(
    min = apply(corr, 1, min, na.rm = TRUE),
    mean = apply(corr, 1, mean, na.rm = TRUE),
    median = apply(corr, 1, median, na.rm = TRUE),
    max = apply(corr, 1, max, na.rm = TRUE),
    extra = apply(x$IG, 1, function(row) sum(row > x$phi))
  )
  print(tbl)
}
