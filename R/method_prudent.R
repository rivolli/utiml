#' PruDent classifier for multi-label Classification
#'
#' Create a PruDent classifier to predict multi-label data. To this, two
#' round of Binary Relevance is executed, such that, the first iteration
#' generates new attributes to enrich the second prediction.
#'
#' In the second phase only labels whose information gain is greater than a
#' specific phi value is added.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param phi A value between 0 and 1 to determine the information gain. The
#'  value 0 include all labels in the second phase and the 1 none.
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
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
#'    and Data Engineering, 27(9), 2480-2493.
#' @export
#'
#' @examples
#' model <- prudent(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' # Use different phi correlation with J48 classifier
#' model <- prudent(toyml, 'J48', 0.3)
#'
#' # Set a specific parameter
#' model <- prudent(toyml, 'KNN', k=5)
#' }
prudent <- function(mdata, base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                    phi = 0, ..., cores = getOption("utiml.cores", 1),
                    seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (phi < 0 || phi > 1) {
    stop("The phi threshold must be between 0 and 1, inclusive")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # PruDent Model class
  pdmodel <- list(
    labels = rownames(mdata$labels),
    call = match.call(),
    IG = utiml_labels_IG(mdata),
    phi = phi,

    # 1 Iteration - Base Level
    basemodel = br(mdata, base.algorithm, ..., cores = cores, seed = seed)
  )

  labeldata <- as.data.frame(mdata$dataset[mdata$labels$index])
  for (i in seq(ncol(labeldata))) {
    labeldata[, i] <- factor(labeldata[, i], levels=c(0, 1))
  }
  #base.preds <- as.matrix(mdata$dataset[mdata$labels$index])

  # 2 Iteration - Meta level
  IG <- matrix(pdmodel$IG >= phi,
               ncol = ncol(pdmodel$IG), dimnames = dimnames(pdmodel$IG))

  labels <- utiml_rename(pdmodel$labels)
  pdmodel$metamodels <- utiml_lapply(labels, function(label) {
    mmodel <- NULL
    extracols <- labeldata[, which(IG[label,]), drop = FALSE]
    if (ncol(extracols) > 0) {
      nmcol <- paste("extra", colnames(extracols), sep = ".")
      colnames(extracols) <- nmcol
      base <- utiml_create_binary_data(mdata, label, extracols)
      dataset <- utiml_prepare_data(base, "mldPruDent", mdata$name, "prudent",
                                    base.algorithm, new.features = nmcol)
      mmodel <- utiml_create_model(dataset, ...)
    }
    mmodel
  }, cores, seed)

  utiml_restore_seed()
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
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
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
#' # Passing a specif parameter for SVM predict algorithm
#' pred <- predict(model, toyml, na.action = na.fail)
#' }
predict.PruDentmodel <- function(object, newdata,
                                 probability = getOption("utiml.use.probs",
                                                         TRUE),
                                 ..., cores = getOption("utiml.cores", 1),
                                 seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "PruDentmodel") {
    stop("First argument must be an PruDentmodel object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()
  newdata <- utiml_newdata(newdata)

  # 1 Iteration - Base level
  base.scores <- predict.BRmodel(object$basemodel, newdata, TRUE, ...,
                                 cores=cores, seed=seed)
  base.preds <- as.bipartition(base.scores)
  labeldata <- as.data.frame(base.preds)
  for (i in seq(ncol(labeldata))) {
    labeldata[,i] <- factor(labeldata[,i], levels=c(0, 1))
  }

  # 2 Iteration - Meta level
  IG <- matrix(object$IG >= object$phi,
               ncol = ncol(object$IG), dimnames = dimnames(object$IG))

  labels <- utiml_rename(object$labels)
  predictions <- utiml_lapply(labels, function(labelname) {
    extracols <- labeldata[, which(IG[labelname,]), drop = FALSE]
    if (ncol(extracols) > 0) {
      colnames(extracols) <- paste("extra", colnames(extracols), sep = ".")
      utiml_predict_binary_model(object$metamodels[[labelname]],
                                 cbind(newdata, extracols), ...)
    }
    else {
      utiml_binary_prediction(base.preds[, labelname], base.scores[, labelname])
    }
  }, cores, seed)

  # Choosing the Final Classification
  for (i in seq(predictions)) {
    scores <- cbind(base = base.scores[,i],
                    meta = predictions[[i]]$probability)
    baseinst <- apply(abs(0.5 - scores), 1, which.max) == 1

    predictions[[i]]$probability[baseinst] <- base.scores[baseinst, i]
    predictions[[i]]$bipartition[baseinst] <- base.preds[baseinst, i]
  }

  utiml_restore_seed()
  utiml_predict(predictions, probability)
}

# Calculate the Information Gain for each pair of labels
#
# @param mdata A mldr dataset containing the label information.
# @return A matrix where the rows and columns represents the labels.
# @references
#  Alali, A., & Kubat, M. (2015). PruDent: A Pruned and Confident Stacking
#   Approach for Multi-Label Classification. IEEE Transactions on Knowledge
#   and Data Engineering, 27(9), 2480-2493.
utiml_labels_IG <- function (mdata) {
  entropy <- function (prob) {
    prob0 <- 1 - prob
    ifelse(prob == 0 || prob == 1,
           0, -prob * log2(prob) - prob0 * log2(prob0))
  }

  labelnames <- rownames(mdata$labels)
  classes <- mdata$dataset[,mdata$labels$index]
  q <- length(labelnames)
  ig <- matrix(nrow = q, ncol = q, dimnames = list(labelnames, labelnames))
  for (i in seq(q)) {
    for (j in seq(q)) {
      Hya <- entropy(mdata$labels$freq[i])
      hasJ <- classes[j] == 1
      Hyab <- mdata$labels$freq[j] *
        entropy(sum(classes[hasJ, i] == 1) / sum(hasJ)) +
        (1 - mdata$labels$freq[j]) *
        entropy(sum(classes[classes[j] == 0, i] == 1) / sum(!hasJ))

      ig[i,j] <- Hya  - Hyab
      ig[j,i] <- ig[i,j]
    }
    ig[i,i] <- 0
  }
  ig
}

#' Print PruDent model
#' @param x The prudent model
#' @param ... ignored
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
    median = apply(corr, 1, stats::median, na.rm = TRUE),
    max = apply(corr, 1, max, na.rm = TRUE),
    extra = apply(x$IG, 1, function(row) sum(row > x$phi))
  )
  print(tbl)
}
