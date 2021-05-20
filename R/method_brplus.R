#' BR+ or BRplus for multi-label Classification
#'
#' Create a BR+ classifier to predict multi-label data. This is a simple approach
#' that enables the binary classifiers to discover existing label dependency by
#' themselves. The main idea of BR+ is to increment the feature space of the
#' binary classifiers to let them discover existing label dependency by
#' themselves.
#'
#' This implementation has different strategy to predict the final set of labels
#' for unlabeled examples, as proposed in original paper.
#'
#' @family Transformation methods
#' @family Stacking methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{BRPmodel} containing the set of fitted
#'  models, including:
#'  \describe{
#'    \item{freq}{The label frequencies to use with the 'Stat' strategy}
#'    \item{initial}{The BR model to predict the values for the labels to
#'      initial step}
#'    \item{models}{A list of final models named by the label names.}
#'  }
#' @references
#'  Cherman, E. A., Metz, J., & Monard, M. C. (2012). Incorporating label
#'    dependency into the binary relevance framework for multi-label
#'    classification. Expert Systems with Applications, 39(2), 1647-1655.
#' @export
#'
#' @examples
#' # Use SVM as base algorithm
#' model <- brplus(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \donttest{
#' # Use Random Forest as base algorithm and 2 cores
#' model <- brplus(toyml, 'RF', cores = 2, seed = 123)
#' }
brplus <- function(mdata,
                   base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                   ..., cores = getOption("utiml.cores", 1),
                   seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(mdata, "mldr")) {
    stop("First argument must be an mldr object")
  }
  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # BRplus Model class
  brpmodel <- list(labels = rownames(mdata$labels), call = match.call())
  freq <- mdata$labels$freq
  names(freq) <- brpmodel$labels
  brpmodel$freq <- sort(freq)

  brpmodel$initial <- br(mdata, base.algorithm, ..., cores = cores, seed = seed)

  labeldata <- as.data.frame(mdata$dataset[mdata$labels$index])
  for (i in seq(ncol(labeldata))) {
    labeldata[, i] <- factor(labeldata[, i], levels=c(0, 1))
  }
  labels <- utiml_rename(seq(mdata$measures$num.labels), brpmodel$labels)
  brpmodel$models <- utiml_lapply(labels, function(li) {
    basedata <- utiml_create_binary_data(mdata, brpmodel$labels[li],
                                         labeldata[-li])
    dataset <- utiml_prepare_data(basedata, "mldBRP", mdata$name, "brplus",
                                  base.algorithm)
    utiml_create_model(dataset, ...)
  }, cores, seed)

  utiml_restore_seed()
  class(brpmodel) <- "BRPmodel"
  brpmodel
}

#' Predict Method for BR+ (brplus)
#'
#' This function predicts values based upon a model trained by \code{brplus}.
#'
#' The strategies of estimate the values of the new features are separated in
#' two groups:
#' \describe{
#'  \item{No Update (\code{NU})}{This use the initial prediction of BR to all
#'   labels. This name is because no modification is made to the initial
#'   estimates of the augmented features during the prediction phase}
#'  \item{With Update}{This strategy update the initial prediction in that the
#'   final predict occurs. There are three possibilities to define the order of
#'   label sequences:
#'    \describe{
#'      \item{Specific order (\code{Ord})}{The order is define by the user,
#'       require a new argument  called \code{order}.}
#'      \item{Static order (\code{Stat})}{Use the frequency of single labels in
#'       the training set to define the sequence, where the least frequent
#'       labels are predicted first}
#'      \item{Dinamic order (\code{Dyn})}{Takes into account the confidence of
#'       the initial prediction for each independent single label, to define a
#'       sequence, where the labels predicted with less confidence are updated
#'       first.}
#'    }
#'  }
#' }
#'
#' @param object Object of class '\code{BRPmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param strategy The strategy prefix to determine how to estimate the values
#'  of the augmented features of unlabeled examples.
#'
#'  The possible values are: \code{'Dyn'}, \code{'Stat'}, \code{'Ord'} or
#'  \code{'NU'}. See the description for more details. (Default: \code{'Dyn'}).
#' @param order The label sequence used to update the initial labels results
#'  based on the final results. This argument is used only when the
#'  \code{strategy = 'Ord'} (Default: \code{list()})
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
#' @references
#'  Cherman, E. A., Metz, J., & Monard, M. C. (2012). Incorporating label
#'    dependency into the binary relevance framework for multi-label
#'    classification. Expert Systems with Applications, 39(2), 1647-1655.
#' @seealso \code{\link[=brplus]{BR+}}
#' @export
#'
#' @examples
#' # Predict SVM scores
#' model <- brplus(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \donttest{
#' # Predict SVM bipartitions and change the method to use No Update strategy
#' pred <- predict(model, toyml, strategy = 'NU', probability = FALSE)
#'
#' # Predict using a random sequence to update the labels
#' labels <- sample(rownames(dataset$train$labels))
#' pred <- predict(model, toyml, strategy = 'Ord', order = labels)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#' }
predict.BRPmodel <- function(object, newdata,
                             strategy = c("Dyn", "Stat", "Ord", "NU"),
                             order = list(),
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., cores = getOption("utiml.cores", 1),
                             seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "BRPmodel")) {
    stop("First argument must be an BRPmodel object")
  }

  strategy <- match.arg(strategy)

  labels <- object$labels
  if (strategy == "Ord") {
    if (!utiml_is_equal_sets(order, labels)) {
      stop("Invalid order (all labels must be on the chain)")
    }
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()
  if (!anyNA(seed)) {
    set.seed(seed)
  }

  newdata <- utiml_newdata(newdata)
  initial.preds <- predict.BRmodel(object$initial, newdata, probability=FALSE,
                                   ..., cores=cores, seed=seed)
  labeldata <- as.data.frame(as.bipartition(initial.preds))
  for (i in seq(ncol(labeldata))) {
    labeldata[, i] <- factor(labeldata[, i], levels=c(0, 1))
  }

  if (strategy == "NU") {
    indices <- utiml_rename(seq_along(labels), labels)
    predictions <- utiml_lapply(indices, function(li) {
      utiml_predict_binary_model(object$models[[li]],
                                 cbind(newdata, labeldata[, -li]), ...)
    }, cores, seed)
  }
  else {
    order <- switch (strategy,
      Dyn = names(sort(apply(as.probability(initial.preds), 2, mean))),
      Stat = names(object$freq),
      Ord = order
    )

    predictions <- list()
    for (labelname in order) {
      other.labels <- !labels %in% labelname
      model <- object$models[[labelname]]

      data <- cbind(newdata, labeldata[, other.labels, drop = FALSE])
      predictions[[labelname]] <- utiml_predict_binary_model(model, data, ...)
      labeldata[, labelname] <- factor(predictions[[labelname]]$bipartition,
                                       levels=c(0, 1))
    }
  }

  utiml_restore_seed()
  utiml_predict(predictions[labels], probability)
}

#' Print BRP model
#' @param x The brp model
#' @param ... ignored
#'
#' @return No return value, called for print model's detail
#'
#' @export
print.BRPmodel <- function(x, ...) {
  cat("Classifier BRplus (also called BR+)\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$models), "Models (labels):\n")
  print(names(x$models))
}
