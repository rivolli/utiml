#' BR+ or BRplus for multi-label Classification
#'
#' Create a BR+ classifier to predic multi-label data. This is a simple approach
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
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
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
#'    classification. Expert Systems with Applications, 39(2), 1647–1655.
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- brplus(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- brplus(toyml, 'RF', CORES = 4)
#' }
brplus <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                   ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }
  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # BRplus Model class
  brpmodel <- list(labels = rownames(mdata$labels), call = match.call())
  freq <- mdata$labels$freq
  names(freq) <- brpmodel$labels
  brpmodel$freq <- sort(freq)

  brpmodel$initial <- br(mdata, base.method, ..., CORES = CORES)

  labeldata <- mdata$dataset[mdata$labels$index]
  labels <- utiml_renames(seq(mdata$measures$num.labels), brpmodel$labels)
  brpmodel$models <- utiml_lapply(labels, function(li) {
    basedata <- create_br_data(mdata, brpmodel$labels[li], labeldata[-li])
    dataset <- prepare_br_data(basedata, "mldBRP", base.method)
    create_br_model(dataset, ...)
  }, CORES)

  class(brpmodel) <- "BRPmodel"
  brpmodel
}

#' Predict Method for BR+ (brplus)
#'
#' This function predicts values based upon a model trained by \code{brplus}.
#'
#' The strategies of estimate the values of the new features are separeted in
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
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @references
#'  Cherman, E. A., Metz, J., & Monard, M. C. (2012). Incorporating label
#'    dependency into the binary relevance framework for multi-label
#'    classification. Expert Systems with Applications, 39(2), 1647–1655.
#' @seealso \code{\link[=brplus]{BR+}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- brplus(toyml)
#' pred <- predict(model, dataset$test)
#'
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
                             ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "BRPmodel") {
    stop("First argument must be an BRPmodel object")
  }

  strategies <- c("Dyn", "Stat", "Ord", "NU")
  if (!strategy[1] %in% strategies) {
    stop(paste("Strategy value must be '",
               paste(strategies, collapse = "' or '"), "'", sep = ""))
  }

  labels <- object$labels
  if (strategy[1] == "Ord") {
    if (!utiml_is_equal_sets(order, labels)) {
      stop("Invalid order (all labels must be on the chain)")
    }
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  newdata <- utiml_newdata(newdata)

  if (strategy[1] == "NU") {
    initial.preds <- as.matrix(predict(object$initial, newdata,
                             probability = FALSE, ..., CORES = CORES))
    indices <- utiml_renames(seq(length(labels)), labels)
    predictions <- utiml_lapply(indices, function(li) {
      predict_br_model(object$models[[li]], cbind(newdata,
                                                  initial.preds[, -li]), ...)
    }, CORES)
  }
  else {
    initial.probs <- predict(object$initial, newdata,
                             probability = TRUE, ..., CORES = CORES)
    initial.preds <- as.matrix(as.bipartition(initial.probs))
    orders <- list(Dyn = names(sort(apply(initial.preds, 2, mean))),
                   Stat = names(object$freq),
                   Ord = order)

    predictions <- list()
    for (labelname in orders[[strategy[1]]]) {
      model <- object$models[[labelname]]
      new.columns <- initial.preds[, !labels %in% labelname, drop = FALSE]
      data <- cbind(newdata, new.columns)
      predictions[[labelname]] <- predict_br_model(model, data, ...)
      initial.preds[, labelname] <- predictions[[labelname]]$bipartition
    }
  }

  as.multilabelPrediction(predictions[labels], probability)
}

#' Print BRP model
#' @export
print.BRPmodel <- function(x, ...) {
  cat("Classifier BRplus (also called BR+)\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$models), "Models (labels):\n")
  print(names(x$models))
}
