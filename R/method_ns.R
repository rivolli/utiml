#' Nested Stacking for multi-label Classification
#'
#' Create a Nested Stacking model for multilabel classification.
#'
#' Nested Stacking is based on Classifier Chains transformation method to
#' predict multi-label data. It differs from CC to predict the labels values in
#' the training step and to regularize the output based on the labelsets
#' available on training data.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param chain A vector with the label names to define the chain order. If
#'   empty the chain is the default label sequence of the dataset. (Default:
#'   \code{list()})
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param predict.params A list of default arguments passed to the predict
#'  method. (default: \code{list()})
#' @return An object of class \code{NSmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{chain}{A vector with the chain order}
#'    \item{labels}{A vector with the label names in expected order}
#'    \item{labelset}{The matrix containing only labels values}
#'    \item{models}{A list of models named by the label names.}
#'   }
#' @references
#'  Senge, R., Coz, J. J. del, & Hüllermeier, E. (2013). Rectifying classifier
#'    chains for multi-label classification. In Workshop of Lernen, Wissen &
#'    Adaptivität (LWA 2013) (pp. 162–169). Bamberg, Germany.
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- ns(toyml)
#' pred <- predict(model, toyml)
#'
#' # Use a specific chain with J48 classifier
#' mychain <- sample(rownames(toyml$labels))
#' model <- ns(toyml, 'J48', mychain)
#'
#' # Set a specific parameter
#' model <- ns(toyml, 'KNN', k=5)
#' }
ns <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
               chain = c(), ..., predict.params = list()) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  labels <- rownames(mdata$labels)
  chain <- utiml_ifelse(length(chain) == 0, labels, chain)
  if (!utiml_is_equal_sets(chain, labels)) {
    stop("Invalid chain (all labels must be on the chain)")
  }

  # NS Model class
  nsmodel <- list(labels = labels, chain = chain, call = match.call(),
                  models = list(),
                  labelsets = as.matrix(mdata$dataset[, mdata$labels$index]))

  basedata <- mdata$dataset[mdata$attributesIndexes]
  newattrs <- matrix(nrow = mdata$measures$num.instances, ncol = 0)
  for (labelIndex in seq(length(chain))) {
    label <- chain[labelIndex]

    # Create data
    dataset <- cbind(basedata, mdata$dataset[label])
    mldCC <- prepare_br_data(dataset, "mldCC", base.method,
                             chain.order = labelIndex)

    # Call dynamic multilabel model with merged parameters
    model <- create_br_model(mldCC, ...)
    result <- do.call(predict_br_model,
                      c(list(model = model, newdata = basedata),
                        predict.params))

    basedata <- cbind(basedata, result$bipartition)
    names(basedata)[ncol(basedata)] <- label

    nsmodel$models[[label]] <- model
  }

  class(nsmodel) <- "NSmodel"
  nsmodel
}

#' Predict Method for Nested Stacking
#'
#' This function predicts values based upon a model trained by \code{ns}.
#' The scores of the prediction was adapted once this method uses a correction
#' of labelsets to predict only classes present on training data. To more
#' information about this implementation see \code{\link{subset_correction}}.
#'
#' @param object Object of class '\code{NSmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=ns]{Nested Stacking (NS)}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- ns(toyml)
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, toyml, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, toyml, na.action = na.fail)
#' }
predict.NSmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ...) {
  # Validations
  if (class(object) != "NSmodel") {
    stop("First argument must be an NSmodel object")
  }

  newdata <- utiml_newdata(newdata)
  predictions <- list()
  for (label in object$chain) {
    predictions[[label]] <- predict_br_model(object$models[[label]], newdata,
                                             ...)
    newdata <- cbind(newdata, predictions[[label]]$bipartition)
    names(newdata)[ncol(newdata)] <- label
  }

  result <- as.multilabelPrediction(predictions[object$labels], probability)
  subset_correction(result, object$labelsets)
}

#' Print NS model
#' @export
print.NSmodel <- function(x, ...) {
    cat("Nested Stacking Model\n\nCall:\n")
    print(x$call)
    cat("\n Chain: (", length(x$chain), "labels )\n")
    print(x$chain)
}