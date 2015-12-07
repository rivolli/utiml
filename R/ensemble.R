#' Average vote combination for a single-label prediction
#'
#' Compute the prediction for a single-label using the average votes schema.
#' The probabilities result is computed using the averaged values.
#'
#' @family Ensemble utilites
#' @param predictions A list of binary predictions.
#' @return A new single-label prediction.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#' result <- average_ensemble_votes(predictions)
average_ensemble_votes <- function(predictions) {
  compute_ensemble_votes(predictions, mean)
}

#' Generic vote combination for ensemble prediction
#'
#' @param predictions A list of single-label predictions, usually with scores.
#' @param method The method that will be compute the results.
#'  This method receive a list of probability values and must return a single
#'  probability value.
#' @return A single-label prediction (a mlprediction object).
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#'
#' ## Return the max value for each example
#' result <- compute_ensemble_votes(predictions, max)
compute_ensemble_votes <- function(predictions, method) {
  if (length(predictions) == 0) {
    stop("Predictions can not be empty")
  }

  if (class(predictions[[1]]) == "binary.prediction") {
    table.prediction <- as.multilabelPrediction(predictions, TRUE)
  }
  else {
    table.prediction <- do.call(cbind, predictions)
  }

  as.binaryPrediction(apply(table.prediction, 1, method))
}

#' Majority vote combination for single-label prediction
#'
#' Compute the single-label prediction using the majority votes schema.
#' The probabilities result is computed using only the majority instances.
#' In others words, if a example is predicted as posivite, only the positive
#' confidences are used to compute the averaged value.
#'
#' @family Ensemble utilites
#' @param predictions A list of binary predictions.
#' @return A new single-label prediction.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#'
#' result <- majority_ensemble_votes(predictions)
majority_ensemble_votes <- function(predictions) {
  if (length(predictions) == 0) {
    stop("Predictions can not be empty")
  }

  if (class(predictions[[1]]) == "binary.prediction") {
    probabilities <- as.multilabelPrediction(predictions, TRUE)
    bipartitions <- as.bipartition(probabilities)
  }
  else {
    probabilities <- do.call(cbind, predictions)
    bipartitions <- compute_fixed_threshold(probabilities)
  }

  votes <- apply(bipartitions, 1, mean)
  scores <- apply(probabilities, 1, mean)
  result <- scores

  # Compute the positive probabilities
  positive <- votes > 0.5 | (votes == 0.5 & scores >= 0.5)
  result[positive] <- unlist(lapply(which(positive), function(row) {
    mean(probabilities[row, bipartitions[row, ] == 1])
  }))

  # Compute the negative p
  result[!positive] <- unlist(lapply(which(!positive), function(row) {
    mean(probabilities[row, bipartitions[row, ] == 0])
  }))

  as.binaryPrediction(result)
}

#' Maximum vote combination for single-label prediction
#'
#' Compute the single-label prediction using the maximum votes schema. The
#' probabilities result is computed using the maximum value.
#'
#' @family Ensemble utilites
#' @param predictions A list of binary predictions.
#' @return A new single-label prediction.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#'
#' result <- maximum_ensemble_votes(predictions)
maximum_ensemble_votes <- function(predictions) {
  compute_ensemble_votes(predictions, max)
}

#' Minimum vote combination for single-label prediction
#'
#' Compute the single-label prediction using the minimum votes schema. The
#' probabilities result is computed using the minimum value.
#'
#' @family Ensemble utilites
#' @param predictions A list of binary predictions.
#' @return A new single-label prediction.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#'
#' result <- minimum_ensemble_votes(predictions)
minimum_ensemble_votes <- function(predictions) {
  compute_ensemble_votes(predictions, min)
}

#' Product vote combination for single-label prediction
#'
#' Compute the ensemble prediction using the product votes schema. The
#' probabilities result is computed using the product of all values.
#'
#' @family Ensemble utilites
#' @param predictions A list of binary predictions.
#' @return A new single-label prediction.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1)[, "y1"],
#'  predict(model, toyml[1:10], k=3)[, "y1"],
#'  predict(model, toyml[1:10], k=5)[, "y1"]
#' )
#'
#' result <- product_ensemble_votes(predictions)
product_ensemble_votes <- function(predictions) {
  compute_ensemble_votes(predictions, prod)
}

#' Compute the multi-label ensemble predictions based on some vote schema
#'
#' @param predictions A list of multi-label predictions.
#' @param vote.schema Define the way that ensemble must compute the predictions.
#'  The default valid options are:
#'  \describe{
#'    \code{'avg'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'maj'}{Compute the averages of probabilities},
#'    \code{'max'}{Compute the votes scaled between 0 and \code{m}
#'      (number of interations)},
#'    \code{'min'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'prod'}{Compute the product of all votes for each instance}
#'  }
#'  If NULL then all predictions are returned.
#' @param probability A logical value. If \code{TRUE} the predicted values are
#'  the score between 0 and 1, otherwise the values are bipartition 0 or 1.
#' @return A new mlresult that final result
#' @note You can create your own vote schema, just create a method that receive
#'  a list with probabilities values and return a single value with the result.
#'  Your method must be like the methods mean, max and min.
#' @export
#'
#' @examples
#' model <- br(toyml, "KNN")
#' predictions <- list(
#'  predict(model, toyml[1:10], k=1),
#'  predict(model, toyml[1:10], k=3),
#'  predict(model, toyml[1:10], k=5)
#' )
#'
#' result <- compute_multilabel_ensemble_votes(predictions, "avg")
compute_multilabel_ensemble_votes <- function(predictions,
                                              vote.schema,
                                              probability = TRUE) {
  if (is.null(vote.schema)) {
    return(predictions)
  }

  prob.predictions <- lapply(predictions, as.probability)
  new.prediction <- list()
  for (label in colnames(predictions[[1]])) {
    bin.pred <- lapply(prob.predictions, function(prediction) {
      prediction[, label]
    })
    new.prediction[[label]] <- compute_binary_ensemble_votes(bin.pred,
                                                             vote.schema)
  }

  as.multilabelPrediction(new.prediction, probability)
}

#' Compute some vote schema for single-label predictions
#'
#' @param predictions A list of single-label predictions
#' @param vote.schema the vote schema name
#' @return A binary.prediction
compute_binary_ensemble_votes <- function (predictions, vote.schema) {
  method.name <- utiml_get_schema_method(vote.schema)
  custom_ensemble_votes <- function (predictions) {
    compute_ensemble_votes(predictions, vote.schema)
  }

  do.call(method.name, list(predictions = predictions))
}

#' Define the method related with the vote schema
#'
#' @param vote.schema Define the way that ensemble must compute the predictions.
#' @return The method that will compute the votes
utiml_get_schema_method <- function(vote.schema) {
  votes <- c(
    avg  = "average_ensemble_votes",
    maj  = "majority_ensemble_votes",
    max  = "maximum_ensemble_votes",
    min  = "minimum_ensemble_votes",
    prod = "product_ensemble_votes",
    vote.schema = "custom_ensemble_votes"
  )

  if (votes[[vote.schema]] == 'custom_ensemble_votes') {
    if (!exists(vote.schema, mode = "function")) {
      stop(paste("The compute ensemble method '", vote.schema,
                 "' is not a valid function", sep=''))
    }
  }

  votes[[vote.schema]]
}
