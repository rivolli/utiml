#
# This file contains functions related with ensemble
# The most function are available public in the package
# The functions are sorted in alphabetical order
#

#' @title Average vote combination for ensemble prediction
#' @family Ensemble utilites
#' @description Compute the ensemble prediction using the average
#'  votes schema. The probabilities result is computed using the
#'  averaged values.
#'
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,1,1,1)),
#'  as.binaryPrediction(c(0.6,0.1,0.7,0.2)),
#'  as.binaryPrediction(c(0.8,0.1,0.4,0.3))
#' )
#' result <- utiml_ensemble_average_votes(predictions)
#' result$probability
#' # --> 0.8, 0.4, 0.7, 0.5
utiml_ensemble_average_votes <- function (predictions) {
  utiml_ensemble_compute_votes(predictions, mean)
}

#' @title Generic vote combination for ensemble prediction
#'
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#' @param method.name The name or the method that will be compute
#'  the results. This method receive a list of probability values
#'  and must return a single value probability.
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,1,1,1)),
#'  as.binaryPrediction(c(0.6,0.1,0.7,0.2)),
#'  as.binaryPrediction(c(0.8,0.1,0.4,0.3))
#' )
#'
#' # Compute the maximum vote combination
#' result <- utiml_ensemble_compute_votes(predictions, max)
utiml_ensemble_compute_votes <- function (predictions, method.name) {
  if (length(predictions) == 0)
    stop("Predictions can not be empty")
  as.binaryPrediction(apply(as.multilabelPrediction(predictions, TRUE), 1, method.name))
}

#' @title Majority vote combination for ensemble prediction
#' @family Ensemble utilites
#' @description Compute the ensemble prediction using the majority
#'  votes schema. The probabilities result is computed using only
#'  the majority instances. In others words, if a example is
#'  predicted as posivite, only the positive confidences are used to
#'  compute the averaged value.
#
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,1,1,1)),
#'  as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
#'  as.binaryPrediction(c(0.8,0.3,0.4,0.1))
#' )
#' result <- utiml_ensemble_majority_votes(predictions)
utiml_ensemble_majority_votes <- function (predictions) {
  if (length(predictions) == 0)
    stop("Predictions can not be empty")

  probabilities <- as.multilabelPrediction(predictions, TRUE)
  bipartitions <- attr(probabilities, "classes")

  votes <- apply(bipartitions, 1, mean)
  scores <- apply(probabilities, 1, mean)
  result <- scores

  #Compute the positive probabilities
  positive <- votes > 0.5 | (votes == 0.5 & scores >= 0.5)
  result[positive] <- unlist(lapply(which(positive), function (row){
    mean(probabilities[row, bipartitions[row,] == 1])
  }))

  #Compute the negative p
  result[!positive] <- unlist(lapply(which(!positive), function (row){
    mean(probabilities[row, bipartitions[row,] == 0])
  }))

  as.binaryPrediction(result)
}

#' @title Maximum vote combination for ensemble prediction
#' @family Ensemble utilites
#' @description Compute the ensemble prediction using the maximum
#'  votes schema. The probabilities result is computed using the
#'  maximum value.
#'
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,0.2,0.6,0.1)),
#'  as.binaryPrediction(c(0.6,0.1,0.7,0.2)),
#'  as.binaryPrediction(c(0.8,0.1,0.4,0.3))
#' )
#' result <- utiml_ensemble_maximum_votes(predictions)
#' result$probability
#' # --> 1, 0.2, 0.7, 0.3
utiml_ensemble_maximum_votes <- function (predictions) {
  utiml_ensemble_compute_votes(predictions, max)
}

#' @title Minimum vote combination for ensemble prediction
#' @family Ensemble utilites
#' @description Compute the ensemble prediction using the minimum
#'  votes schema. The probabilities result is computed using the
#'  minimum value.
#'
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,0.2,0.6,0.1)),
#'  as.binaryPrediction(c(0.6,0.1,0.7,0.2)),
#'  as.binaryPrediction(c(0.8,0.1,0.4,0.3))
#' )
#' result <- utiml_ensemble_minimum_votes(predictions)
#' result$probability
#' # --> 0.6, 0.1, 0.4, 0.1
utiml_ensemble_minimum_votes <- function (predictions) {
  utiml_ensemble_compute_votes(predictions, min)
}

#' @title Product vote combination for ensemble prediction
#' @family Ensemble utilites
#' @description Compute the ensemble prediction using the product
#'  votes schema. The probabilities result is computed using the
#'  product of all values.
#'
#' @param predictions A list of multi-label predictions
#'  (a \code{\link{mlprediction}} object).
#'
#' @return A multi-label prediction (a \code{\link{mlprediction}} object).
#' @seealso \code{link{mlprediction}}
#' @export
#'
#' @examples
#' predictions <- list(
#'  as.binaryPrediction(c(1,1,0.5,1)),
#'  as.binaryPrediction(c(0.9,0.1,0.5,0.2)),
#'  as.binaryPrediction(c(0.8,0.5,0.5,0.3))
#' )
#' result <- utiml_ensemble_product_votes(predictions)
#' result$probability
#' # --> 0.72, 0.05, 0.125, 0.06
utiml_ensemble_product_votes <- function (predictions) {
  utiml_ensemble_compute_votes(predictions, prod)
}

#' @title Compute the ensemble predictions based on some vote schema
#'
#' @param predictions A list of mlresult
#' @param vote.schema Define the way that ensemble must compute the predictions.
#'  The valid options are describe in \link{utiml_vote.schema_method}.
#' @param probability A logical value. If \code{TRUE} the predicted values are
#'  the score between 0 and 1, otherwise the values are bipartition 0 or 1.
#'
#' @return A new mlresult that final result
#' @export
#'
#' @examples
#' predictions <- list()
#' predictions$model1 <- prediction(brmodel1, testdata)
#' predictions$model2 <- prediction(brmodel2, testdata)
#' result <- utiml_compute_ensemble_predictions(predictions, "MAJ")
utiml_compute_multilabel_ensemble <- function (predictions, vote.schema, probability = TRUE) {
  method <- utiml_vote.schema_method(vote.schema)
  new.prediction <- list()
  for (label in colnames(predictions[[1]])) {
    lpred <- lapply(predictions, function (prediction){
      probs <- utiml_ifelse(attr(prediction, "type") == "probability",
                    prediction[,label], attr(prediction, "probs")[,label])

      as.binaryPrediction(probs)
    })
    new.prediction[[label]] <- utiml_compute_binary_ensemble(method, lpred)
  }

  as.multilabelPrediction(new.prediction, probability)
}

#' @title Compute the ensemble predictions for binary predictions
#'
#' @param method.name The method to compute the votes
#' @param binary.predictions A list of binary.prediction
#'
#' @return A new binary.result
#' @export
utiml_compute_binary_ensemble <- function (method.name, binary.predictions) {
  do.call(method.name, list(predictions = binary.predictions))
}

#' @title Define the method related with the vote schema
#'
#' @param vote.schema Define the way that ensemble must compute the predictions.
#'  The valid options are:
#' \describe{
#'  \code{'MAJ'}{Compute the averages of probabilities},
#'  \code{'MAX'}{Compute the votes scaled between 0 and \code{m} (number of interations)},
#'  \code{'MIN'}{Compute the proportion of votes, scale data between min and max of votes}
#'  \code{'AVG'}{Compute the proportion of votes, scale data between min and max of votes}
#'  \code{'PROD'}{Compute the proportion of votes, scale data between min and max of votes}
#' }
#' @return The method that will compute the votes
#' @export
utiml_vote.schema_method <- function (vote.schema) {
  votes <- list(
    MAJ = utiml_ensemble_majority_votes,
    MAX = utiml_ensemble_maximum_votes,
    MIN = utiml_ensemble_minimum_votes,
    AVG = utiml_ensemble_average_votes,
    PROD = utiml_ensemble_product_votes
  )
  votes[[vote.schema]]
}
