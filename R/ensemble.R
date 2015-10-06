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
#'  as.binaryPrediction(c(1  ,1  ,1  ,1  )),
#'  as.binaryPrediction(c(0.6,0.1,0.8,0.2)),
#'  as.binaryPrediction(c(0.8,0.3,0.4,0.1))
#' )
#' result <- utiml_ensemble_majority_votes(predictions)
utiml_ensemble_majority_votes <- function (predictions) {
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

  #Compute the negative probabilities
  result[!positive] <- unlist(lapply(which(!positive), function (row){
    mean(probabilities[row, bipartitions[row,] == 0])
  }))

  as.binaryPrediction(result)
}

#' @title Compute the ensemble predictions based on some vote schema
#'
#' @param predictions A list of matrix predictions
#' @param vote.schema Define the way that ensemble must compute the predictions.
#' The valid options are: \describe{
#'  \code{'score'}{Compute the averages of probabilities},
#'  \code{'majority'}{Compute the votes scaled between 0 and \code{m} (number of interations)},
#'  \code{'prop'}{Compute the proportion of votes, scale data between min and max of votes} }
#'
#' @return A list of mlresult as a result obtained from a multi-label transformation method
#' @export
#'
#' @examples
#' ...
#' predictions <- list()
#' predictions$model1 <- prediction(brmodel1, testdata)
#' predictions$model2 <- prediction(brmodel2, testdata)
#' result <- utiml_compute_ensemble_predictions(predictions, "majority")
#' ...
# utiml_compute_ensemble_predictions <- function (predictions, vote.schema) {
#   m <- length(predictions)
#   sumtable <- predictions[[1]]
#   for (i in 2:m)
#     sumtable <- sumtable + predictions[[i]]
#
#   avgtable <- if (vote.schema == "score")
#     sumtable / m
#   else if (vote.schema == "majority")
#     utiml_normalize(sumtable, m, 0)
#   else
#     utiml_normalize(sumtable) #proportionally
#
#   apply(avgtable, 2, as.binaryPrediction)
# }
