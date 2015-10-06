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

utiml_ensemble_majority_votes <- function (predictions) {
  probabilities <- as.resultMLPrediction(predictions)
  bipartitions <- attr(probabilities, "probs")

  votes <- apply(bipartitions, 1, mean)
  scores <- apply(probabilities, 1, mean)

  #Compute the positive probabilities
  positive <- votes > 0.5 | (votes == 0.5 && scores > 0.5)
  result <- unlist(lapply(which(positive), function (row){
    mean(probabilities[row, bipartitions[row,] == 1])
  }))

  #Compute the negative probabilities
  result <- c(result, unlist(lapply(which(!positive), function (row){
    mean(probabilities[row, bipartitions[row,] == 0])
  })))

  as.binaryPrediction(result[names(votes)])
}
