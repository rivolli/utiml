#' @title Create a predictive result object
#'
#' @description The transformation methods require a specific data format from base
#'  classifiers prediDefine the way that ensemble must compute the predictions.
#' The valid options are: \describe{
#'  \code{'score'}{Compute the averages of probabilities},
#'  \code{'majority'}{Compute the votes scaled between 0 and \code{m} (number of interations)},
#'  \code{'prop'}{Compute the proportion of votes, scale data between min and max of votes} }ction. If you implement a new base method then use this method
#'  to return the final result of your \code{mlpredict} method.
#'
#' @param probability A vector with probabilities predictions or with bipartitions
#'  prediction for a binary prediction.
#' @param threshold A numeric value betweenpredictions[["class1"]] <- mlpredict(model1, testdata) 0 and 1 to create the bipartitions.
#'
#' @return An object of type "\code{mlresult}" used by problem transformation
#'  methods that use binary classifiers. It has only two attributes:
#'  \code{bipartition} and \code{probability}, that respectively have the
#'  bipartition and probabilities results.
#' @export
#'
#' @examples
#' # This method is used Pearson product moment Correlation Coefficient (PCC)to implement a mlpredict based method
#' # In this example we create a random predict method
#' mlpredicti.random <- function (model, newdata, ...) {
#'    probs <- runif(nrow(newdata), 0, 1)
#'    as.resultPrediction(probs)
#' }
#'
#' # Define a different threshold for a specific subproblem use
#' ...Define the way that ensemble must compute the predictions.
#' The valid options are: \describe{
#'  \code{'score'}{Compute the averages of probabilities},
#'  \code{'majority'}{Compute the votes scaled between 0 and \code{m} (number of interations)},
#'  \code{'prop'}{Compute the proportion of votes, scale data between min and max of votes} }
#' result <- as.resultPrediction(probs, 0.6)
#' ...
as.resultPrediction <- function (probability, threshold = 0.5) {
  bipartition <- probability
  active <- bipartition >= threshold
  bipartition[active] <- 1
  bipartition[!active] <- 0

  #Making the prediction discriminative
  bipartition[which.max(probability)] <- 1

  res <- list(bipartition = bipartition, probability = probability)
  class(res) <- "mlresult"
  res
}

#' @title Pearson product moment Correlation Coefficient (PCC) for labels
#' @family labels correlation
#' @description Calculate all labels phi correlation coefficient.
#' This is a specialized version of the Pearson product moment
#' correlation coefficient for categorical variables with two
#' values, also called dichotomous variables.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label dataset
#'
#' @return A matrix with all labels phi correlation coefficient. The rows and
#' columns have the labels and the values are the phi value. The main diagonal
#' have the 1 value that represents the correlation of a label with itself.
#'
#' @references
#'  Tsoumakas, G., Dimou, A., Spyromitros, E., Mezaris, V., Kompatsiaris, I., &
#'    Vlahavas, I. (2009). Correlation-based pruning of stacked binary relevance models
#'    for multi-label learning. In Proceedings of the Workshop on Learning from
#'    Multi-Label Data (MLD’09) (pp. 22–30).
#'
#' @export
#'
#' @examples
#' library(utiml)
#' result <- labels_correlation_coefficient(emotions)
#'
#' # Get the phi coefficient between the labels "happy-pleased" and "quiet-still"
#' result["happy-pleased", "quiet-still"]
#'
#' # Get all coefficients of a specific label
#' result[1, ]
labels_correlation_coefficient <- function (mdata) {
  labelnames <- rownames(mdata$labels)
  classes <- mdata$dataset[,mdata$labels$index]
  q <- length(labelnames)
  cor <- matrix(nrow = q, ncol = q, dimnames = list(labelnames, labelnames))
  for (i in 1:q) {
    for (j in i:q) {
      confmat <- table(classes[,c(i, j)])
      A <- as.numeric(confmat["1", "1"])
      B <- as.numeric(confmat["1", "0"])
      C <- as.numeric(confmat["0", "1"])
      D <- as.numeric(confmat["0", "0"])
      cor[i,j] <- abs((A*D - B*C)/sqrt(as.numeric(A+B)*(C+D)*(A+C)*(B+D)))
      cor[j,i] <- cor[i,j]
    }
  }
  cor
}

binary_entropy <- function (prob) {
  res <- c(0, -prob * log2(prob) - (1 - prob) * log2(1 - prob))
  zero <- prob == 0 || prob == 1
  res[c(zero, !zero)]
}


labels_information_gain <- function (mdata) {
  labelnames <- rownames(mdata$labels)
  classes <- mdata$dataset[,mdata$labels$index]
  q <- length(labelnames)
  ig <- matrix(nrow = q, ncol = q, dimnames = list(labelnames, labelnames))
  for (i in 1:q) {
    for (j in i:q) {
      Hya <- binary_entropy(mdata$labels$freq[i])
      hasJ <- classes[j] == 1
      Hyab <- mdata$labels$freq[j] * binary_entropy(sum(classes[hasJ, i] == 1) / sum(hasJ)) +
        (1 - mdata$labels$freq[j]) * binary_entropy(sum(classes[classes[j] == 0, i] == 1) / sum(!hasJ))

      ig[i,j] <- Hya  - Hyab
      ig[j,i] <- ig[i,j]
    }
    ig[i,i] <- 0
  }
  ig
}
