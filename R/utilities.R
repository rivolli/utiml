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
#' # This method is used to implement a mlpredict based method
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

#' @title Phi Correlation Coefficient
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
#' @export
#'
#' @examples
#' library(utiml)
#' result <- labelsPhiCorrelationCoefficient(emotions)
#'
#' # Get the phi coefficient between the labels "happy-pleased" and "quiet-still"
#' result["happy-pleased", "quiet-still"]
#'
#' # Get all coefficients of a specific label
#' result[1, ]
labelsPhiCorrelationCoefficient <- function (mdata) {
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

#' @title Apply a fixed threshold in the results
#' @family threshold
#' @description Transfom a prediction matrix with scores/probabilities in
#'  a bipartion prediction matrix. It is possible use a single value for
#'  all labels or specify a specific threshold for each label.
#'
#' @param prediction A matrix with scores/probabilities where the columns
#'    are the labels and the rows are the instances.
#' @param threshold A single value between 0 and 1 or a list with threshold
#'    values contained one value per label.
#'
#' @return A matrix with bipartition results
#' @export
#'
#' @examples
#' # Create a prediction matrix with scores
#' result <- matrix(
#'    data = rnorm(9, 0.5, 0.2),
#'    ncol = 3,
#'    dimnames = list(NULL, c("lbl1",  "lb2", "lb3"))
#' )
#'
#' # Use 0.5 as threshold
#' simple.threshold(result)
#'
#' # Use an threshold for each label
#' simple.threshold(result, c(0.4, 0.6, 0.7))
simple.threshold <- function (prediction, threshold = 0.5) {
  if (length(threshold) == 1)
    threshold <- rep(threshold, ncol(prediction))
  else if (length(threshold) != ncol(prediction))
    stop("The number of threshold must be the same number of labels or an unique value")

  #Making the prediction discriminative
  for (row in nrow(prediction))
    prediction[row, which.max(prediction[row,])] <- 1

  result <- do.call(cbind, lapply(1:ncol(prediction), function (col) {
    as.integer(prediction[,col] >= threshold[col])
  }))
  dimnames(result) <- dimnames(prediction)
  result
}
