#' Create a subset of a dataset
#'
#' @param mdata A \code{mldr} objetc dataset
#' @param rows A vector with the row indexes (instance indexes)
#' @param cols A vector with the column indexes (attribute indexes)
#'
#' @return A new mldr subset
#' @export
#'
#' @examples
#' #Return a emotion subset with the first 40 examples and only with two attributes
#' rows <- 1:40
#' cols <- c("Mean_Acc1298_Mean_Mem40_Centroid", "Mean_Acc1298_Mean_Mem40_MFCC_10")
#' mldr_subset(emotions, rows, cols)
mldr_subset <- function (mdata, rows, cols) {
  mldr_from_dataframe(
    cbind(mdata$dataset[rows, cols], mdata$dataset[rows, rownames(mdata$labels)]),
    labelIndices = (length(cols) + 1):(length(cols)+1):(length(cols)+mdata$measures$num.labels),
    name = mdata$name
  )
}

#' Create a random subset of a dataset
#'
#' @param mdata A \code{mldr} objetc dataset
#' @param num.rows The number of expected rows (instances)
#' @param num.cols The number of expected columns (attributes)
#'
#' @return A new mldr subset
#' @export
#'
#' @examples
#' #Return a emotion subset with 100 examples and 40 predictive attributes
#' mldr_random_subset(emotions, 100, 40)
mldr_random_subset <- function (mdata, num.rows, num.cols) {
  rows <- sample(mdata$measures$num.instances, num.rows)
  cols <- sample(mdata$attributesIndexes, num.cols)
  mldr_subset(mdata, rows, cols)
}

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
