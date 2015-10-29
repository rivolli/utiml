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

subsetcorretion <- function (mlresult, train_y, threshold = 0.5) {
  bipartition <- as.bipartition(mlresult)
  probability <- as.probability(mlresult)

  if (ncol(mlresult) != ncol(train_y))
    stop("The number of columns in the predicted result are different from the training data")

  #Bipartition correction
  labelsets <- unique(train_y)
  rownames(labelsets) <- apply(labelsets, 1, paste, collapse = "")

  order <- names(sort(table(apply(train_y, 1, paste, collapse = "")), decreasing = TRUE))
  labelsets <- labelsets[order,]

  new.prediction <- t(apply(bipartition, 1, function (y) {
    labelsets[names(which.min(apply(labelsets, 1, function (row) sum(row != y)))),]
  }))
  dimnames(new.prediction) <- dimnames(bipartition)

  #Probabilities correction
  new.probability <- probability
  for (r in 1:nrow(probability)) {
    row <- probability[r,]

    max_index <- new.prediction[r,] - row > threshold
    min_index <- new.prediction[r,] - row <= -threshold

    indexes <- min_index | max_index
    max_v <- min(c(row[row > threshold & !indexes], threshold + 0.1))
    min_v <- max(c(row[row < threshold & !indexes], threshold - 0.1))

    #Normalize values
    new.probability[r, max_index] = row[max_index] * (max_v - threshold) + threshold
    new.probability[r, min_index] = row[min_index] * (threshold - min_v) + min_v
  }

  multilabel.prediction(new.predicted, new.probability)
}

#' @title Subset Correction of a predicted result
#' @family threshold
#' @description This method restrict a multi-label learner prediction to only
#'  label combinations whose existence is testified by the (training) data. To
#'  this all labelsets that are predicted but are not found on training data is
#'  replaced by the most similar labelset.
#'
#'  If the most simillar is not unique, those label combinations with higher
#'  frequency in the training data are preferred. The Hamming loss is used
#'  to determine the difference between the labelsets.
#'
#' @param predicted_y A matrix of bipartitions predictions, where the columns
#'  represent the labels values and the rows the examples values.
#' @param train_y A matrix with all labels values of the training data.
#'
#' @return A new matrix of bipartitions where all results are present in the
#'  training labelsets.
#'
#' @references
#'  Senge, R., Coz, J. J. del, & Hüllermeier, E. (2013). Rectifying classifier
#'    chains for multi-label classification. In Workshop of Lernen, Wissen &
#'    Adaptivität (LWA 2013) (pp. 162–169). Bamberg, Germany.
#'
#' @seealso \code{\link[=ns.subsetcorrection.score]{Subset score correction}}
#' @export
#'
#' @examples
#' library(utiml)
#'
#' trainls <- matrix(c(1,1,1, 1,0,1, 1,0,1, 1,0,0, 1,1,1, 1,0,1), ncol = 3, byrow = TRUE)
#' colnames(trainls) <- c("c1", "c2", "c3")
#'
#' predict <- matrix(c(1,1,0, 0,0,1, 0,0,0), ncol = 3, byrow = TRUE)
#' colnames(predict) <- c("c1", "c2", "c3")
#'
#' ns.subsetcorrection(predict, trainls)
#' #       c2 c1 c3
#' # [1,]  1  1  1
#' # [2,]  1  0  1
#' # [3,]  1  0  0
ns.subsetcorrection <- function (predicted_y, train_y) {
  if (ncol(predicted_y) != ncol(train_y))
    stop("The number of columns in the predicted result are different from the training data")

  labelsets <- unique(train_y)
  rownames(labelsets) <- apply(labelsets, 1, paste, collapse = "")

  order <- names(sort(table(apply(train_y, 1, paste, collapse = "")), decreasing = TRUE))
  labelsets <- labelsets[order,]

  new.predicted <- t(apply(predicted_y, 1, function (y) {
    labelsets[names(which.min(apply(labelsets, 1, function (row) sum(row != y)))),]
  }))
  dimnames(new.predicted) <- dimnames(predicted_y)



  new.predicted
}

#' @title Subset Score Correction of a predicted result
#' @family threshold
#' @description This method is based on \code{\link{ns.subsetcorrection}},
#'  however, instead of using bipartition values, this use the probabilities
#'  scores.
#'
#'  As default method does not specify how to do this, this method changes the
#'  values of all labels scores that based on a threshold value does not in the
#'  correct set. The values are distributed between the superior and inferior
#'  nearest value from threshold, limiting it to 0.05 from threshold value.
#'
#' @param predicted_y A matrix of probabilites/scores predictions, where the
#'  columns represent the labels values and the rows the examples values.
#' @param train_y A matrix with all labels values of the training data.
#' @param threshold A numeric value between 0 and 1 to use as base to determine
#'  which values needs be reescaled to preserve the corrected labelsets.
#'  (default: 0.5)
#'
#' @return A new matrix of probabilities where all results are present in the
#'  training labelsets, based on the threshold value.
#'
#' @seealso \code{\link[=ns.subsetcorrection]{Subset correction}}
#' @export
#'
#' @examples
#' library(utiml)
#'
#' trainls <- matrix(c(1,1,1, 1,0,1, 1,0,1, 1,0,0, 1,1,1, 1,0,1), ncol = 3, byrow = TRUE)
#' colnames(trainls) <- c("c1", "c2", "c3")
#'
#' predict <- matrix(c(
#'    0.57, 0.84, 0.27,
#'    0.40, 0.49, 0.74,
#'    0.39, 0.62, 0.45
#'  ), ncol = 3, byrow = TRUE)
#' colnames(predict) <- c("c1", "c2", "c3")
#'
#' ns.subsetcorrection.score(predict, trainls)
#' #       c2    c1   c3
#' # [1,]  0.570 0.84 0.5189
#' # [2,]  0.540 0.49 0.7400
#' # [3,]  0.539 0.62 0.5450
ns.subsetcorrection.score <- function (predicted_y, train_y, threshold = 0.5) {
  if (ncol(predicted_y) != ncol(train_y))
    stop("The number of columns in the predicted result are different from the training data")

  new.predicted <- as.matrix(predicted_y)

  y <- ns.subsetcorrection(simple.threshold(new.predicted, threshold), train_y)
  for (r in 1:nrow(predicted_y)) {
    row <- predicted_y[r,]

    max_index <- y[r,] - row > threshold
    min_index <- y[r,] - row <= -threshold

    indexes <- min_index | max_index
    max_v <- min(c(row[row > threshold & !indexes], threshold + 0.1))
    min_v <- max(c(row[row < threshold & !indexes], threshold - 0.1))

    #Normalize values
    new.predicted[r, max_index] = row[max_index] * (max_v - threshold) + threshold
    new.predicted[r, min_index] = row[min_index] * (threshold - min_v) + min_v
  }

  new.predicted
}
