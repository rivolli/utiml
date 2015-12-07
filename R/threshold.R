#' Apply a fixed threshold in the results
#'
#' Transfom a prediction matrix with scores/probabilities in a bipartion
#' prediction matrix. A global fixed threshold can be used of all labels or
#' different fixed thresholds, one for each label.
#'
#' @family threshold
#' @param prediction A matrix with scores/probabilities where the columns
#'    are the labels and the rows are the instances.
#' @param threshold A single value between 0 and 1 or a list with threshold
#'    values contained one value per label.
#' @return A matrix with bipartition results
#' @references
#'  Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'  Comparative Study on Threshold Selection Methods. In First International
#'  Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#' @export
#'
#' @examples
#' # Create a prediction matrix with scores
#' result <- matrix(
#'  data = rnorm(9, 0.5, 0.2),
#'  ncol = 3,
#'  dimnames = list(NULL, c('lbl1',  'lb2', 'lb3'))
#' )
#'
#' # Use 0.5 as threshold
#' fixed_threshold(result)
#'
#' # Use an threshold for each label
#' fixed_threshold(result, c(0.4, 0.6, 0.7))
fixed_threshold <- function(prediction, threshold = 0.5) {
  if (length(threshold) == 1) {
    threshold <- rep(threshold, ncol(prediction))
  }
  else if (length(threshold) != ncol(prediction)) {
    stop(paste("The threshold values must be a single value or the same",
               "number of labels"))
  }

  # Making the prediction discriminative
  for (row in seq(nrow(prediction))){
    prediction[row, which.max(prediction[row, ])] <- 1
  }

  result <- do.call(cbind, lapply(seq(ncol(prediction)), function(col) {
    as.integer(prediction[, col] >= threshold[col])
  }))
  dimnames(result) <- dimnames(prediction)

  result
}

#' Proportional Thresholding (PCut)
#'
#' Define the proportion of examples for each label will be positive.
#' The Proportion Cut (PCut) method can be a label-wise or global method that
#' calibrates the threshold(s) from the training data globally or per label.
#'
#' @family threshold
#' @param prediction A matrix or mlresult.
#' @param ratio A single value between 0 and 1 or a list with ratio values
#'  contained one value per label.
#' @return A matrix or mlresult based as the type of prediction parameter.
#' @references
#'  Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'  Comparative Study on Threshold Selection Methods. In First International
#'  Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#'
#'  Largeron, C., Moulin, C., & Géry, M. (2012). MCut: A Thresholding Strategy
#'  for Multi-label Classification. In 11th International Symposium, IDA 2012
#'  (pp. 172–183).
#' @export
#'
#' @examples
#' prediction <- matrix(runif(16), ncol = 4)
#' pcut_threshold(prediction, .45)
pcut_threshold <- function (prediction, ratio) {
  UseMethod("pcut_threshold")
}

#' @describeIn pcut_threshold Proportional Thresholding (PCut) method for matrix
#' @export
pcut_threshold.default <- function (prediction, ratio) {
  n <- nrow(prediction)
  num.elem <- ceiling(ratio * n)
  if (length(num.elem) == 1) {
    num.elem <- rep(num.elem, ncol(prediction))
    names(num.elem) <- colnames(prediction)
  }
  else if (length(num.elem) != ncol(prediction)) {
    stop(paste("The number of elements values must be a single value or the",
               "same number of labels"))
  }
  else if (is.null(names(num.elem))) {
    names(num.elem) <- colnames(prediction)
  }

  indexes <- utiml_renames(seq(ncol(prediction)), colnames(prediction))
  result <- do.call(cbind, lapply(indexes, function (ncol) {
    values <- c(rep(1, num.elem[ncol]), rep(0, n - num.elem[ncol]))
    prediction[order(prediction[,ncol], decreasing=TRUE), ncol] <- values
    prediction[,ncol]
  }))

  # Fill empty instance predictions
  empty.instances <- apply(result, 1, sum)
  for (row in which(empty.instances == 0)) {
    result[[which.max(prediction[row, ])]] <- 1
  }

  result
}

#' @describeIn pcut_threshold Proportional Thresholding (PCut) method for
#'  mlresult
#' @export
pcut_threshold.mlresult <- function (prediction, ratio) {
  probs   <- as.probability(prediction)
  classes <- pcut_threshold.default(probs, ratio)
  get_multilabel_prediction(classes, probs, is.probability(prediction))
}

score_driven_threshold <- function () {

}

#' Rank Cut (RCut) threshold method
#'
#' The Rank Cut (RCut) method is an instance-wise strategy, which outputs the k
#' labels with the highest scores for each instance at the deployment.
#'
#' @family threshold
#' @param prediction A matrix or mlresult.
#' @param k The number of elements that will be positive.
#' @return A matrix or mlresult based as the type of prediction parameter.
#' @references
#'  Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'  Comparative Study on Threshold Selection Methods. In First International
#'  Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#' @export
#'
#' @examples
#' prediction <- matrix(runif(16), ncol = 4)
#' rcut_threshold(prediction, 2)
rcut_threshold <- function (prediction, k) {
  UseMethod("rcut_threshold")
}

#' @describeIn rcut_threshold Rank Cut (RCut) threshold method for matrix
#' @export
rcut_threshold.default <- function (prediction, k) {
  values <- c(rep(1, k), rep(0, ncol(prediction) - k))
  result <- apply(prediction, 1, function (row) {
    row[order(row, decreasing=TRUE)] <- values
    row
  })
  t(result)
}

#' @describeIn rcut_threshold Rank Cut (RCut) threshold method for mlresult
#' @export
rcut_threshold.mlresult <- function (prediction, k) {
  probs   <- as.probability(prediction)
  classes <- rcut_threshold.default(probs, k)
  get_multilabel_prediction(classes, probs, is.probability(prediction))
}

scut_threshold <- function () {

}

#' Subset Correction of a predicted result
#'
#' This method restrict a multi-label learner prediction to only label
#' combinations whose existence is testified by the (training) data. To this all
#' labelsets that are predicted but are not found on training data is replaced
#' by the most similar labelset.
#'
#' If the most simillar is not unique, those label combinations with higher
#' frequency in the training data are preferred. The Hamming loss distance is
#' used to determine the difference between the labelsets.
#'
#' @family threshold
#' @param mlresult An object of mlresult that contain the scores and bipartition
#'  values.
#' @param train_y A matrix/data.frame with all labels values of the training
#'  dataset.
#' @param threshold A numeric value between 0 and 1 to use as base to determine
#'  which values needs be reescaled to preserve the corrected labelsets.
#'  (default: 0.5)
#' @return A new mlresult where all results are present in the training
#'  labelsets.
#' @note The original paper describes a method to create only bipartitions, but
#'  we adapeted the method to use also scores. Based on the threshold values the
#'  scores higher than the threshold value, but must be lower are changed to
#'  respect this restriction.
#'
#' @references
#'  Senge, R., Coz, J. J. del, & Hüllermeier, E. (2013). Rectifying classifier
#'    chains for multi-label classification. In Workshop of Lernen, Wissen &
#'    Adaptivität (LWA 2013) (pp. 162–169). Bamberg, Germany.
#' @export
subset_correction <- function(mlresult, train_y, threshold = 0.5) {
  bipartition <- as.bipartition(mlresult)
  probability <- as.probability(mlresult)

  if (ncol(mlresult) != ncol(train_y)) {
    stop("The number of columns in the predicted result are different from the
         training data")
  }

  # Bipartition correction
  labelsets <- as.matrix(unique(train_y))
  rownames(labelsets) <- apply(labelsets, 1, paste, collapse = "")

  order <- names(sort(table(apply(train_y, 1, paste, collapse = "")),
                      decreasing = TRUE))
  labelsets <- labelsets[order, ]

  new.prediction <- t(apply(bipartition, 1, function(y) {
    labelsets[names(which.min(apply(labelsets, 1, function(row) {
      sum(row != y)
    }))), ]
  }))

  # Probabilities correction
  new.probability <- probability
  for (r in seq(nrow(probability))) {
    row <- probability[r, ]

    max_index <- new.prediction[r, ] - row > threshold
    min_index <- new.prediction[r, ] - row <= -threshold

    indexes <- min_index | max_index
    max_v <- min(c(row[row > threshold & !indexes], threshold + 0.1))
    min_v <- max(c(row[row < threshold & !indexes], threshold - 0.1))

    # Normalize values
    new.probability[r, max_index] = row[max_index] * (max_v - threshold)
      + threshold
    new.probability[r, min_index] = row[min_index] * (threshold - min_v)
      + min_v
  }

  multilabel.prediction(new.prediction, new.probability,
                        is.probability(mlresult))
}
