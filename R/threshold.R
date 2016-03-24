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

  result <- do.call(cbind, lapply(seq(ncol(prediction)), function(col) {
    as.integer(prediction[, col] >= threshold[col])
  }))
  dimnames(result) <- dimnames(prediction)

  # Avoid instances without labels
  for (row in which(apply(result, 1, sum) < 1)) {
    result[row, which.max(prediction[row, ])] <- 1
  }

  result
}

#' Maximum Cut Thresholding (MCut)
#'
#' The Maximum Cut (MCut) automatically determines a threshold for each instance
#' that selects a subset of labels with higher scores than others. This leads to
#' the selection of the middle of the interval defined by these two scores as
#' the threshold.
#'
#' @family threshold
#' @param prediction A matrix or mlresult.
#' @return A matrix or mlresult based as the type of prediction parameter.
#' @references
#' Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
#'  for Multi-label Classification. In 11th International Symposium, IDA 2012
#'  (pp. 172-183).
#' @export
#'
#' @examples
#' prediction <- matrix(runif(16), ncol = 4)
#' mcut_threshold(prediction)
mcut_threshold <- function (prediction) {
  UseMethod("mcut_threshold")
}

#' @describeIn mcut_threshold Maximum Cut Thresholding (MCut) method for matrix
#' @export
mcut_threshold.default <- function (prediction) {
  result <- apply(prediction, 1, function (row) {
    sorted.row <- sort(row, decreasing = T)
    difs <- unlist(lapply(seq(length(row)-1), function (i) {
      sorted.row[i] - sorted.row[i+1]
    }))
    t <- which.max(difs)
    mcut <- (sorted.row[t] + sorted.row[t+1]) / 2
    row <- ifelse(row > mcut, 1, 0)
    row
  })
  t(result)
}

#' @describeIn mcut_threshold Maximum Cut Thresholding (MCut) for mlresult
#' @export
mcut_threshold.mlresult <- function (prediction) {
  probs   <- as.probability(prediction)
  classes <- mcut_threshold.default(probs)
  get_multilabel_prediction(classes, probs, FALSE)
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
#' Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'  Comparative Study on Threshold Selection Methods. In First International
#'  Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#'
#' Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
#'  for Multi-label Classification. In 11th International Symposium, IDA 2012
#'  (pp. 172-183).
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
    result[row, which.max(prediction[row, ])] <- 1
  }

  result
}

#' @describeIn pcut_threshold Proportional Thresholding (PCut) for mlresult
#' @export
pcut_threshold.mlresult <- function (prediction, ratio) {
  probs   <- as.probability(prediction)
  classes <- pcut_threshold.default(probs, ratio)
  get_multilabel_prediction(classes, probs, FALSE)
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
    row[order(row, decreasing = TRUE)] <- values
    row
  })
  t(result)
}

#' @describeIn rcut_threshold Rank Cut (RCut) threshold method for mlresult
#' @export
rcut_threshold.mlresult <- function (prediction, k) {
  probs   <- as.probability(prediction)
  classes <- rcut_threshold.default(probs, k)
  get_multilabel_prediction(classes, probs, FALSE)
}

score_driven_threshold <- function () {
  #TODO
}

#' SCut Score-based method
#'
#' This is a label-wise method that adjusts the threshold for each label to
#' achieve a specific loss function using a validation set or cross validation.
#'
#' Different from the others threshold methods instead of return the bipartition
#' results it returs the threshold values for each label.
#'
#' @family threshold
#' @param prediction A matrix or mlresult.
#' @param expected The expected labels for the prediction. May be a matrix with
#'  the label values or a mldr object.
#' @param loss.function A loss function to be optmized. If you want to use your
#'  own error function see the notes and example. (Default: mse)
#' @param CORES The number of cores to parallelize the computation Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return A numeric vector with the threshold values for each label
#' @note The loss function is a R method that receive two lists, the expected
#'  values of the label and the predicted values, respectively. Positive values
#'  are represented by the 1 and the negative by the 0.
#' @references
#'  Fan, R.-E., & Lin, C.-J. (2007). A study on threshold selection for
#'   multi-label classification. Department of Computer Science, National
#'   Taiwan University.
#'
#'  Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'   Comparative Study on Threshold Selection Methods. In First International
#'   Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#' @export
#'
#' @examples
#' names <- list(1:10, c("a", "b", "c"))
#' prediction <- matrix(runif(30), ncol = 3, dimnames = names)
#' classes <- matrix(sample(0:1, 30, rep = TRUE), ncol = 3, dimnames = names)
#' thresholds <- scut_threshold(prediction, classes)
#' bipartition <- fixed_threshold(prediction, thresholds)
#'
#' \dontrun{
#' # Penalizes only FP predictions
#' mylossfunc <- function (real, predicted) {
#'    mean(predicted - real * predicted)
#' }
#' prediction <- predict(br(toyml), toyml)
#' scut_threshold(prediction, toyml, loss.function = mylossfunc, CORES = 5)
#' }
scut_threshold <- function (prediction, expected, loss.function = mse,
                            CORES = getOption("utiml.cores", 1)) {
  UseMethod("scut_threshold")
}

#' @describeIn scut_threshold Default scut_threshold
#' @export
scut_threshold.default <- function (prediction, expected, loss.function = mse,
                                    CORES = getOption("utiml.cores", 1)) {
  if (mode(loss.function) != "function") {
    stop("Invalid loss function")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  if (class(expected) == "mldr") {
    expected <- expected$dataset[expected$labels$index]
  }

  labels <- utiml_renames(colnames(prediction))
  thresholds <- utiml_lapply(labels, function (col) {
    scores <- prediction[, col]
    index <- order(scores)
    ones <- which(expected[index, col] == 1)
    difs <- c(Inf)
    for (i in seq(length(ones)-1)) {
      difs <- c(difs, ones[i+1] - ones[i])
    }

    evaluated.thresholds <- c()
    result <- c()
    for (i in ones[which(difs > 1)]) {
      thr <- scores[index[i]]
      res <- loss.function(expected[, col], ifelse(scores < thr, 0, 1))
      evaluated.thresholds <- c(evaluated.thresholds, thr)
      result <- c(result, res)
    }

    ifelse(length(ones) > 0,
           as.numeric(evaluated.thresholds[which.min(result)]),
           max(scores) + 0.0001) # All expected values are in the negative class
  }, CORES)

  unlist(thresholds)
}

#' @describeIn scut_threshold Mlresult scut_threshold
#' @export
scut_threshold.mlresult <- function (prediction, expected, loss.function = mse,
                                     CORES = getOption("utiml.cores", 1)) {
  probs   <- as.probability(prediction)
  scut_threshold.default(probs, expected, loss.function, CORES)
}

#' Subset Correction of a predicted result
#'
#' This method restrict a multi-label learner to predict only label combinations
#' whose existence is present in the (training) data. To this all labelsets
#' that are predicted but are not found on training data is replaced by the most
#' similar labelset.
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
#' @param base.threshold A numeric value between 0 and 1 to use as base to
#'  determine which values needs be reescaled to preserve the corrected
#'  labelsets. (Default: 0.5)
#' @return A new mlresult where all results are present in the training
#'  labelsets.
#' @note The original paper describes a method to create only bipartitions
#'  result, but we adapeted the method to change the scores. Based on the
#'  base.threshold value the scores higher than the threshold value, but must be
#'  lower are changed to respect this restriction.
#' @references
#'  Senge, R., Coz, J. J. del, & Hullermeier, E. (2013). Rectifying classifier
#'    chains for multi-label classification. In Workshop of Lernen, Wissen &
#'    Adaptivitat (LWA 2013) (pp. 162-169). Bamberg, Germany.
#' @export
#'
#' @examples
#' \dontrun{
#' prediction <- predict(br(toyml), toyml)
#' subset_correction(prediction, toyml$dataset[toyml$labels$index])
#' }
subset_correction <- function(mlresult, train_y, base.threshold = 0.5) {
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

    max_index <- new.prediction[r, ] - row > base.threshold
    min_index <- new.prediction[r, ] - row <= -base.threshold

    indexes <- min_index | max_index
    max_v <- min(c(row[row > base.threshold & !indexes], base.threshold + 0.1))
    min_v <- max(c(row[row < base.threshold & !indexes], base.threshold - 0.1))

    # Normalize values
    new.probability[r, max_index] = row[max_index] * (max_v - base.threshold) +
      base.threshold
    new.probability[r, min_index] = row[min_index] * (base.threshold - min_v) +
      min_v
  }

  get_multilabel_prediction(new.prediction, new.probability,
                            is.probability(mlresult))
}
