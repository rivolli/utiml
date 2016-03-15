#' Compute the confusion matrix for a multi-label prediction
#'
#' The multi-label confusion matrix is an object that contains the prediction,
#' the expected values and also a lot of pre-processed information related with
#' these data.
#
#' @param mdata A mldr dataset
#' @param mlresult A mlresult prediction
#'
#' @return A mlconfmat object that contains:
#' \describe{
#'    \item{Z}{The bipartition matrix prediction.}
#'    \item{Fx}{The score/probability matrix prediction.}
#'    \item{R}{The ranking matrix prediction.}
#'    \item{Y}{The expected matrix bipartition.}
#'    \item{TP}{The True Positive matrix values.}
#'    \item{FP}{The False Positive matrix values.}
#'    \item{TN}{The True Negative matrix values.}
#'    \item{FN}{The False Negative matrix values.}
#'    \item{Zi}{The total of positive predictions for each instance.}
#'    \item{Yi}{The total of positive expected for each instance.}
#'    \item{TPi}{The total of True Positive predictions for each instance.}
#'    \item{FPi}{The total of False Positive predictions for each instance.}
#'    \item{TNi}{The total of True Negative predictions for each instance.}
#'    \item{FNi}{The total False Negative predictions for each instance.}
#'    \item{Zl}{The total of positive predictions for each label.}
#'    \item{Yl}{The total of positive expected for each label.}
#'    \item{TPl}{The total of True Positive predictions for each label.}
#'    \item{FPl}{The total of False Positive predictions for each label.}
#'    \item{TNl}{The total of True Negative predictions for each label.}
#'    \item{FNl}{The total False Negative predictions for each label.}
#'  }
#' @export
#'
#' @examples
#' \dontrun{
#' prediction <- predict(br(toyml), toyml)
#'
#' mlconfmat <- multilabel_confusion_matrix(toyml, prediction)
#'
#' # Label with the most number of True Positive values
#' which.max(mlconfmat$TPl)
#'
#' # Number of wrong predictions for each label
#' errors <- mlconfmat$FPl + mlconfmat$FNl
#'
#' # Examples predict with all labels
#' which(mlconfmat$Zi == toyml$measures$num.labels)
#'
#' # You can join one or more mlconfmat
#' part1 <- create_subset(toyml, 1:50)
#' part2 <- create_subset(toyml, 51:100)
#' confmatp1 <- multilabel_confusion_matrix(part1, prediction[1:50, ])
#' confmatp2 <- multilabel_confusion_matrix(part2, prediction[51:100, ])
#' mlconfmat <- confmatp1 + confmatp2
#' }
multilabel_confusion_matrix <- function (mdata, mlresult) {
  expected <- mdata$dataset[, mdata$labels$index]
  bipartition <- as.bipartition(mlresult)
  scores <- as.probability(mlresult)
  ranking <- t(apply(1 - scores, 1, rank, ties.method = "first"))

  predict_and_expected <- expected & bipartition
  predict_and_nexpected <- !expected & bipartition
  npredict_and_nexpected <- !expected & !bipartition
  npredict_and_expected <- expected & !bipartition

  cm <- list(
    Z = bipartition,
    Y = expected,
    Fx = scores,
    R = ranking,
    TP = predict_and_expected,
    FP = predict_and_nexpected,
    TN = npredict_and_nexpected,
    FN = npredict_and_expected,
    Zi = rowSums(bipartition),
    Yi = rowSums(expected),
    Zl = colSums(bipartition),
    Yl = colSums(expected),
    TPi = rowSums(predict_and_expected),
    FPi = rowSums(predict_and_nexpected),
    TNi = rowSums(npredict_and_nexpected),
    FNi = rowSums(npredict_and_expected),
    TPl = colSums(predict_and_expected),
    FPl = colSums(predict_and_nexpected),
    TNl = colSums(npredict_and_nexpected),
    FNl = colSums(npredict_and_expected)
  )
  class(cm) <- "mlconfmat"
  cm
}

#' Join two multi-label confusion matrix
#'
#' @param mlcm1 A mlconfmat
#' @param mlcm2 Other mlconfmat
#'
#' @return mlconfmat
#' @export
`+.mlconfmat` <- function (mlcm1, mlcm2) {
  if (ncol(mlcm1$Z) != ncol(mlcm1$Z)) {
    stop("Different number of labels for each confusion matrix")
  }

  mlcm1$Z <- rbind(mlcm1$Z, mlcm2$Z)
  mlcm1$Y <- rbind(mlcm1$Y, mlcm2$Y)
  mlcm1$Fx <- rbind(mlcm1$Fx, mlcm2$Fx)
  mlcm1$R <- rbind(mlcm1$R, mlcm2$R)
  mlcm1$TP <- rbind(mlcm1$TP, mlcm2$TP)
  mlcm1$FP <- rbind(mlcm1$FP, mlcm2$FP)
  mlcm1$TN <- rbind(mlcm1$TN, mlcm2$TN)
  mlcm1$FN <- rbind(mlcm1$FN, mlcm2$FN)
  mlcm1$Zi <- c(mlcm1$Zi, mlcm2$Zi)
  mlcm1$Yi <- c(mlcm1$Yi, mlcm2$Yi)
  mlcm1$Zl <- mlcm1$Zl + mlcm2$Zl
  mlcm1$Yl <- mlcm1$Yl + mlcm2$Yl
  mlcm1$TPi <- c(mlcm1$TPi, mlcm2$TPi)
  mlcm1$FPi <- c(mlcm1$FPi, mlcm2$FPi)
  mlcm1$TNi <- c(mlcm1$TNi, mlcm2$TNi)
  mlcm1$FNi <- c(mlcm1$FNi, mlcm2$FNi)
  mlcm1$TPl <- mlcm1$TPl + mlcm2$TPl
  mlcm1$FPl <- mlcm1$FPl + mlcm2$FPl
  mlcm1$TNl <- mlcm1$TNl + mlcm2$TNl
  mlcm1$FNl <- mlcm1$FNl + mlcm2$FNl

  mlcm1
}

#' Evaluate multi-label predictions
#'
#' This method is used to evaluate multi-label predictions. You can create a
#' confusion matrix object or use directly the test dataset and the predictions.
#' You can also especify whiches measures do you desire use.
#'
#' @param object A mldr dataset or a mlconfmat confusion matrix
#' @param mlresult The prediction result (Optional, required only when the
#'  mldr is used).
#' @params measures The measures names to be computed. Call
#' \code{multilabel_measures()} to see the expected measures. You also can
#' use \code{"bipartition"}, \code{"ranking"}, \code{"label-based"},
#' \code{"example-based"}, \code{"macro-based"} and \code{"micro-based"} to
#' include a set of measures. (Default: "all").
#'
#' @return a vector with the expected measures
#' @references
#'  Madjarov, G., Kocev, D., Gjorgjevikj, D., & Džeroski, S. (2012). An
#'    extensive experimental comparison of methods for multi-label learning.
#'    Pattern Recognition, 45(9), 3084–3104.
#'  Zhang, M.-L., & Zhou, Z.-H. (2014). A Review on Multi-Label Learning
#'    Algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8),
#'    1819–1837.
#'  Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel Learning.
#'    ACM Comput. Surv., 47(3), 52:1–52:38.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' prediction <- predict(br(toyml), toyml)
#'
#' # Compute all measures
#' multilabel_evaluate(toyml, prediction)
#'
#' # Compute bipartition measures
#' multilabel_evaluate(toyml, prediction, "bipartition")
#'
#' # Compute multilples measures
#' multilabel_evaluate(toyml, prediction, c("accuracy", "F1", "macro-based"))
#'
#' # Compute the confusion matrix before the measures
#' cm <- multilabel_confusion_matrix(toyml, prediction)
#' multilabel_evaluate(cm)
#' multilabel_evaluate(cm, "example-based")
#' multilabel_evaluate(cm, c("hamming-loss", "subset-accuracy", "F1"))
#' }
multilabel_evaluate <- function(object, ...) {
  UseMethod("multilabel_evaluate")
}

#' @describeIn multilabel_evaluate Default S3 method
#' @export
multilabel_evaluate.mldr <- function (mdata, mlresult, measures = c("all"),
                                      ...) {
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (class(mlresult) != "mlresult") {
    stop("Second argument must be an mlresult object")
  }

  mlconfmat <- multilabel_confusion_matrix(mdata, mlresult)
  multilabel_evaluate.mlconfmat(mlconfmat, measures, ...)
}

#' @describeIn multilabel_evaluate Default S3 method
#' @export
multilabel_evaluate.mlconfmat <- function (mlconfmat, measures = c("all"),
                                           ...) {
  if (class(mlconfmat) != "mlconfmat") {
    stop("First argument must be an mlconfmat object")
  }

  default.methods <- list(
    'accuracy' = "mlmeasure_accuracy",
    'average-precision' = "mlmeasure_average_precision",
    'coverage' = "mlmeasure_coverage",
    'F1' = "mlmeasure_f1",
    'hamming-loss' = "mlmeasure_hamming_loss",
    'is-error' = "mlmeasure_is_error",
    'macro-accuracy' = "mlmeasure_macro_accuracy",
    'macro-F1' = "mlmeasure_macro_f1",
    'macro-precision' = "mlmeasure_macro_precision",
    'macro-recall' = "mlmeasure_macro_recall",
    'margin-loss' = "mlmeasure_margin_loss",
    'micro-accuracy' = "mlmeasure_micro_accuracy",
    'micro-F1' = "mlmeasure_micro_f1",
    'micro-precision' = "mlmeasure_micro_precision",
    'micro-recall' = "mlmeasure_micro_recall",
    'one-error' = "mlmeasure_one_error",
    'precision' = "mlmeasure_precision",
    'ranking-error' = "mlmeasure_ranking_error",
    'ranking-loss' = "mlmeasure_ranking_loss",
    'recall' = "mlmeasure_recall",
    'subset-accuracy' = "mlmeasure_subset_accuracy"
  )

  #Extra methods
  measures <- multilabel_measure_names(measures)
  midx <- measures %in% names(default.methods)
  extra.methods <- measures[!midx]

  if (!all(sapply(extra.methods, exists, mode="function"))) {
    stop(paste("Some methods are not found: ",
               extra.methods[!sapply(extra.methods, exists, mode="function")]))
  }
  names(extra.methods) <- extra.methods
  all.methods <- c(unlist(default.methods[measures[midx]]), extra.methods)

  extra = list(...)
  sapply(all.methods, function (mname) {
    params <- c(list(mlconfmat = mlconfmat), extra)
    do.call(mname, params)
  })
}

#' MULTILABEL MEASURES -------------------------------------------------------

#' Multi-label Accuracy Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_accuracy <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / rowSums(mlconfmat$Z | mlconfmat$Y), na.rm = TRUE) /
    nrow(mlconfmat$Y)
}

#' Multi-label Average Precision Measure
#' @param mlconfmat Confusion matrix
#' @references Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-
#' based system for text categorization. Machine Learning, 39(2), 135–168.
mlmeasure_average_precision <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i){
    rks <- mlconfmat$R[i, mlconfmat$Y[i,] == 1]
    sum(sapply(rks, function (r) sum(rks <= r) / r))
  }) / mlconfmat$Yi)
}

#' Multi-label Coverage Measure
#' @param mlconfmat Confusion matrix
#' @references Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-
#' based system for text categorization. Machine Learning, 39(2), 135–168.
mlmeasure_coverage <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i) {
    max(mlconfmat$R[i, mlconfmat$Y[i,] == 1]) - 1
  }))
}

#' Multi-label F1 Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_f1 <- function (mlconfmat, ...) {
  sum((2 * mlconfmat$TPi) / (mlconfmat$Zi + mlconfmat$Yi), na.rm = TRUE) /
    nrow(mlconfmat$Y)
}

#' Multi-label Hamming Loss Measure
#' @param mlconfmat Confusion matrix
#' @references Schapire, R. E., & Singer, Y. (1999). Improved boosting
#'  algorithm using confidence-rated predictions. Machine Learning, 297–336.
mlmeasure_hamming_loss <- function (mlconfmat, ...) {
  mean(apply(xor(mlconfmat$Z, mlconfmat$Y), 1, sum) / ncol(mlconfmat$Y))
}

#' Multi-label Is Error Measure
#' @param mlconfmat Confusion matrix
#' @param ranking The expected matrix ranking
#' @references Crammer, K., & Singer, Y. (2003). A Family of Additive Online
#'  Algorithms for Category Ranking. Journal of Machine Learning Research, 3(6),
#'  1025–1058.
mlmeasure_is_error <- function (mlconfmat, ranking, ...) {
  if (missing(ranking)) {
    stop("Argument ranking not informed for measure 'is-error'")
  }

  mean(rowSums(mlconfmat$R != ranking) != 0)
}

#' Multi-label Macro-Accuracy Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_macro_accuracy <- function (mlconfmat, ...) {
  mean(mlmeasure_binary_accuracy(mlconfmat$TPl, mlconfmat$FPl, mlconfmat$TNl,
                                  mlconfmat$FNl))
}

#' Multi-label Macro-F1 Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_macro_f1 <- function (mlconfmat, ...) {
  mean(mlmeasure_binary_f1(mlconfmat$TPl, mlconfmat$FPl,
                            mlconfmat$TNl, mlconfmat$FNl))
}

#' Multi-label Macro-Precision Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_macro_precision <- function (mlconfmat, ...) {
  mean(mlmeasure_binary_precision(mlconfmat$TPl, mlconfmat$FPl,
                                   mlconfmat$TNl, mlconfmat$FNl))
}

#' Multi-label Macro-Recall Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_macro_recall <- function (mlconfmat, ...) {
  mean(mlmeasure_binary_recall(mlconfmat$TPl, mlconfmat$FPl,
                                mlconfmat$TNl, mlconfmat$FNl))
}

#' Multi-label Margin Loss Measure
#' @param mlconfmat Confusion matrix
#' @references Loza Mencía, E., & Fürnkranz, J. (2010). Efficient Multilabel
#' Classification Algorithms for Large-Scale Problems in the Legal Domain.
#' In Semantic Processing of Legal Texts (pp. 192–215).
mlmeasure_margin_loss <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i){
    idxY <- mlconfmat$Y[i,] == 1
    max(0, max(mlconfmat$R[i, idxY], 0) -
          min(mlconfmat$R[i, !idxY], length(idxY)))
  }))
}

#' Multi-label Micro-Accuracy Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_micro_accuracy <- function (mlconfmat, ...) {
  mlmeasure_binary_accuracy(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                             sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

#' Multi-label Micro-F1 Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_micro_f1 <- function (mlconfmat, ...) {
  mlmeasure_binary_f1(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                       sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

#' Multi-label Micro-Precision Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_micro_precision <- function (mlconfmat, ...) {
  mlmeasure_binary_precision(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                              sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

#' Multi-label Micro-Recall Measure
#' @param mlconfmat Confusion matrix
#' @references Gibaja, E., & Ventura, S. (2015). A Tutorial on Multilabel
#'  Learning. ACM Comput. Surv., 47(3), 52:1–52:38.
mlmeasure_micro_recall <- function (mlconfmat, ...) {
  mlmeasure_binary_recall(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                           sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

#' Multi-label One Error Measure
#' @param mlconfmat Confusion matrix
#' @references Schapire, R. E., & Singer, Y. (2000). BoosTexter: A boosting-
#' based system for text categorization. Machine Learning, 39(2), 135–168.
mlmeasure_one_error <- function (mlconfmat, ...) {
  rowcol <- cbind(seq(nrow(mlconfmat$Y)), apply(mlconfmat$R, 1, which.min))
  mean(1 - mlconfmat$Y[rowcol])
}

#' Multi-label Precision Measure
#' @param mlconfmat Confusion matrix
#' @references Godbole, S., & Sarawagi, S. (2004). Discriminative Methods for
#' Multi-labeled Classification. In Proceedings of the 8th Pacific-Asia
#' Conference on Knowledge Discovery and Data Mining (PAKDD 2004) (pp. 22–30).
mlmeasure_precision <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / mlconfmat$Zi, na.rm = TRUE) / nrow(mlconfmat$Y)
}

#' Multi-label Ranking Error Measure
#' @param mlconfmat Confusion matrix
#' @references Park, S.-H., & Fürnkranz, J. (2008). Multi-Label Classification
#'  with Label Constraints. Proceedings of the ECML PKDD 2008 Workshop on
#'  Preference Learning (PL-08, Antwerp, Belgium), 157–171.
mlmeasure_ranking_error <- function (mlconfmat, ranking, ...) {
  if (missing(ranking)) {
    stop("Argument ranking not informed for measure 'is-error'")
  }
  #TODO
}

#' Multi-label Hamming Loss Measure
#' @param mlconfmat Confusion matrix
#' @references Schapire, R. E., & Singer, Y. (1999). Improved boosting
#'  algorithm using confidence-rated predictions. Machine Learning, 297–336.
mlmeasure_ranking_loss <- function (mlconfmat, ...) {
  weight <- 1 / (mlconfmat$Yi * (length(mlconfmat$Yl) - mlconfmat$Yi))
  weight <- ifelse(weight == Inf, 0, weight)
  E <- sapply(seq(nrow(mlconfmat$Y)), function (i) {
    idxY <- mlconfmat$Y[i,] == 1
    rkNY <- mlconfmat$R[i, !idxY]
    sum(unlist(lapply(mlconfmat$R[i, idxY], function (r) sum(r > rkNY))))
  })
  mean(weight * E)
}

#' Multi-label Recall Measure
#' @param mlconfmat Confusion matrix
#' @references Godbole, S., & Sarawagi, S. (2004). Discriminative Methods for
#' Multi-labeled Classification. In Proceedings of the 8th Pacific-Asia
#' Conference on Knowledge Discovery and Data Mining (PAKDD 2004) (pp. 22–30).
mlmeasure_recall <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / mlconfmat$Yi, na.rm = TRUE) / nrow(mlconfmat$Y)
}

#' Multi-label Subset Accuracy Measure
#' @param mlconfmat Confusion matrix
#' @references Zhu, S., Ji, X., Xu, W., & Gong, Y. (2005). Multilabelled
#'  Classification Using Maximum Entropy Method. In Proceedings of the 28th
#'  Annual International ACM SIGIR Conference on Research and Development in
#'  Information Retrieval (SIGIR’05) (pp. 274–281).
mlmeasure_subset_accuracy <- function (mlconfmat, ...) {
  mean(apply(mlconfmat$Z == mlconfmat$Y, 1, all))
}

#' BINARY MEASURES -----------------------------------------------------------

#' Compute the binary accuracy
#' @param TP The number of True Positive values
#' @param FP The number of False Positive values
#' @param TN The number of True Negative values
#' @param FN The number of False Negative values
#'
#' @return Accuracy value between 0 and 1
mlmeasure_binary_accuracy <- function (TP, FP, TN, FN) {
  (TP + TN) / (TP + FP + TN + FN)
}

#' Compute the binary precision
#' @param TP The number of True Positive values
#' @param FP The number of False Positive values
#' @param TN The number of True Negative values
#' @param FN The number of False Negative values
#'
#' @return Precision value between 0 and 1
mlmeasure_binary_precision <- function (TP, FP, TN, FN) {
  ifelse(TP + FP == 0, 0, TP / (TP + FP))
}

#' Compute the binary recall
#' @param TP The number of True Positive values
#' @param FP The number of False Positive values
#' @param TN The number of True Negative values
#' @param FN The number of False Negative values
#'
#' @return Recall value between 0 and 1
mlmeasure_binary_recall <- function (TP, FP, TN, FN) {
  ifelse(TP + FN == 0, 0, TP / (TP + FN))
}

#' Compute the binary F1 measure
#' @param TP The number of True Positive values
#' @param FP The number of False Positive values
#' @param TN The number of True Negative values
#' @param FN The number of False Negative values
#'
#' @return F1 measure value between 0 and 1
mlmeasure_binary_f1 <- function (TP, FP, TN, FN) {
  prec <-  mlmeasure_binary_precision(TP, FP, TN, FN)
  rec  <- mlmeasure_binary_recall(TP, FP, TN, FN)
  ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
}

#' MEASURES METHODS ----------------------------------------------------------

#' Return the tree with the measure names
#' @return list
multilabel_all_measures_names <- function (){
  list(
    'all' = c(
      "bipartition",
      "ranking"
    ),
    'bipartition' = c(
      "label-based",
      "example-based"
    ),
    'ranking' = c(
      "one-error",
      "coverage",
      "ranking-loss",
      "average-precision",
      "margin-loss"
    ),
    'label-based' = c(
      "micro-based",
      "macro-based"
    ),
    'example-based' = c(
      "subset-accuracy",
      "hamming-loss",
      "recall",
      "precision",
      "accuracy",
      "F1"
    ),
    'micro-based' = c(
      "micro-precision",
      "micro-recall",
      "micro-F1"
    ),
    'macro-based' = c(
      "macro-precision",
      "macro-recall",
      "macro-F1"
    )
  )
}

#' Return the name of measures
#'
#' @param measures The group of measures (Default: "all").
#'
#' @return array of character contained the measures names.
#'
#' @examples
#' multilabel_measure_names()
#' multilabel_measure_names("bipartition")
#' multilabel_measure_names(c("micro-based", "macro-based"))
multilabel_measure_names <- function (measures =  c("all")) {
  measures.names <- multilabel_all_measures_names()

  names <- unlist(lapply(measures, function (measure){
    if (is.null(measures.names[[measure]])) {
      measure
    } else {
      multilabel_measure_names(measures.names[[measure]])
    }
  }))
  unique(sort(names))
}

#' Return the name of all measures
#'
#' @return array of character contained the measures names.
#' @export
#'
#' @examples
#' multilabel_measures()
multilabel_measures <- function () {
  sort(c(multilabel_measure_names(), names(multilabel_all_measures_names())))
}

print.mlconfmat <- function (x) {
  cat("Multi-label Confusion Matrix\n\n")

  cat("Absolute Matrix:\n-------------------------------------\n")
  TP <- sum(x$TPi)
  FP <- sum(x$FPi)
  FN <- sum(x$FNi)
  TN <- sum(x$TNi)
  cm <-  matrix(c(TP, FN, TP + FN,
                  FP, TN, FP + TN,
                  TP + FP, FN + TN, TP + FP + FN + TN), ncol=3,
                dimnames = list(c("Prediction_1", "Predicion_0", "TOTAL"),
                                c("Expected_1", "Expected_0", "TOTAL")))
  print(cm)

  cat("\nProportinal Matrix:\n-------------------------------------\n")
  cm[1:2, 1:2] <- prop.table(cm[1:2, 1:2])
  cm[1:2, 3] <- apply(cm[1:2, 1:2], 1, sum)
  cm[3, ] <- apply(cm[1:2, ], 2, sum)
  print(cm)

  cm <- cbind(x$TPl, x$FPl, x$FNl, x$TNl)
  correct <- x$TPl + x$TNl
  wrong <- x$FPl + x$FNl

  cat("\nLabel Matrix\n-------------------------------------\n")
  cm <- cbind(
    cm, correct, wrong,
    round(prop.table(cm, 1), 2),
    round(prop.table(cbind(correct, wrong), 1), 2),
    round(apply(mlconfmat$R, 2, mean), 2),
    round(apply(mlconfmat$Fx, 2, mean), 2)
  )
  colnames(cm) <- c("TP", "FP", "FN", "TN", "Correct", "Wrong",
                    "%TP", "%FP", "%FN", "%TN", "%Correct", "%Wrong",
                    "MeanRanking", "MeanScore")
  print(as.data.frame(cm))
}
