multilabel_evaluate <- function(...) {
  UseMethod("multilabel_evaluate")
}

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

multilabel_evaluate.mlconfmat <- function (mlconfmat, measures = c("all"),
                                           ...) {
  if (class(mlconfmat) != "mlconfmat") {
    stop("First argument must be an mlconfmat object")
  }

  default.methods <- list(
    'accuracy' = "multilabel_accuracy",
    'average-precision' = "multilabel_average_precision",
    'coverage' = "multilabel_coverage",
    'F1' = "multilabel_f1",
    'hamming-loss' = "multilabel_hamming_loss",
    'is-error' = "multilabel_is_error",
    'macro-accuracy' = "multilabel_macro_accuracy",
    'macro-F1' = "multilabel_macro_f1",
    'macro-precision' = "multilabel_macro_precision",
    'macro-recall' = "multilabel_macro_recall",
    'margin-loss' = "multilabel_margin_loss",
    'micro-accuracy' = "multilabel_micro_accuracy",
    'micro-F1' = "multilabel_micro_f1",
    'micro-precision' = "multilabel_micro_precision",
    'micro-recall' = "multilabel_micro_recall",
    'one-error' = "multilabel_one_error",
    'precision' = "multilabel_precision",
    'ranking-error' = "multilabel_ranking_error",
    'ranking-loss' = "multilabel_ranking_loss",
    'recall' = "multilabel_recall",
    'subset-accuracy' = "multilabel_subset_accuracy"
  )

  #Extra methods
  measures <- multilabel_measure_names(measures)
  midx <- measures %in% names(default.methods)
  extra.methods <- measures[!midx]

  #if (!all(sapply(extra.methods, exists, mode="function"))) {
  #  stop(paste("Some methods are not found: ",
  #             extra.methods[!sapply(extra.methods, exists, mode="function")]))
  #}
  names(extra.methods) <- extra.methods
  all.methods <- c(unlist(default.methods[measures[midx]]), extra.methods)

  extra = list(...)
  sapply(all.methods, function (mname) {
    params <- c(list(mlconfmat = mlconfmat), extra)
    do.call(mname, params)
  })
}

multilabel_confusion_matrix <- function (mdata, mlresult) {
  expected <- mdata$dataset[, mdata$labels$index]
  bipartition <- as.bipartition(mlresult)
  ranking <- t(apply(1 - as.probability(mlresult), 1, rank,
                     ties.method = "first"))

  predict_and_expected <- expected & bipartition
  predict_and_nexpected <- !expected & bipartition
  npredict_and_nexpected <- !expected & !bipartition
  npredict_and_expected <- expected & !bipartition

  cm <- list(
    Z = bipartition,
    Y = expected,
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

`+.mlconfmat` <- function (mlcm1, mlcm2) {
  if (ncol(mlcm1$Z) != ncol(mlcm1$Z)) {
    stop("Different number of labels for each confusion matrix")
  }

  mlcm1$Z <- rbind(mlcm1$Z, mlcm2$Z)
  mlcm1$Y <- rbind(mlcm1$Y, mlcm2$Y)
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

#' MULTILABEL MEASURES -------------------------------------------------------
multilabel_accuracy <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / rowSums(mlconfmat$Z | mlconfmat$Y), na.rm = TRUE) /
    nrow(mlconfmat$Y)
}

multilabel_average_precision <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i){
    rks <- mlconfmat$R[i, mlconfmat$Y[i,] == 1]
    sum(sapply(rks, function (r) sum(rks <= r) / r))
  }) / mlconfmat$Yi)
}

multilabel_coverage <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i) {
    max(mlconfmat$R[i, mlconfmat$Y[i,] == 1]) - 1
  }))
}

multilabel_f1 <- function (mlconfmat, ...) {
  sum((2 * mlconfmat$TPi) / (mlconfmat$Zi + mlconfmat$Yi), na.rm = TRUE) /
    nrow(mlconfmat$Y)
}

multilabel_hamming_loss <- function (mlconfmat, ...) {
  mean(apply(xor(mlconfmat$Z, mlconfmat$Y), 1, sum) / ncol(mlconfmat$Y))
}

multilabel_is_error <- function (mlconfmat, ranking, ...) {
  mean(rowSums(mlconfmat$R != ranking) != 0)
}

multilabel_macro_accuracy <- function (mlconfmat, ...) {
  mean(multilabel_binary_accuracy(mlconfmat$TPl, mlconfmat$FPl, mlconfmat$TNl,
                                  mlconfmat$FNl))
}

multilabel_macro_f1 <- function (mlconfmat, ...) {
  mean(multilabel_binary_f1(mlconfmat$TPl, mlconfmat$FPl,
                            mlconfmat$TNl, mlconfmat$FNl))
}

multilabel_macro_precision <- function (mlconfmat, ...) {
  mean(multilabel_binary_precision(mlconfmat$TPl, mlconfmat$FPl,
                                   mlconfmat$TNl, mlconfmat$FNl))
}

multilabel_macro_recall <- function (mlconfmat, ...) {
  mean(multilabel_binary_recall(mlconfmat$TPl, mlconfmat$FPl,
                                mlconfmat$TNl, mlconfmat$FNl))
}

multilabel_margin_loss <- function (mlconfmat, ...) {
  mean(sapply(seq(nrow(mlconfmat$Y)), function (i){
    idxY <- mlconfmat$Y[i,] == 1
    max(0, max(mlconfmat$R[i, idxY], 0) -
          min(mlconfmat$R[i, !idxY], length(idxY)))
  }))
}

multilabel_micro_accuracy <- function (mlconfmat, ...) {
  multilabel_binary_accuracy(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                             sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

multilabel_micro_f1 <- function (mlconfmat, ...) {
  multilabel_binary_f1(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                       sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

multilabel_micro_precision <- function (mlconfmat, ...) {
  multilabel_binary_precision(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                              sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

multilabel_micro_recall <- function (mlconfmat, ...) {
  multilabel_binary_recall(sum(mlconfmat$TPl), sum(mlconfmat$FPl),
                           sum(mlconfmat$TNl), sum(mlconfmat$FNl))
}

multilabel_one_error <- function (mlconfmat, ...) {
  rowcol <- cbind(seq(nrow(mlconfmat$Y)), apply(mlconfmat$R, 1, which.min))
  mean(1 - mlconfmat$Y[rowcol])
}

multilabel_precision <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / mlconfmat$Zi, na.rm = TRUE) / nrow(mlconfmat$Y)
}

multilabel_ranking_error <- function (mlconfmat, ranking, ...) {
  #TODO mean(rowSums(abs(mlconfmat$R - ranking)))
}

multilabel_ranking_loss <- function (mlconfmat, ...) {
  weight <- 1 / (mlconfmat$Yi * (length(mlconfmat$Yl) - mlconfmat$Yi))
  weight <- ifelse(weight == Inf, 0, weight)
  E <- sapply(seq(nrow(mlconfmat$Y)), function (i) {
    idxY <- mlconfmat$Y[i,] == 1
    rkNY <- mlconfmat$R[i, !idxY]
    sum(unlist(lapply(mlconfmat$R[i, idxY], function (r) sum(r > rkNY))))
  })
  mean(weight * E)
}

multilabel_recall <- function (mlconfmat, ...) {
  sum(mlconfmat$TPi / mlconfmat$Yi, na.rm = TRUE) / nrow(mlconfmat$Y)
}

multilabel_subset_accuracy <- function (mlconfmat, ...) {
  mean(apply(mlconfmat$Z == mlconfmat$Y, 1, all))
}

#' BINARY MEASURES -----------------------------------------------------------
multilabel_binary_accuracy <- function (TP, FP, TN, FN) {
  (TP + TN) / (TP + FP + TN + FN)
}

multilabel_binary_precision <- function (TP, FP, TN, FN) {
  ifelse(TP + FP == 0, 0, TP / (TP + FP))
}

multilabel_binary_recall <- function (TP, FP, TN, FN) {
  ifelse(TP + FN == 0, 0, TP / (TP + FN))
}

multilabel_binary_f1 <- function (TP, FP, TN, FN) {
  prec <-  multilabel_binary_precision(TP, FP, TN, FN)
  rec  <- multilabel_binary_recall(TP, FP, TN, FN)
  ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
}

#' MEASURES METHODS ----------------------------------------------------------
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

multilabel_measures <- function () {
  measures.names <- multilabel_all_measures_names()
}

mlconfmat.print <- function (x) {
  #TODO
}
