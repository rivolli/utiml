multilabel_evaluate <- function (mdata, mlresult, measures = c("all"), ...) {
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (class(mlresult) != "mlresult") {
    stop("Second argument must be an mlresult object")
  }

  default.methods <- list(
    'accuracy' = "multilabel_accuracy",
    'average-precision' = "multilabel_average_precision",
    'coverage' = "multilabel_coverage",
    'F1' = "multilabel_f1",
    'hamming-loss' = "multilabel_hamming_loss",
    'is-error' = "multilabel_is_error",
    'macro-F1' = "multilabel_macro_f1",
    'macro-precision' = "multilabel_macro_precision",
    'macro-recall' = "multilabel_macro_recall",
    'margin-loss' = "multilabel_margin_loss",
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
  if (!all(sapply(extra.methods, exists, mode="function"))) {
    stop(paste("Some methods are not found: ",
               extra.methods[!sapply(extra.methods, exists, mode="function")]))
  }
  names(extra.methods) <- extra.methods
  all.methods <- c(unlist(default.methods[measures[midx]]), extra.methods)

  mlconfmat <- multilabel_confusion_matrix(mdata, mlresult)
  extra = list(...)
  sapply(all.methods, function (mname) {
    params <- c(list(mlconfmat = mlconfmat), extra)
    do.call(mname, params)
  })
}

multilabel_confusion_matrix <- function (mdata, mlresult) {
  expected <- mdata$dataset[, mdata$labels$index]
  bipartition <- as.bipartition(mlresult)
  ranking <- t(apply(as.probability(mlresult), 1, function (row) {
    index <- order(row, decreasing = TRUE)
    row[index] <- seq(length(row))
    row
  }))

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
    max(0, max(mlconfmat$R[i, idxY]) - min(mlconfmat$R[i, !idxY]))
  }))
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
