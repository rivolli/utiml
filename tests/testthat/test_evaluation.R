context("Evaluation methods")
set.seed(1234)
parts <- create_holdout_partition(toyml)
result <- predict(br(parts$train, "SVM"), parts$test)

test_that("Multi-label confusion matrix", {
  mlconfmat <- multilabel_confusion_matrix(parts$test, result)

  expected <- parts$test$dataset[,parts$test$labels$index]
  predicted <- as.bipartition(result)

  expect_equal(dim(mlconfmat$Z), dim(result))
  expect_equal(dim(mlconfmat$Y), dim(result))
  expect_equal(dim(mlconfmat$R), dim(result))
  expect_equal(mlconfmat$Z, predicted)
  expect_equal(mlconfmat$Y, expected)

  expect_true(all(apply(mlconfmat$R, 1, function (row) row %in% 1:5)))

  expect_equal(dim(mlconfmat$TP), dim(result))
  expect_equal(dim(mlconfmat$TN), dim(result))
  expect_equal(dim(mlconfmat$FP), dim(result))
  expect_equal(dim(mlconfmat$FN), dim(result))

  expect_equal(length(mlconfmat$Zi), parts$test$measures$num.instances)
  expect_equal(length(mlconfmat$Yi), parts$test$measures$num.instances)
  expect_equal(length(mlconfmat$Zl), parts$test$measures$num.labels)
  expect_equal(length(mlconfmat$Yl), parts$test$measures$num.labels)
  expect_equal(mlconfmat$Yi, apply(expected, 1, sum))
  expect_equal(mlconfmat$Zi, apply(predicted, 1, sum))
  expect_equal(mlconfmat$Yl, apply(expected, 2, sum))
  expect_equal(mlconfmat$Zl, apply(predicted, 2, sum))

  totals <- mlconfmat$TPi + mlconfmat$TNi + mlconfmat$FPi + mlconfmat$FNi
  expect_true(all(totals == 5))

  expect_equal(mlconfmat$TPi, apply(expected & predicted, 1, sum))
  expect_equal(mlconfmat$TNi, apply(!expected & !predicted, 1, sum))
  expect_equal(mlconfmat$FPi, apply(!expected & predicted, 1, sum))
  expect_equal(mlconfmat$FNi, apply(expected & !predicted, 1, sum))

  expect_equal(mlconfmat$TPl, apply(expected & predicted, 2, sum))
  expect_equal(mlconfmat$TNl, apply(!expected & !predicted, 2, sum))
  expect_equal(mlconfmat$FPl, apply(!expected & predicted, 2, sum))
  expect_equal(mlconfmat$FNl, apply(expected & !predicted, 2, sum))
})

test_that("Bipartition measures", {
  labels <- as.matrix(parts$test$dataset[, parts$test$labels$index])
  expected <- parts$test$dataset[, parts$test$labels$index]

  #100% correct
  test.result <- get_multilabel_prediction(labels, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)
  expect_equal(multilabel_accuracy(mlconfmat), 1)
  expect_equal(multilabel_f1(mlconfmat), 1)
  expect_equal(multilabel_subset_accuracy(mlconfmat), 1)
  expect_equal(multilabel_precision(mlconfmat), 1)
  expect_equal(multilabel_recall(mlconfmat), 1)
  expect_equal(multilabel_hamming_loss(mlconfmat), 0)

  expect_equal(multilabel_macro_precision(mlconfmat), 1)
  expect_equal(multilabel_micro_precision(mlconfmat), 1)
  expect_equal(multilabel_macro_recall(mlconfmat), 1)
  expect_equal(multilabel_micro_recall(mlconfmat), 1)
  expect_equal(multilabel_macro_f1(mlconfmat), 1)
  expect_equal(multilabel_micro_f1(mlconfmat), 1)

  #100% incorrect
  for (i in seq(ncol(labels))) {
    pos <- labels[, i] == 1
    neg <- !pos
    labels[pos, i] <- 0
    labels[neg, i] <- 1
  }
  test.result <- get_multilabel_prediction(labels, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)
  expect_equal(multilabel_accuracy(mlconfmat), 0)
  expect_equal(multilabel_f1(mlconfmat), 0)
  expect_equal(multilabel_subset_accuracy(mlconfmat), 0)
  expect_equal(multilabel_precision(mlconfmat), 0)
  expect_equal(multilabel_recall(mlconfmat), 0)
  expect_equal(multilabel_hamming_loss(mlconfmat), 1)

  expect_equal(multilabel_macro_precision(mlconfmat), 0)
  expect_equal(multilabel_micro_precision(mlconfmat), 0)
  expect_equal(multilabel_macro_recall(mlconfmat), 0)
  expect_equal(multilabel_micro_recall(mlconfmat), 0)
  expect_equal(multilabel_macro_f1(mlconfmat), 0)
  expect_equal(multilabel_micro_f1(mlconfmat), 0)

  #Random
  set.seed(1234)
  for (i in seq(ncol(labels))) {
    labels[, i] <- utiml_normalize(rnorm(nrow(labels)))
  }
  labels <- fixed_threshold(labels, 0.5)
  test.result <- get_multilabel_prediction(labels, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)
  measures <- list(
    Accuracy = mean(rowSums(expected & labels) / rowSums(expected | labels)),
    FMeasure = mean(2 * rowSums(expected & labels) /
                      (rowSums(expected) + rowSums(labels))),
    SubsetAccuracy = mean(rowSums(expected == labels) == ncol(labels)),
    Precision = mean(rowSums(expected & labels) / rowSums(labels)),
    Recall = mean(rowSums(expected & labels) / rowSums(expected)),
    HammingLoss = mean(unlist(lapply(seq(nrow(labels)), function (i) {
      sum(xor(labels[i,], expected[i,])) / ncol(labels)
    }))),
    MacroPrecision = mean(
      colSums(labels == 1 & expected == 1) / colSums(labels == 1)
    ),
    MicroPrecision = sum(colSums(labels == 1 & expected == 1)) /
      sum(colSums(labels == 1)),
    MacroRecall = mean(
      colSums(labels == 1 & expected == 1) / colSums(expected == 1)
    ),
    MicroRecall = sum(colSums(labels == 1 & expected == 1)) /
      sum(colSums(expected == 1)),
    MacroFMeasure = (function (){
      prec <- colSums(labels == 1 & expected == 1) / colSums(labels == 1)
      rec <- colSums(labels == 1 & expected == 1) / colSums(expected == 1)
      mean(2 * prec * rec / (prec + rec))
    })(),
    MicroFMeasure = (function (){
      prec <- sum(colSums(labels == 1 & expected == 1)) /
        sum(colSums(labels == 1))
      rec <- sum(colSums(labels == 1 & expected == 1)) /
        sum(colSums(expected == 1))
      2 * prec * rec / (prec + rec)
    })()
  )
  expect_equal(multilabel_accuracy(mlconfmat), measures$Accuracy)
  expect_equal(multilabel_f1(mlconfmat), measures$FMeasure)
  expect_equal(multilabel_subset_accuracy(mlconfmat), measures$SubsetAccuracy)
  expect_equal(multilabel_precision(mlconfmat), measures$Precision)
  expect_equal(multilabel_recall(mlconfmat), measures$Recall)
  expect_equal(multilabel_hamming_loss(mlconfmat), measures$HammingLoss)

  expect_equal(multilabel_macro_precision(mlconfmat), measures$MacroPrecision)
  expect_equal(multilabel_micro_precision(mlconfmat), measures$MicroPrecision)
  expect_equal(multilabel_macro_recall(mlconfmat), measures$MacroRecall)
  expect_equal(multilabel_micro_recall(mlconfmat), measures$MicroRecall)
  expect_equal(multilabel_macro_f1(mlconfmat), measures$MacroFMeasure)
  expect_equal(multilabel_micro_f1(mlconfmat), measures$MicroFMeasure)
})

test_that("Ranking measures", {
  labels <- as.matrix(parts$test$dataset[, parts$test$labels$index])
  expected <- parts$test$dataset[, parts$test$labels$index]

  #100% correct
  test.result <- get_multilabel_prediction(labels, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)

  expect_equal(multilabel_one_error(mlconfmat), 0)
  expect_equal(multilabel_coverage(mlconfmat),
               parts$test$measures$cardinality - 1)
  expect_equal(multilabel_ranking_loss(mlconfmat), 0)
  expect_equal(multilabel_average_precision(mlconfmat), 1)
  expect_equal(multilabel_margin_loss(mlconfmat), 0)
  expect_equal(multilabel_is_error(mlconfmat, mlconfmat$R), 0)

  #100% incorrect
  for (i in seq(ncol(labels))) {
    pos <- labels[, i] == 1
    neg <- !pos
    labels[pos, i] <- 0
    labels[neg, i] <- 1
  }
  test.result <- get_multilabel_prediction(labels, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)

  expect_equal(multilabel_one_error(mlconfmat), 1)
  expect_equal(multilabel_coverage(mlconfmat), 4)
  expect_equal(multilabel_ranking_loss(mlconfmat), 1)
  #TODO study how to determine the worst case
  #expect_equal(multilabel_average_precision(mlconfmat), 0)
  expect_equal(multilabel_margin_loss(mlconfmat), 4)
  dif.rank <- mlconfmat$R[, 5:1]
  colnames(dif.rank) <- colnames(mlconfmat$R)
  expect_equal(multilabel_is_error(mlconfmat, dif.rank), 1)

  #Random
  set.seed(1234)
  for (i in seq(ncol(labels))) {
    labels[, i] <- utiml_normalize(rnorm(nrow(labels)))
  }
  bipartition <- fixed_threshold(labels, 0.5)
  test.result <- get_multilabel_prediction(bipartition, labels, TRUE)
  mlconfmat <- multilabel_confusion_matrix(parts$test, test.result)
})
