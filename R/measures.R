#' @title Compute the precision measure for binary classification
#' @description  Precision (also called positive predictive value)
#'  is the fraction of retrieved instances that are relevant. In
#'  other words, is the number of true positives (items correctly
#'  labeled as belonging to the positive class) divided by the
#'  total number of elements labeled as belonging to the positive
#'  class.
#'
#' @param expected The expected list values (only 0 and 1)
#' @param predict The predicted list values (only 0 and 1)
#'
#' @return The precision in the 0..1 interval
#' @export
#'
#' @examples
#' utiml_measure_precision(c(1,1,1,0,0), c(1,1,0,0,0)) # 1
#' utiml_measure_precision(c(1,1,1,0,0), c(1,1,0,1,1)) # 0.5
#' utiml_measure_precision(c(1,1,1,0,0), c(0,0,0,1,1)) # 0
#'
#' # No predicted positive values the result is a NaN
#' utiml_measure_precision(c(1,1,1,0,0), c(0,0,0,0,0))
#'
#' # No expected positive values the result always is 0
#' utiml_measure_precision(c(0,0,0,0,0), c(0,1,1,0,0))
utiml_measure_precision <- function(expected, predict) {
    sum(predict & expected)/sum(predict)
}

utiml_measure_recall <- function(expected, predict) {
    sum(predict & expected)/sum(expected)
}

utiml_measure_f1 <- function(expected, predict) {
    Precision <- utiml_measure_precision(expected, predict)
    Recall <- utiml_measure_recall(expected, predict)
    (2 * Precision * Recall)/(Precision + Recall)
}

utiml_measure_labels <- function(mdata, predicted, measure) {
    values <- lapply(rownames(mdata$labels), function(label) {
        do.call(measure, list(mdata$dataset[label], predicted[, label]))
    })
    names(values) <- rownames(mdata$labels)
    unlist(values)
}

#' Cost-based loss function for multi-label classification
#'
#' @param mdata A mldr dataset containing the test data.
#' @param mlresult An object of mlresult that contain the scores and bipartition
#'  values.
#' @param cost The cost of classification each positive label. If a single value
#'  is informed then the all labels have tha same cost.
#' @references
#'  Al-Otaibi, R., Flach, P., & Kull, M. (2014). Multi-label Classification: A
#'  Comparative Study on Threshold Selection Methods. In First International
#'  Workshop on Learning over Multiple Contexts (LMCE) at ECML-PKDD 2014.
#' @export
multilabel_loss_function <- function (mdata, mlresult, cost = 0.5) {
  if (length(cost) == 1) {
    cost <- rep(cost, mdata$measures$num.labels)
    names(cost) <- rownames(mdata$labels)
  }
  else if (is.null(names(cost))) {
    names(cost) <- rownames(mdata$label)
  }

  prediction <- as.bipartition(mlresult)
  labels <- utiml_renames(rownames(mdata$labels))
  partial.results <- lapply(labels, function (lname) {
    FN <- sum(mdata$dataset[,lname] == 1 & prediction [,lname] == 0) /
      mdata$measures$num.instances
    FP <- sum(mdata$dataset[,lname] == 0 & prediction [,lname] == 1) /
      mdata$measures$num.instances
    freq <- mdata$labels[lname, "freq"]
    2 * ((cost[lname] * freq * FN) + ((1 - cost[lname]) * (1 - freq) * FP))
  })

  mean(unlist(partial.results))
}
