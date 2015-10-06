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
utiml_measure_precision <- function (expected, predict) {
  sum(predict & expected) / sum(predict)
}

utiml_measure_recall <- function (expected, predict) {
  sum(predict & expected) / sum(expected)
}

utiml_measure_f1 <- function (expected, predict) {
  Precision <- utiml_measure_precision(expected, predict)
  Recall <- utiml_measure_recall(expected, predict)
  (2 * Precision * Recall) / (Precision + Recall)
}

utiml_measure_labels <- function (mdata, predicted, measure) {
  values <- lapply(rownames(mdata$labels), function (label) {
    do.call(measure, list(mdata$dataset[label], predicted[,label]))
  })
  names(values) <- rownames(mdata$labels)
  unlist(values)
}
