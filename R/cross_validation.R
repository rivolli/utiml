#' Multi-label cross-validation
#'
#' Perform the cross validation procedure for multi-label learning.
#' @export
cv <- function(method, mdata, ..., cv.folds=10,
               cv.sampling=c("random", "iterative", "stratified"),
               cv.results=FALSE, cv.measures="all",
               cv.cores=getOption("utiml.cores", 1),
               cv.seed=getOption("utiml.seed", NA)) {
  if (!is.na(cv.seed)) {
    set.seed(cv.seed)
  }

  cvdata <- create_kfold_partition(mdata, cv.folds, cv.sampling)
  results <- parallel::mclapply(seq(cv.folds), function (k){
    ds <- partition_fold(cvdata, k)
    model <- do.call(method, c(list(mdata=ds$train), ...))
    pred <- predict(model, ds$test, ...)
    multilabel_evaluate(ds$test, pred, cv.measures, labels=TRUE)
  }, mc.cores = cv.cores)

  if (cv.results) {
    labels <- rownames(mdata$labels)
    lfolds <- lapply(results, "[[", "labels")
    list(
      multilabel=do.call(rbind, lapply(results, "[[", "multilabel")),
      labels=sapply(labels,
                    function(lbl) t(sapply(lfolds, function(x) x[lbl,])),
                    simplify = FALSE)
    )
  }
  else {
    colMeans(do.call(rbind, lapply(results, "[[", "multilabel")))
  }
}
