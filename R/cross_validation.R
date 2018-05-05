#' Multi-label cross-validation
#'
#' Perform the cross validation procedure for multi-label learning.
#'
#' @family evaluation
#' @param mdata A mldr dataset.
#' @param method The multi-label classification method. It also accepts the name
#'  of the method as a string.
#' @param ... Additional parameters required by the method.
#' @param cv.folds Number of folds. (Default: 10)
#' @param cv.sampling The method to split the data. The default methods are:
#' \describe{
#'    \item{random}{Split randomly the folds.}
#'    \item{iterative}{Split the folds considering the labels proportions
#'                      individually. Some specific label can not occurs in all
#'                      folds.}
#'    \item{stratified}{Split the folds considering the labelset proportions.}
#'  }
#'  (Default: "random")
#' @param cv.results Logical value indicating if the folds results should be
#'  reported (Default: FALSE).
#' @param cv.predictions Logical value indicating if the predictions should be
#'  reported (Default: FALSE).
#' @param cv.measures The measures names to be computed. Call
#'  \code{multilabel_measures()} to see the expected measures. You can also
#'  use \code{"bipartition"}, \code{"ranking"}, \code{"label-based"},
#'  \code{"example-based"}, \code{"macro-based"}, \code{"micro-based"} and
#'  \code{"label-problem"} to include a set of measures. (Default: "all").
#' @param cv.cores The number of cores to parallelize the cross validation
#'  procedure. (Default: \code{options("utiml.cores", 1)})
#' @param cv.seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#'
#' @return If cv.results and cv.prediction are FALSE, the return is a vector
#'  with the expected multi-label measures, otherwise, a list contained the
#'  multi-label and the other expected results (the label measures and/or the
#'  prediction object) for each fold.
#'
#' @export
#'
#' @examples
#' #Run 10 folds for BR method
#' res1 <- cv(toyml, br, base.algorithm="RANDOM", cv.folds=10)
#'
#' #Run 3 folds for RAkEL method and get the fold results and the prediction
#' res2 <- cv(mdata=toyml, method="rakel", base.algorithm="RANDOM", k=2, m=10,
#'  cv.folds=3, cv.results=TRUE, cv.predictions=TRUE)
cv <- function(mdata, method, ..., cv.folds=10,
               cv.sampling=c("random", "iterative", "stratified"),
               cv.results=FALSE, cv.predictions=FALSE, cv.measures="all",
               cv.cores=getOption("utiml.cores", 1),
               cv.seed=getOption("utiml.seed", NA)) {
  if (!is.na(cv.seed)) {
    set.seed(cv.seed)
  }

  cvdata <- create_kfold_partition(mdata, cv.folds, cv.sampling)
  results <- parallel::mclapply(seq(cv.folds), function (k){
    ds <- partition_fold(cvdata, k)
    model <- do.call(method, c(list(mdata=ds$train), ...))
    pred <- stats::predict(model, ds$test, ...)

    c(
      list(pred=pred),
      multilabel_evaluate(ds$test, pred, cv.measures, labels=TRUE)
    )
  }, mc.cores = cv.cores)

  obj = list(multilabel=do.call(rbind, lapply(results, "[[", "multilabel")))

  if (cv.results) {
    labels <- rownames(mdata$labels)
    lfolds <- lapply(results, "[[", "labels")
    obj$labels <- sapply(labels,
                         function(lbl) t(sapply(lfolds, function(x) x[lbl,])),
                         simplify = FALSE)
  }

  if (cv.predictions) {
    obj$predictions <- lapply(results, "[[", "pred")
  }

  if (length(obj) == 1) {
    return(colMeans(obj[[1]]))
  } else {
    obj
  }
}
