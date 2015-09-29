rdbr <- function (mdata,
                  base.method = "SVM",
                  ...,
                  estimate.models = TRUE,
                  save.datasets = FALSE,
                  CORES = 1
) {
  rdbrmodel <- dbr(
    mdata, base.method, ...,
    estimate.models = estimate.models,
    save.datasets = save.datasets,
    CORES = CORES)
  class(rdbrmodel) <- "RDBRmodel"
  rdbrmodel
}

predict.RDBRmodel <- function (object,
                               newdata,
                               ...,
                               max.iterations = 5,
                               batch.mode = FALSE,
                               estimative = NULL,
                               probability = TRUE,
                               CORES = 1
) {
  #Validations
  if(class(object) != 'RDBRmodel')
    stop('First argument must be an RDDBRmodel object')

  if (is.null(object$estimation) && is.null(estimative))
    stop('The model requires an estimative matrix')

  if (max.iterations < 1)
    stop('The number of iteractions must be positive')

  if (CORES < 1)
    stop('Cores must be a positive value')

  newdata <- utiml_newdata(newdata)
  if (is.null(estimative))
    estimative <- predict(object$estimation, newdata, ..., probability = FALSE, CORES = CORES)

  labels <- names(object$models)
  if (batch.mode) {
    for (i in 1:max.iterations) {
      predictions <- utiml_lapply(1:length(labels), function (li) {
        br.predict_model(object$models[[li]], cbind(newdata, estimative[,-li]), ...)
      }, CORES)
      names(predictions) <- labels
      new.estimative <- do.call(cbind, lapply(predictions, function (lbl) lbl$bipartition))
      if (all(new.estimative == estimative)) break
      estimative <- new.estimative
    }
  }
  else {
    for (i in 1:max.iterations) {
      old.estimative <- estimative
      predictions <- list()
      for (li in 1:length(labels)) {
        predictions[[li]] <- br.predict_model(object$models[[li]], cbind(newdata, estimative[,-li]), ...)
        estimative[,li] <- predictions[[li]]$bipartition
      }
      names(predictions) <- labels
      if (all(old.estimative == estimative)) break
    }
  }

  as.resultMLPrediction(predictions, probability)
}

print.RDBRmodel <- function (x, ...) {
  cat("Classifier RDBR\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$models), "Models (labels):\n")
  print(names(x$models))
}
