mbr <- function (mdata,
                  base.method = "SVM",
                  phi = 0,
                  ...,
                  predict.params = list(),
                  save.datasets = FALSE,
                  CORES = 1
) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #MBR Model class
  mbrmodel <- list()
  mbrmodel$labels <- rownames(mdata$labels)
  mbrmodel$phi <- phi

  #1 Iteration - Base Level
  mbrmodel$basemodel <- br(mdata, base.method, ..., save.datasets = TRUE, CORES = CORES)
  params <- list(object = mbrmodel$basemodel,
                 newdata = mdata$dataset[mdata$attributesIndexes],
                 probability = FALSE, CORES = CORES)
  base.preds <- do.call(predict, c(params, predict.params))

  #2 Iteration - Meta level
  corr <- mbrmodel$correlation <- labelsPhiCorrelationCoefficient(mdata)
  datasets <- lapply(mbrmodel$basemodel$datasets, function (dataset) {
    extracolumns <- base.preds[,colnames(corr)[corr[dataset$labelname,] > phi]]
    colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
    base <- cbind(dataset$data[-dataset$labelindex], extracolumns, dataset$data[dataset$labelindex])
    binary_transformation(base, "mldBR", base.method)
  })
  mbrmodel$metamodels <- utiml_lapply(datasets, br.create_model, CORES, ...)

  if (save.datasets)
    mbrmodel$datasets <- list(base = mbrmodel$basemodel$datasets, meta = datasets)

  mbrmodel$basemodel$datasets <- NULL

  mbrmodel$call <- match.call()
  class(mbrmodel) <- "MBRmodel"

  mbrmodel
}

predict.MBRmodel <- function (object,
                              newdata,
                              ...,
                              probability = TRUE,
                              CORES = 1
) {
  #Validations
  if(class(object) != 'MBRmodel')
    stop('First argument must be an MBRmodel object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #1 Iteration - Base level
  base.preds <- predict(object$basemodel, newdata, ..., probability = FALSE, CORES = CORES)

  #2 Iteration - Meta level
  corr <- object$correlation
  predictions <- utiml_lapply(object$labels, function (labelname) {
    extracolumns <- base.preds[,colnames(corr)[corr[labelname,] > object$phi]]
    colnames(extracolumns) <- paste("extra", colnames(extracolumns), sep = ".")
    br.predict_model(object$metamodels[[labelname]], cbind(newdata, extracolumns), ...)
  }, CORES)
  names(predictions) <- object$labels

  as.resultMLPrediction(predictions, probability)
}

print.MBRmodel <- function (x, ...) {
  cat("Classifier Meta-BR (also called 2BR)\n\nCall:\n")
  print(x$call)
  cat("\nPhi:", x$phi, "\n")
  cat("\nCorrelation Table Overview:\n")
  corr <- x$correlation
  diag(corr) <- NA
  tbl <- data.frame(
    min = apply(corr, 1, min, na.rm = TRUE),
    mean = apply(corr, 1, mean, na.rm = TRUE),
    median = apply(corr, 1, median, na.rm = TRUE),
    max = apply(corr, 1, max, na.rm = TRUE),
    extra = apply(x$correlation, 1, function (row) sum(row > x$phi))
  )
  print(tbl)
}
