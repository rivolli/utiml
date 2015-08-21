br <- function (mdata,
                base.method = "SVM",
                ...,
                specific.params = list(),
                save.datasets = FALSE,
                CORES = 1
              ) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (length(base.method) != 1 && length(base.method) != mdata$measures$num.labels)
    stop('Invalid number of base methods (use only one for all labels or one for each label)')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BR Model class
  brmodel <- list()
  brmodel$labels = rownames(mdata$labels)

  #Relating Base methods with labels
  if (length(base.method) != mdata$measures$num.labels)
    base.method <- rep(base.method, mdata$measures$num.labels)
  names(base.method) <- brmodel$labels

  #Transformation
  datasets <- lapply(mldr_transform(mdata), function (dataset) {
    label <- colnames(dataset)[length(dataset)]

    #Convert the class column as factor
    dataset[,label] <- as.factor(dataset[,label])
    dataset$data[, label]

    #Create data
    dataset <- list(data = dataset, labelname = label, methodname = base.method[label])
    class(dataset) <- "mldBR"

    #Set specific parameters
    dataset$specific.params <- if (!is.null(specific.params[[label]])) specific.params[[label]]

    dataset
  })
  names(datasets) <- brmodel$labels
  if (save.datasets) {
    brmodel$datasets <- datasets
  }

  #Create Dynamically the model
  create_model <- function (dataset, ...) {
    #Merge defaul parameter with specific parameters
    params <- c(list(dataset=dataset), ...)
    for (pname in names(dataset$specific.params)) {
      params[[pname]] <- dataset$specific.param[[pname]]
    }

    #Call dynamic multilabel model with merged parameters
    funcname <- paste("mltrain", dataset$methodname, sep=".")
    if (!existsFunction(funcname))
      stop(paste("The function '", funcname, "(dataset, ...)' is not implemented", sep=''))

    model <- do.call(funcname, params)
    attr(model, "BRlabel") <- dataset$labelname
    attr(model, "BRmethod") <- dataset$methodname

    model
  }

  #Create models
  brmodel$models <- if (CORES == 1)
      lapply(datasets, create_model, ...)
    else
      parallel::mclapply(datasets, create_model, ..., mc.cores=CORES) #min(CORES, length(datasets))

  brmodel$call <- match.call()
  class(brmodel) <- "BRmodel"

  brmodel
}

predict.BRmodel <- function (object,
                             newdata,
                             ...,
                             probability = TRUE,
                             specific.params = list(),
                             CORES = 1
                             ) {
  #Validations
  if(class(object) != 'BRmodel')
    stop('First argument must be an BRmodel object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  predict_model <- function (model, ...) {
    label <- attr(model, "BRlabel")
    method <- attr(model, "BRmethod")

    params <- c(list(model = model, newdata = newdata), ...)
    for (pname in names(specific.params[[label]])) {
      params[[pname]] <- specific.params[[label]][[pname]]
    }

    #Call dynamic multilabel model with merged parameters
    funcname <- paste("mlpredict", method, sep=".")
    if (!existsFunction(funcname))
      stop(paste("The function '", funcname, "(dataset, ...)' is not implemented", sep=''))

    do.call(funcname, params)
  }

  #Create models
  predictions <- if (CORES == 1)
    lapply(object$models, predict_model, ...)
  else
    parallel::mclapply(object$models, predict_model, ..., mc.cores=CORES) #min(CORES, length(datasets))

  result <- if (probability)
      sapply(predictions, function (lblres) as.numeric(as.character(lblres$probability)))
    else
      sapply(predictions, function (lblres) as.numeric(as.character(lblres$bipartition)))
  rownames(result) <- names(predictions[[1]]$bipartition)

  result
}

getTestData <- function () {
  emotions$dataset[sample(1:emotions$measures$num.instances, 10), emotions$attributesIndexes]
}

print.BRmodel <- function (x, ...) {
  cat("Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$labels), "Models (labels):\n")
  print(x$labels)
}

print.mldBR <- function (x, ...) {
  cat("Binary Relevance Transformation Dataset\n\n")
  cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep="")
  cat("Dataset info:\n")
  cat(" ", ncol(x$data) - 1, "Predictive attributes\n")
  cat(" ", nrow(x$data), "Examples\n")
  cat("  ", round((sum(x$data[,ncol(x$data)] == 1) / nrow(x$data)) * 100, 1), "% of positive examples\n", sep="")
}

summary.mldBR <- function (x, ...) {
  summary(x$data)
}

