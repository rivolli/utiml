brplus <- function (mdata,
                    base.method = "SVM",
                    ...,
                    save.datasets = FALSE,
                    CORES = 1
                  ) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BRplus Model class
  brpmodel <- list()
  freq <- mdata$labels$freq
  names(freq) <- rownames(mdata$labels)
  brpmodel$freq <- sort(freq)
  brpmodel$initial <- br(mdata, base.method, ..., save.datasets = save.datasets, CORES = CORES)

  basedata <- mdata$dataset[mdata$attributesIndexes]
  labeldata <- mdata$dataset[mdata$labels$index]
  datasets <- utiml_lapply(1:mdata$measures$num.labels, function (li) {
    br.transformation(cbind(basedata, labeldata[-li], labeldata[li]), "mldBRP", base.method)
  }, CORES)
  names(datasets) <- rownames(mdata$labels)
  brpmodel$models <- utiml_lapply(datasets, br.create_model, CORES, ...)

  if (save.datasets) {
    bprmodel$datasets <- list(initial = brpmodel$initial$datasets, final = datasets)
    brpmodel$initial$datasets <- NULL
  }

  brpmodel$call <- match.call()
  class(brpmodel) <- "BRPmodel"

  brpmodel
}

predict.BRPmodel <- function (object,
                              newdata,
                              strategy = c("Dyn", "Stat", "Ord", "NU"),
                              ...,
                              probability = TRUE,
                              order = list(),
                              CORES = 1
                             ) {
  #Validations
  if(class(object) != 'BRPmodel')
    stop('First argument must be an BRPmodel object')

  strategies <- c("Dyn", "Stat", "Ord", "NU")
  if(!strategy[1] %in% strategies)
    stop(paste("Strategy value must be '", paste(strategies, collapse = "' or '"), "'", sep=""))

  labels <- model$initial$labels
  if (strategy[1] == "Ord") {
    if (length(order) != length(labels))
      stop('The ordered list must be the same size of the labels')

    if (!all(order %in% labels))
      stop('The ordered list must contain all label names')
  }

  if (CORES < 1)
    stop('Cores must be a positive value')

  if (strategy[1] == "NU") {
    initial.preds <- predict(object$initial, newdata, ..., probability = FALSE, CORES = CORES)
    predictions <- utiml_lapply(1:length(labels), function (li) {
      br.predict_model(object$models[[li]], cbind(newdata, initial.preds[,-li]), ...)
    }, CORES)
    names(predictions) <- labels
  }
  else {
    initial.probs <- predict(object$initial, newdata, ..., probability = TRUE, CORES = CORES)
    initial.preds <- simple.threshold(initial.probs)
    orders <- list(
      Dyn = names(sort(apply(initial.preds, 2, mean))),
      Stat = names(model$freq),
      Ord = order
    )

    predictions <- list()
    for (labelname in orders[[strategy[1]]]) {
      model <- object$models[[labelname]]
      data <- cbind(newdata, initial.preds[,!labels %in% labelname])
      predictions[[labelname]] <- br.predict_model(model, data, ...)
      initial.preds[,labelname] <- predictions[[labelname]]$bipartition
    }
  }

  result <- as.resultMLPrediction(predictions, probability)
  result[,labels]
}

print.BRPmodel <- function (x, ...) {
  cat("Classifier BR+ (also called BRplus)\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$labels), "Models (labels):\n")
  print(x$labels)
}

