ns <- function (mdata,
                base.method = "SVM",
                chain = c(),
                ...,
                predict.params = list(),
                save.datasets = FALSE,
                CORES = 1
) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  labels <- rownames(mdata$labels)
  if (length(chain) == 0)
    chain <- rownames(mdata$labels)
  else {
    if (length(chain) != mdata$measures$num.labels ||
        length(setdiff(union(chain,labels),intersect(chain,labels))) > 0
    ) {
      stop('Invalid chain (all labels must be on the chain)')
    }
  }

  #NS Model class
  nsmodel <- list()
  nsmodel$labels <- labels
  nsmodel$chain <- chain
  nsmodel$models <- list()
  nsmodel$labelsets <- as.matrix(mdata$dataset[,mdata$labels$index])
  if (save.datasets) {
    nsmodel$datasets <- list()
  }

  basedata <- mdata$dataset[mdata$attributesIndexes]
  newattrs <- matrix(nrow=mdata$measures$num.instances, ncol=0)
  for (labelIndex in 1:length(chain)) {
    label <- chain[labelIndex]

    #Create data
    dataset <- cbind(basedata, mdata$dataset[label])
    mldCC <- br.transformation(dataset, "mldCC", base.method, chain.order = labelIndex)

    #Call dynamic multilabel model with merged parameters
    model <- do.call(mltrain, c(list(dataset=mldCC), ...))

    result <- do.call(mlpredict, c(list(model = model, newdata = basedata), predict.params))
    basedata <- cbind(basedata, result$bipartition)
    names(basedata)[ncol(basedata)] <- label

    if (save.datasets) {
      nsmodel$datasets[[label]] <- mldCC
    }
    nsmodel$models[[label]] <- model
  }

  nsmodel$call <- match.call()
  class(nsmodel) <- "NSmodel"

  nsmodel
}

predict.NSmodel <- function (object,
                             newdata,
                             ...,
                             probability = TRUE
) {
  #Validations
  if(class(object) != 'NSmodel')
    stop('First argument must be an NSmodel object')

  predictions <- list()
  for (label in object$chain) {
    params <- c(list(model = object$models[[label]], newdata = newdata), ...)
    predictions[[label]] <- do.call(mlpredict, params)
    newdata <- cbind(newdata, predictions[[label]]$bipartition)
    names(newdata)[ncol(newdata)] <- label
  }

  result <- as.resultMLPrediction(predictions, probability)[,object$labels]
  subset.correction <- c(ns.subsetcorrection, ns.subsetcorrection.score)
  subset.correction[c(!probability, probability)][[1]](result, object$labelsets)
}

ns.subsetcorrection <- function (predicted_y, train_y) {
  if (ncol(predicted_y) != ncol(train_y))
    stop("The number of columns in the predicted result are different from the training data")

  labelsets <- unique(train_y)
  rownames(labelsets) <- apply(labelsets, 1, paste, collapse = "")

  order <- names(sort(table(apply(train_y, 1, paste, collapse = "")), decreasing = TRUE))
  labelsets <- labelsets[order,]

  new.predicted <- t(apply(predicted_y, 1, function (y) {
    labelsets[names(which.min(apply(labelsets, 1, function (row) sum(row != y)))),]
  }))

  new.predicted
}

ns.subsetcorrection.score <- function (predicted_y, train_y, threshold = 0.5) {
  if (ncol(predicted_y) != ncol(train_y))
    stop("The number of columns in the predicted result are different from the training data")

  new.predicted <- as.matrix(predicted_y)

  y <- ns.subsetcorrection(simple.threshold(predicted_y, threshold), train_y)
  for (r in 1:nrow(predicted_y)) {
    row <- predicted_y[r,]

    # Correct the values greater than threshold but that is expected to be lower
    index <- y[r,] - row <= -(threshold)
    new.predicted[index] <- threshold - (1/1-row[index]) / 10

    # Correct the values lower than threshold but that is expected to be greater
    index <- y[r,] - row > threshold
    new.predicted[index] <- threshold + row[index] / 10
  }

  new.predicted
}

print.NSmodel <- function (x, ...) {
  cat("Nested Stacking Model\n\nCall:\n")
  print(x$call)
  cat("\n Chain: (", length(x$chain), "labels )\n")
  print(x$chain)
}
