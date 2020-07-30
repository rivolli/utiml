utiml_binary_prediction <- function(bipartition, probability) {
  res <- list(bipartition = bipartition, probability = probability)
  class(res) <- "binary.prediction"
  res
}

utiml_create_binary_data <- function (mdata, label.name, extra.columns = NULL) {
  if (is.null(extra.columns)) {
    cbind(mdata$dataset[mdata$attributesIndexes], mdata$dataset[label.name])
  }
  else {
    cbind(mdata$dataset[mdata$attributesIndexes],
          extra.columns,
          mdata$dataset[label.name])
  }
}

utiml_create_pairwise_data <- function (mdata, label1, label2) {
  mdata$dataset[xor(mdata$dataset[label1], mdata$dataset[label2]),
                c(mdata$attributesIndexes,mdata$labels[label1, "index"])]
}

utiml_create_lp_data <- function (mdata) {
  cbind(mdata$dataset[mdata$attributesIndexes],
        classlp=factor(apply(mdata$dataset[mdata$labels$index], 1, paste,
                             collapse="")))
}

utiml_create_model <- function(utiml.object, ...) {
  labelinfo <- table(utiml.object$data[utiml.object$labelname])

  #if ((any(labelinfo < 2) & length(labelinfo) == 2) | length(labelinfo) < 2) {
  if (any(labelinfo < 1) | length(labelinfo) < 2) {
    #There are no sufficient examples to train (create a empty model)
    model <- list()
    class(model) <- "emptyModel"
  } else {
    # Call dynamic multilabel model with merged parameters
    model <- do.call(mltrain, c(list(object = utiml.object), list(...)))
  }
  attr(model, "dataset") <- utiml.object$mldataset
  attr(model, "label") <- utiml.object$labelname

  model
}

utiml_predict <- function (predictions, probability) {
  bipartitions <- do.call(cbind, lapply(predictions, function(lblres) {
    lblres$bipartition
  }))

  probabilities <- do.call(cbind, lapply(predictions, function(lblres) {
    lblres$probability
  }))

  multilabel_prediction(bipartitions, probabilities, probability)
}

utiml_predict_binary_model <- function(model, newdata, ...) {
  result <- do.call(mlpredict, c(list(model = model, newdata = newdata),
                                 list(...)))

  if (any(rownames(result) != rownames(newdata))) {
    where <- paste(attr(model, "dataset"), "/", attr(model, "label"))
    warning(cat("The order of the predicted instances from", where,
                "are wrong!\n", sep=' '))
  }

  #Because the factores is necessary first convert to character
  bipartition <- as.numeric(as.character(result$prediction))
  probability <- result$probability

  zeros <- bipartition == 0
  probability[zeros] <- 1 - probability[zeros]

  names(bipartition) <- names(probability) <- rownames(result)
  utiml_binary_prediction(bipartition, probability)
}

utiml_predict_multiclass_model <- function (model, newdata, labels, probability,
                                            ...) {
  result <- do.call(mlpredict, c(list(model = model, newdata = newdata),
                                 list(...)))
  classes <- do.call(rbind, lapply(
    strsplit(as.character(result$prediction),""), as.numeric)
  )
  dimnames(classes) <- list(rownames(newdata), labels)

  probs <- apply(classes, 2, function (col) {
    ifelse(col == 1, result$probability, 0)
  })

  multilabel_prediction(classes, probs, probability)
}

utiml_prepare_data <- function(dataset, classname, mldataset, mlmethod,
                               base.algorithm, ...) {
  label <- colnames(dataset)[ncol(dataset)]

  # Convert the class column as factor
  dataset[, label] <- as.factor(dataset[, label])

  # Create object
  object <- list(
    data = dataset,
    labelname = label,
    labelindex = ncol(dataset),
    mldataset = mldataset,
    mlmethod = mlmethod,
    base.algorithm = base.algorithm
  )

  extra <- list(...)
  for (nextra in names(extra)) {
    object[[nextra]] <- extra[[nextra]]
  }

  basename <- paste("base", base.algorithm, sep = "")
  class(object) <- c(classname, basename, "mltransformation")

  object
}

#' Summary method for mltransformation
#' @param object A transformed dataset
#' @param ... additional arguments affecting the summary produced.
#' @export
summary.mltransformation <- function(object, ...) {
  summary(object$data, ...)
}
