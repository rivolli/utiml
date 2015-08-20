getResultPrediction <- function (bipartition, ranking) {
  res <- list(bipartition = bipartition, ranking = ranking)
  class(res) <- "mlresult"

  res
}

mltrain.SVM <- function (dataset, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    traindata <- dataset$data[, -ncol(dataset$data)]
    labeldata <- dataset$data[, dataset$labelname]
    model <- e1071::svm(traindata, labeldata, probability = TRUE, ...)
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  model
}

mlpredict.SVM <- function (model, newdata, ...) {
  if (requireNamespace("e1071", quietly = TRUE)) {
    result <- predict(model, newdata, probability = TRUE, ...)
  } else
    stop('There are no installed package (e1071) to use SVM as base method')

  pscore <- attr(result, "probabilities")[,"1"]
  attr(result, "probabilities") <- NULL

  getResultPrediction(result, pscore)
}

# getMultilabelModel <- function (dataset, ...) UseMethod("getMultilabelModel")
#
# getMultilabelModel.default <- function (dataset, ...) {
#   stop(paste("The method 'getModel.", class(dataset), "(dataset, ...)' is dataset$data[, dataset$labelname]not implemented", sep=''))
# }
#
# getMultilabelModel.br.SVM <- function (dataset, ...) {
#   #TODO is not possible use kernlab package if e1071 are installed (review this in next reviews)
#   if (requireNamespace("e1071", quietly = TRUE))
#     model <- e1071::svm(dataset$data[, -ncol(dataset$data)], dataset$data[, dataset$labelname], ...)
#   else if (requireNamespace("kernlab", quietly = TRUE))
#     model <- kernlab::ksvm(as.matrix(dataset$data[, -ncol(dataset$data)]), dataset$data[, dataset$labelname], ...)
#   else
#     stop('There are no installed package (e1071 or kernlab) to use SVM as base method')
#
#   model
# }
#
# getMultilabelModel.br.NB <- function (dataset, ...) {
#   if (requireNamespace("e1071", quietly = TRUE))
#     model <- naiveBayes(dataset[,-ncol(dataset)], dataset$data[, dataset$labelname], ...)
#   else
#     stop('There are no installed package (e1071) to use Naive Bayes as base method')
#
#   model
# }
