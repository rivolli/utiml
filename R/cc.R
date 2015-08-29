#' @title Classifier Chains for multi-label Classification
#' @family Transformation methods
#' @description Create a Classifier Chains model for multilabel classification.
#'
#'   Classifier Chains is a Binary Relevance transformation method based to predict
#'   multi-label data. This is based on the one-versus-all approach to build a
#'   specific model for each label. It is different from BR method due the strategy
#'   of extended the attribute space with the 0/1 label relevances of all previous
#'   classifiers, forming a classifier chain.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string or a named vector with the base method(s)
#'   name(s). If a single value is passed the same base method will be used for
#'   train all subproblems. If a named vector is passed then each subproblem can
#'   be trained by a specific base method. When a named vector is used the size
#'   and the name of its elements must have exactly the number and name of the
#'   labels.
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'   \code{'SVM'})
#' @param chain A vector with the label names to define the chain order. If
#'   empty the chain is the default label sequence of the dataset. (default:
#'   \code{list()})
#' @param ... Others arguments passed to the base method for all subproblems
#'   (recommended only when the same base method is used for all labels).
#' @param specific.params A named list to pass parameters for a specific model
#'   (the name of the list define wich model will use the arguments). (default:
#'   \code{list()})
#' @param predict.params A list of default arguments passed to the predict
#'  method (recommended only when the same base method is used for all labels).
#'  (default: \code{list()})
#' @param predict.specific.params A named list to pass parameters for a
#'  specific predict method. (the name of the list define wich predict method
#'  will use the arguments). (default: \code{list()})
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: FALSE)
#'
#' @return An object of class \code{CCmodel} containing the set of fitted
#'   models, including: \describe{ \item{chain}{A vector with the chain order}
#'   \item{models}{A list of models named by the label names. The model type is
#'   defined by the base method used to train each subproblem} \item{datasets}{A
#'   list of \code{mldCC} named by the label names. Only when the
#'   \code{save.datasets = TRUE}.} }
#'
#' @references
#'  Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains
#'    for multi-label classification. Machine Learning, 85(3), 333–359.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Classifier Chains
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- cc(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use a specific chain with C4.5 classifier
#' mychain <- sample(rownames(emotions$labels))
#' model <- cc(emotions, "C4.5", mychain)
#' pred <- predict(model, testdata)
#'
#' # Set a parameters for all subproblems
#' model <- cc(emotions, "KNN", k=5, predict.params=list(k=5))
#' pred <- predict(model, testdata)
cc <- function (mdata,
                base.method = "SVM",
                chain = c(),
                ...,
                specific.params = list(),
                predict.params = list(),
                predict.specific.params = list(),
                save.datasets = FALSE
              ) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (length(base.method) != 1 && length(base.method) != mdata$measures$num.labels)
    stop('Invalid number of base methods (use only one for all labels or one for each label)')

  if (length(chain) == 0)
    chain <- rownames(mdata$labels)
  else {
    labels <- rownames(mdata$labels)
    if (length(chain) != mdata$measures$num.labels ||
        length(setdiff(union(chain,labels),intersect(chain,labels))) > 0
        ) {
      stop('Invalid chain (all labels must be on the chain)')
    }
  }

  #CC Model class
  ccmodel <- list()
  ccmodel$chain = chain
  ccmodel$models <- list()
  if (save.datasets) {
    ccmodel$datasets <- list()
  }

  #Relating Base methods with labels
  if (length(base.method) != mdata$measures$num.labels) {
    base.method <- rep(base.method, mdata$measures$num.labels)
    names(base.method) <- ccmodel$chain
  } else if (!all(names(base.method) %in% ccmodel$chain))
    stop("The names(base.method) must contain the name of all labels")

  basedata <- mdata$dataset[mdata$attributesIndexes]
  newattrs <- matrix(nrow=mdata$measures$num.instances, ncol=0)
  for (label in chain) {
    #Transformation
    dataset <- cbind(basedata, mdata$dataset[label])

    #Convert the class column as factor
    dataset[,label] <- as.factor(dataset[,label])

    #Create data
    mldCC <- list(data = dataset, labelname = label, labelindex = ncol(dataset), methodname = base.method[label])
    class(mldCC) <- c("mldCC", paste("base", base.method[label], sep=''), "mltransformation")

    #Merge defaul parameter with specific parameters
    params <- c(list(dataset=mldCC), ...)
    for (pname in names(specific.params[[label]])) {
      params[[pname]] <- specific.params[[label]][[pname]]
    }

    #Call dynamic multilabel model with merged parameters
    model <- do.call(mltrain, params)
    attr(model, "labelname") <- label
    attr(model, "methodname") <- dataset$methodname

    extra <- predict.params
    for (arg in names(predict.specific.params[[label]])) {
      extra[[arg]] <- predict.specific.params[[label]][[arg]]
    }
    result <- do.call(mlpredict, c(list(model = model, newdata = basedata), extra))
    basedata <- cbind(basedata, result$bipartition)
    names(basedata)[ncol(basedata)] <- label

    if (save.datasets) {
      ccmodel$datasets[[label]] <- mldCC
    }
    ccmodel$models[[label]] <- model
  }

  ccmodel$call <- match.call()
  class(ccmodel) <- "CCmodel"

  ccmodel
}

#' @title Predict Method for Classifier Chains
#' @description This function predicts values based upon a model trained by \code{cc}.
#'
#' @param object Object of class "\code{CCmodel}", created by \code{\link{cc}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data.
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems (recommended only when the same base method is used for all labels).
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param specific.params A named list to pass parameters for a specific model (the name
#'   of the list define wich model will use the arguments) (default: \code{list()}).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @seealso \code{\link[=cc]{Classifier Chains (CC)}}
#'
#' @export
#'
#' @examples
#' library(utiml)
#'
#' # Emotion multi-label dataset using Binary Relevance
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- cc(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions
#' pred <- predict(model, testdata, probability = FALSE)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
predict.CCmodel <- function (object,
                             newdata,
                             ...,
                             probability = TRUE,
                             specific.params = list()
                            ) {
  #Validations
  if(class(object) != 'CCmodel')
    stop('First argument must be an CCmodel object')

  predictions <- list()
  for (label in object$chain) {
    params <- c(list(model = object$models[[label]], newdata = newdata), ...)
    for (pname in names(specific.params[[label]])) {
      params[[pname]] <- specific.params[[label]][[pname]]
    }

    predictions[[label]] <- do.call(mlpredict, params)
    newdata <- cbind(newdata, predictions[[label]]$bipartition)
    names(newdata)[ncol(newdata)] <- label
  }

  result <- if (probability)
    sapply(predictions, function (lblres) as.numeric(as.character(lblres$probability)))
  else
    sapply(predictions, function (lblres) as.numeric(as.character(lblres$bipartition)))
  rownames(result) <- names(predictions[[1]]$bipartition)

  result
}

print.CCmodel <- function (x, ...) {
  cat("Classifier Chains Model\n\nCall:\n")
  print(x$call)
  cat("\n Chain: (", length(x$chain), "labels )\n")
  print(x$chain)
}

print.mldCC <- function (x, ...) {
  cat("Classifier Chains Transformation Dataset\n\n")
  cat("Label:\n  ", x$labelname, " (", x$methodname, " method)\n\n", sep="")
  cat("Dataset info:\n")
  cat(" ", ncol(x$data) - 1, "Predictive attributes\n")
  cat(" ", nrow(x$data), "Examples\n")
  cat("  ", round((sum(x$data[,ncol(x$data)] == 1) / nrow(x$data)) * 100, 1), "% of positive examples\n", sep="")
}
