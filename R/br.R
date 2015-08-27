#' @title Binary Relevance for multi-label Classification
#' @description Create a Binary Relevance model for multilabel classification.
#'
#'   Binary Relevance is a simple and effective transformation method to predict
#'   multi-label data. This is based on the one-versus-all approach to build a
#'   specific model for each label.
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
#'   \href{https://github.com/rivolli/utiml}{online documentation}. (default:
#'    \code{'SVM'}).
#' @param ... Others arguments passed to the base method for all subproblems
#'   (recommended only when the same base method is used for all labels).
#' @param specific.params A named list to pass parameters for a specific model
#'   (the name of the list define wich model will use the arguments) (default:
#'   \code{list()}).
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not (default: FALSE).
#' @param CORES The number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return An object of class \code{BRmodel} containing the set of fitted
#'   models, including: \describe{ \item{labels}{A vector with the label names}
#'   \item{models}{A list of models named by the label names. The model type is
#'   defined by the base method used to train each subproblem} \item{datasets}{A
#'   list of \code{mldBR} named by the label names. Only when the
#'   \code{save.datasets = TRUE}.} }
#'
#' @section Warning:
#'    RWeka package does not permit use \code{'C4.5'} in parallel mode, use
#'    \code{'C5.0'} or \code{'CART'} instead of it
#'
#' @references
#'  Boutell, M. R., Luo, J., Shen, X., & Brown, C. M. (2004). Learning
#'    multi-label scene classification. Pattern Recognition, 37(9), 1757–1771.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using Binary Relevance
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- br(emotions)
#' pred <- predict(model, testdata)
#'
#' # Change the default base method
#' model <- br(emotions, "C4.5")
#' pred <- predict(model, testdata)
#'
#' # Set a parameters for all subproblems
#' model <- br(emotions, "KNN", k=5)
#' pred <- predict(model, testdata)
#'
#' # Use differents base classifers for different labels and running in parallel
#' methods <- c("SVM", "RF", "CART", "C5.0", "KNN", "RF")
#' names(methods) <- rownames(emotions$labels)
#' model <- br(emotions, methods, CORES=6)
#' pred <- predict(model, testdata, CORES=6)
#'
#' # Change SVM kernel for label 'happy-pleased'
#' extra <- list('happy-pleased' = list("kernel" = "linear"))
#' model <- br(emotions, specific.params=extra)
#' pred <- predict(model, testdata)
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
  if (length(base.method) != mdata$measures$num.labels) {
    base.method <- rep(base.method, mdata$measures$num.labels)
    names(base.method) <- brmodel$labels
  } else if (!all(names(base.method) %in% brmodel$labels))
    stop("The names(base.method) must contain the name of all labels")

  #Transformation
  datasets <- lapply(mldr_transform(mdata), function (dataset) {
    label <- colnames(dataset)[length(dataset)]

    #Convert the class column as factor
    dataset[,label] <- as.factor(dataset[,label])
    dataset$data[, label]

    #Create data
    dataset <- list(data = dataset, labelname = label, methodname = base.method[label])
    class(dataset) <- c("mldBR", paste("base", base.method[label], sep=''), "mltransformation")

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
    model <- do.call(mltrain, params)
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

#' @title Predict Method for Binary Relevance
#' @description This function predicts values based upon a model trained by \code{br}.
#'
#' @param object Object of class "\code{BRmodel}", created by \code{br}.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data.
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems (recommended only when the same base method is used for all labels).
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param specific.params A named list to pass parameters for a specific model (the name
#'   of the list define wich model will use the arguments) (default: \code{list()}).
#' @param CORES The number of cores to parallelize the prediction. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @section Warning:
#'    RWeka package does not permit use \code{'C4.5'} in parallel mode, use
#'    \code{'C5.0'} or \code{'CART'} instead of it
#'
#' @seealso \code{\link[=br]{Binary Relevance (BR)}}
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
#' model <- br(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, testdata, probability = FALSE, CORES = 6)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
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

    params <- c(list(model = model, newdata = newdata), ...)
    for (pname in names(specific.params[[label]])) {
      params[[pname]] <- specific.params[[label]][[pname]]
    }

    do.call(mlpredict, params)
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

