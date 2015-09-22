#' @title BR+ or BRplus for multi-label Classification
#' @family Transformation methods
#' @description Create a BR+ classifier to predic multi-label data.
#'  This is a simple approach that enables the binary classifiers to
#'  discover existing label dependency by themselves. The main idea of
#'  BR+ is to increment the feature space of the binary classifiers to
#'  let them discover exist- ing label dependency by themselves.
#'
#'  This implementation has different strategy to predict the final
#'  set of labels for unlabeled examples, as proposed in original
#'  paper.
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label train
#'   dataset (provided by \pkg{mldr} package).
#' @param base.method A string with the name of base method. The same base method
#'   will be used for train all subproblems and the BR classifers
#'
#'   Default valid options are: \code{'SVM'}, \code{'C4.5'}, \code{'C5.0'},
#'   \code{'RF'}, \code{'NB'} and \code{'KNN'}. To use other base method see
#'   \code{\link{mltrain}} and \code{\link{mlpredict}} instructions. (default:
#'    \code{'SVM'}).
#' @param ... Others arguments passed to the base method for all subproblems.
#' @param save.datasets Logical indicating whether the binary datasets must be
#'   saved in the model or not. (default: FALSE)
#' @param CORES he number of cores to parallelize the training. Values higher
#'   than 1 require the \pkg{parallel} package. (default: 1)
#'
#' @return An object of class \code{BRPmodel} containing the set of fitted
#'   models, including: \describe{
#'    \item{freq}{The label frequencies to use with the "Stat" strategy}
#'    \item{initial}{The BR model to predict the values for the labels to initial step}
#'    \item{models}{A list of final models named by the label names.}
#'    \item{datasets}{A list with \code{initial} and \code{final} datasets of
#'      type \code{mldBRP} named by the label names. Only when the
#'      \code{save.datasets = TRUE}.
#'    }
#' }
#'
#' @section Warning:
#'    RWeka package does not permit use \code{'C4.5'} in parallel mode, use
#'    \code{'C5.0'} or \code{'CART'} instead of it.
#'
#' @references
#'  Cherman, E. A., Metz, J., & Monard, M. C. (2012). Incorporating label
#'    dependency into the binary relevance framework for multi-label
#'    classification. Expert Systems with Applications, 39(2), 1647–1655.
#'
#' @export
#'
#' @examples
#' # Train and predict emotion multilabel dataset using BRPlus
#' library(utiml)
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Use SVM as base method
#' model <- brplus(emotions)
#' pred <- predict(model, testdata)
#'
#' # Use Random Forest as base method and 4 cores
#' model <- brplus(emotions, "RF", CORES = 4)
#' pred <- predict(model, testdata)
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

#' @title Predict Method for BR+ (brplus)
#' @description This function predicts values based upon a model trained by \code{brplus}.
#'
#' The strategies of estimate the values of the new features are separeted in two groups:
#' \describe{
#'  \item{No Update (\code{NU})}{This use the initial prediction of BR to all labels. This
#'    name is because no modification is made to the initial estimates of the augmented
#'    features during the prediction phase}
#'  \item{With Update}{This strategy update the initial prediction in that the final
#'    predict occurs. There are three possibilities to define the order of label sequences:
#'    \describe{
#'      \item{Specific order (\code{Ord})}{The order is define by the user, require a
#'        new argument  called \code{order}.}
#'      \item{Static order (\code{Stat})}{Use the frequency of single labels in the
#'        training set to define the sequence, where the least frequent labels are
#'        predicted first}
#'      \item{Dinamic order (\code{Dyn})}{Takes into account the confidence of the
#'        initial prediction for each independent single label, to define a sequence,
#'        where the labels predicted with less confidence are updated first.}
#'    }
#'  }
#' }
#'
#' @param object Object of class "\code{BRPmodel}", created by \code{\link{brplus}} method.
#' @param newdata An object containing the new input data. This must be a matrix or
#'          data.frame object containing the same size of training data or a mldr object.
#' @param strategy The strategy prefix to determine how to estimate the values of
#'          the augmented features of unlabeled examples.
#'
#'        The possible values are: \code{'Dyn'}, \code{'Stat'}, \code{'Ord'} or \code{'NU'}.
#'        See the description for more details. (default: \code{'Dyn'}).
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param probability Logical indicating whether class probabilities should be returned.
#'   (default: \code{TRUE})
#' @param order The label sequence used to update the initial labels results based on the
#'   final results. This argument is used only when the \code{strategy = "Ord"}
#'   (default: \code{list()})
#' @param CORES The number of cores to parallelize the prediction. Values higher
#'   than 1 require the \pkg{parallel} package (default: 1).
#'
#' @return A matrix containing the probabilistic values or just predictions (only when
#'   \code{probability = FALSE}). The rows indicate the predicted object and the
#'   columns indicate the labels.
#'
#' @references
#'  Cherman, E. A., Metz, J., & Monard, M. C. (2012). Incorporating label
#'    dependency into the binary relevance framework for multi-label
#'    classification. Expert Systems with Applications, 39(2), 1647–1655.
#'
#' @seealso \code{\link[=brplus]{BR+}}
#' @export
#'
#' @examples
#' #' library(utiml)
#'
#' # Emotion multi-label dataset using BR+
#' testdata <- emotions$dataset[sample(1:100, 10), emotions$attributesIndexes]
#'
#' # Predict SVM scores
#' model <- brplus(emotions)
#' pred <- predict(model, testdata)
#'
#' # Predict SVM bipartitions and change the method to use No Update strategy
#' pred <- predict(model, testdata, strategy = "NU", probability = FALSE)
#'
#' # Predict using a random sequence to update the labels
#' labels <- sample(rownames(emotions$labels))
#' pred <- predict(model, testdata, strategy = "Ord", order = labels)
#'
#' # Passing a specif parameter for SVM predict method
#' pred <- predict(model, testdata, na.action = na.fail)
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

  newdata <- utiml_newdata(newdata)

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
  cat("\n", length(x$models), "Models (labels):\n")
  print(names(x$models))
}

