#' Baseline reference for multilabel classification
#'
#' Create a baseline model for multilabel classification.
#'
#' Baseline is a naive multi-label classifier that maximize/minimize a specific
#' measure without induces a learning model. It uses the general information
#' about the labels in training dataset to estimate the labels in a test
#' dataset.
#'
#' The follow strategies are available:
#' \describe{
#'  \item{\code{general}}{Predict the k most frequent labels, where k is the
#'   integer most close of label cardinality.}
#'  \item{\code{F1}}{Predict the most frequent labels that obtain the best F1
#'   measure in training data. In the original paper, the authors use the less
#'   frequent labels.}
#'  \item{\code{hamming-loss}}{Predict the labels that are associated with more
#'   than 50\% of instances.}
#'  \item{\code{subset-accuracy}}{Predict the most common labelset.}
#'  \item{\code{ranking-loss}}{Predict a ranking based on the most frequent
#'   labels.}
#' }
#'
#' @param mdata A mldr dataset used to train the binary models.
#' @param metric Define the strategy used to predict the labels.
#'
#'  The possible values are: \code{'general'}, \code{'F1'},
#'  \code{'hamming-loss'} or \code{'subset-accuracy'}. See the description
#'  for more details. (Default: \code{'general'}).
#' @param ... not used
#' @return An object of class \code{BASELINEmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{predict}{A list with the labels that will be predicted.}
#'   }
#' @references
#'  Metz, J., Abreu, L. F. de, Cherman, E. A., & Monard, M. C. (2012). On the
#'  Estimation of Predictive Evaluation Measure Baselines for Multi-label
#'  Learning. In 13th Ibero-American Conference on AI (pp. 189-198).
#'  Cartagena de Indias, Colombia.
#' @export
#'
#' @examples
#' model <- baseline(toyml)
#' pred <- predict(model, toyml)
#'
#' ## Change the metric
#' model <- baseline(toyml, "F1")
#' model <- baseline(toyml, "subset-accuracy")
baseline <- function (mdata, metric = c("general", "F1", "hamming-loss",
                      "subset-accuracy", "ranking-loss"), ...) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  metric <- match.arg(metric)
  basefnc <- switch (metric,
    "F1" = function (mdata){
      labels <- order(mdata$labels$freq, decreasing = TRUE)
      lbl <- which.max(lapply(seq(mdata$measures$num.labels), function (i, labels){
        Zm <- matrix(0,
                     nrow=mdata$measures$num.instances,
                     ncol=mdata$measures$num.labels)
        colnames(Zm) <- rownames(mdata$labels)
        Zm[,labels[seq(i)]] <- 1
        multilabel_evaluate(mdata, as.mlresult(Zm), "F1")
      }, labels=labels))
      rownames(mdata$labels)[labels[seq(lbl)]]
    },
    "general" = function (mdata){
      freq <- order(mdata$labels$freq, decreasing=TRUE)
      rownames(mdata$labels)[freq[seq(round(mdata$measures$cardinality,0))]]
    },
    "hamming-loss" = function (mdata){
      if (any(mdata$labels$freq > 0.5)) {
        rownames(mdata$labels)[mdata$labels$freq > 0.5]
      } else {
        #Avoid empty predictions, recommend only the most frequent label
        rownames(mdata$labels[order(mdata$labels$freq, decreasing=TRUE),])[1]
      }
    },
    "subset-accuracy" = function (mdata){
      lbl <- as.numeric(unlist(strsplit(names(which.max(mdata$labelsets)), "")))
      rownames(mdata$labels)[lbl == 1]
    },
    "ranking-loss" = function (mdata) {
      rk <- order(mdata$labels$freq, decreasing=TRUE)
      half <- mdata$labels$freq / 2
      half + (0.49 - max(half[-rk[seq(round(mdata$measures$cardinality,0))]]))
    }
  )

  blmodel <- list(
    labels = rownames(mdata$labels),
    metric = metric,
    predict = basefnc(mdata),
    call = match.call()
  )
  class(blmodel) <- "BASELINEmodel"
  blmodel
}

#' Predict Method for BASELINE
#'
#' This function predicts values based upon a model trained by
#' \code{\link{baseline}}.
#'
#' @param object Object of class '\code{BASELINEmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... not used.
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=baseline]{Baseline}}
#' @export
#'
#' @examples
#' model <- baseline(toyml)
#' pred <- predict(model, toyml)
predict.BASELINEmodel <- function (object, newdata,
                        probability = getOption("utiml.use.probs", TRUE), ...){
  # Validations
  if (class(object) != "BASELINEmodel") {
    stop("First argument must be a BASELINEmodel object")
  }

  newdata <- utiml_newdata(newdata)

  if (mode(object$predict) == "numeric") {
    prediction <- matrix(rep(object$predict, nrow(newdata)), byrow = TRUE,
                         nrow=nrow(newdata), ncol=length(object$labels),
                         dimnames = list(rownames(newdata), object$labels))
  } else {
    prediction <- matrix(0, nrow=nrow(newdata), ncol=length(object$labels),
                         dimnames = list(rownames(newdata), object$labels))
    prediction[, object$predict] <- 1
  }

  as.mlresult(prediction, probability = probability)
}
