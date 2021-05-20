#' Random k-labelsets for multilabel classification
#'
#' Create a RAkEL model for multilabel classification.
#'
#' RAndom k labELsets is an ensemble of LP models where each classifier is
#' trained with a small set of labels, called labelset. Two different strategies
#' for constructing the labelsets are the disjoint and overlapping labelsets.
#'
#' @family Transformation methods
#' @family Powerset
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param k The number of labels used in each labelset. (Default: \code{3})
#' @param m The number of LP models. Used when overlapping is TRUE, otherwise it
#'  is ignored. (Default: \code{2 * length(labels)})
#' @param overlapping Logical value, that defines if the method must overlapping
#'  the labelsets. If FALSE the method uses disjoint labelsets.
#'  (Default: \code{TRUE})
#' @param ... Others arguments passed to the base algorithm for all subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is running in parallel. (Default:
#'  \code{options("utiml.seed", NA)})
#' @return An object of class \code{RAkELmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{labelsets}{A list with the labelsets used to build the LP models.}
#'    \item{model}{A list of the generated models, named by the label names.}
#'   }
#' @references
#'  Tsoumakas, G., Katakis, I., & Vlahavas, I. (2011). Random k-labelsets for
#'  multilabel classification. IEEE Transactions on Knowledge and Data
#'  Engineering, 23(7), 1079-1089.
#' @export
#'
#' @examples
#' model <- rakel(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#' \donttest{
#' ## SVM using k = 4 and m = 100
#' model <- rakel(toyml, "SVM", k=4, m=100)
#'
#' ## Random Forest using disjoint labelsets
#' model <- rakel(toyml, "RF", overlapping=FALSE)
#' }
rakel <- function (mdata,
                   base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                   k = 3, m = 2 * mdata$measures$num.labels, overlapping = TRUE,
                   ..., cores = getOption("utiml.cores", 1),
                   seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(mdata, "mldr")) {
    stop("First argument must be an mldr object")
  }

  # RAkEL Model class
  rkmodel <- list(
    labels = rownames(mdata$labels),
    overlapping = overlapping,
    k = k,
    m = ifelse(overlapping, m, ceiling(mdata$measures$num.labels / k)),
    labelsets = list(),
    call = match.call()
  )

  utiml_preserve_seed()
  if (!anyNA(seed)) {
    set.seed(seed)
  }

  if (overlapping) {
    #RAkEL overllaping
    rkmodel$labelsets <- lapply(seq(rkmodel$m), function(i) {
      sample(rkmodel$labels, k)
    })

    #TODO validate if all labels are used

  } else {
    #RAkEL disjoint
    labels  <- rkmodel$labels
    for (i in seq(rkmodel$m)) {
      labelset <- sample(labels, min(k, length(labels)))
      rkmodel$labelsets[[length(rkmodel$labelsets) + 1]] <- labelset
      labels <- setdiff(labels, labelset)
    }
  }

  lbl.index <- mdata$measures$num.inputs
  rkmodel$models <- utiml_lapply(rkmodel$labelsets, function (labels) {
    data <- mldr::mldr_from_dataframe(
      cbind(mdata$dataset[mdata$attributesIndexes], mdata$dataset[labels]),
      seq(lbl.index + 1, lbl.index + length(labels)),
      name = mdata$name
    )
    lp(data, base.algorithm = base.algorithm, ...)
  }, cores, seed)

  utiml_restore_seed()
  class(rkmodel) <- "RAkELmodel"
  rkmodel
}

#' Predict Method for RAkEL
#'
#' This function predicts values based upon a model trained by
#' \code{\link{rakel}}.
#'
#' @param object Object of class '\code{RAkELmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the prediction. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=rakel]{Random k Labelsets (RAkEL)}}
#' @export
#'
#' @examples
#' model <- rakel(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.RAkELmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "RAkELmodel")) {
    stop("First argument must be a RAkELmodel object")
  }

  previous.value <- getOption("utiml.empty.prediction")
  options(utiml.empty.prediction = TRUE)

  newdata <- utiml_newdata(newdata)
  utiml_preserve_seed()

  results <- utiml_lapply(object$models, function (lpmodel){
    predict.LPmodel(lpmodel, newdata)
  }, cores, seed)

  if (object$overlapping) {
    nvotes <- as.numeric(table(unlist(object$labelsets))[object$labels])
    votes <- matrix(0, nrow=nrow(newdata), ncol=length(nvotes),
                    dimnames = list(rownames(newdata), object$labels))
    for (result in results) {
      votes[, colnames(result)] <- votes[, colnames(result)] +
        as.bipartition(result)
    }

    prediction <- as.mlresult(t(t(votes) / nvotes), probability, threshold=0.5)
    rm(votes, nvotes)
  } else {
    prediction <- multilabel_prediction(
      do.call(cbind, lapply(results, as.bipartition))[,object$labels],
      do.call(cbind, lapply(results, as.probability))[,object$labels],
      probability
    )
  }
  rm(results)

  utiml_restore_seed()
  options(utiml.empty.prediction = previous.value)

  prediction
}

#' Print RAkEL model
#' @param x The rakel model
#' @param ... ignored
#'
#' @return No return value, called for print model's detail
#'
#' @export
print.RAkELmodel <- function(x, ...) {
  cat("RAkEL",ifelse(x$overlapping, "Overlapping", "Disjoint"), "Model")
  cat("\n\nCall:\n")
  print(x$call)
  cat("\nLabelsets size:",x$k,"\n")
  cat(x$m, "LP Models. Labelsets:\n")
  print(do.call(rbind, lapply(x$labelsets, function (v) {
    length(v) <- x$k
    v
  })))
}
