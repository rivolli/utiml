#' Ensemble of Single Label
#'
#' Create an Ensemble of Single Label model for multilabel classification.
#'
#' ESL is an ensemble of multi-class model that uses the less frequent labels.
#' This is based on the label ignore approach different members of the ensemble.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param m The number of members used in the ensemble. (Default: 10)
#' @param w The weight given to the choice of the less frequent labels. When it
#'  is 0, the labels will be random choose, when it is 1 the complement of the
#'  label frequency is used as the probability to choose each label. Values
#'  greater than 1 will privilege the less frequent labels. (Default: 1)
#' @param ... Others arguments passed to the base algorithm for all subproblems
#' @param cores The number of cores to parallelize the training. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{ESLmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the labels' frequencies.}
#'    \item{models}{A list of the multi-class models.}
#'   }
#'
#' @export
#'
#' @examples
#' model <- esl(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \donttest{
#' # Use SVM as base algorithm
#' model <- esl(toyml, "SVM")
#' pred <- predict(model, toyml)
#'
#' # Change the base algorithm and use 2 CORES
#' model <- esl(toyml[1:50], 'RF', cores = 2, seed = 123)
#'
#' # Set a parameters for all subproblems
#' model <- esl(toyml, 'KNN', k=5)
#' }
esl <- function(mdata,
                base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                m=10, w=1, ..., cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(mdata, "mldr")) {
    stop("First argument must be an mldr object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  freqs <- mdata$labels$freq
  labels <- rownames(mdata$labels)
  names(freqs) <- labels

  # ESL Model class
  eslmodel <- list(labels = labels, call = match.call(), m=m, w=w)

  # Create models
  eslmodel$models <- utiml_lapply(seq(m), function (i){
    Class.values <- apply(mdata$dataset[,labels], 1, function(row) {
      bips <- which(row == 1)
      names(which.max(((1 - freqs[bips])*w) + stats::runif(length(bips))))
    })
    train <- cbind(mdata$dataset[,mdata$attributesIndexes], Class=Class.values)

    utiml_create_model(
      utiml_prepare_data(train, "mldSL", mdata$name, "esl", base.algorithm),
      ...
    )
  }, cores, seed)

  class(eslmodel) <- "ESLmodel"
  eslmodel
}

#' Predict Method for Ensemble of Single Label
#'
#' This function predicts values based upon a model trained by
#' \code{\link{esl}}.
#'
#' @param object Object of class '\code{ESLmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=esl]{Ensemble of Single Label (ESL)}}
#' @export
#'
#' @examples
#' model <- esl(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.ESLmodel <- function(object, newdata,
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., cores = getOption("utiml.cores", 1),
                             seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "ESLmodel")) {
    stop("First argument must be an ESLmodel object")
  }

  newdata <- utiml_newdata(newdata)

  labels <- object$labels

  votes <- do.call(cbind, utiml_lapply(object$models, function(model){
    #TODO refactory it
    #TODO use probability
    pred <- do.call(mlpredict, c(list(model = model, newdata = newdata), ...))
    #pred <- predict(model, newdata) #, type = "prob"
    as.character(pred[,"prediction"])
  }, cores, seed))

  rownames(votes) <- rownames(newdata)
  probs <- t(apply(votes, 1, function(x){
    row <- rep(0, length(labels))
    names(row) <- labels
    vt <- table(x)/object$m
    row[names(vt)] <- vt
    row
  }))

  fixed_threshold(probs, 0.0001, probability)
}

#' Print ESL model
#' @param x The esl model
#' @param ... ignored
#'
#' @return No return value, called for print model's detail
#'
#' @export
print.ESLmodel <- function(x, ...) {
  cat("Ensemble of Single Label Model\n\nCall:\n")
  print(x$call)
  cat("\n", x$m, "Models")
  cat("\n", x$w, "is the weight for the less frequent labels\n")
}

