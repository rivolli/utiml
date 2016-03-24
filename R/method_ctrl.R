#' CTRL model for multi-label Classification
#'
#' Create a binary relevance with ConTRolled Label correlation exploitation
#' (CTRL) model for multilabel classification.
#'
#' CTRL employs a two-stage filtering procedure to exploit label correlations
#' in a controlled manner. In the first stage, error-prone class labels are
#' pruned from Y to generate the candidate label set for correlation
#' exploitation. In the second stage, classification models are built for each
#' class label by exploiting its closely-related labels in the candidate
#' label set.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param m The max number of Binary Relevance models used in the binary
#'  ensemble. (Default: 5)
#' @param validation.size The size of validation set, used internally to prunes
#'  error-prone class labels. The value must be between 0.1 and 0.5.
#'  (Default: 0.3)
#' @param validation.threshold Thresholding parameter determining whether any
#'  class label in Y is regarded as error-prone or not. (Default: 0.3)
#' @param ... Others arguments passed to the base method for all subproblems
#' @param predict.params A list of default arguments passed to the predictor
#'  method. (default: \code{list()})
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of class \code{CTRLmodel} containing the set of fitted
#'   models, including: \describe{
#'   \item{rounds}{The value passed in the m parameter}
#'   \item{validation.size}{The value passed in the validation.size parameter}
#'   \item{validation.threshold}{The value passed in the validation.threshold
#'      parameter}
#'   \item{Y}{Name of labels less susceptible to error, according to the
#'      validation process}
#'   \item{R}{List of close-related labels related with Y obtained by using
#'      feature selection technique}
#'   \item{models}{A list of the generated models, for each label a list of
#'      models was built based on close-related labels.}
#' }
#' @details Dependencies:
#'  The degree of label correlations are estimated via supervised feature
#'    selection techniques. Thus, this implementation use the
#'    \link[FSelector]{relief} method available in \pkg{FSelector} package.
#' @references
#'  Li, Y., & Zhang, M. (2014). Enhancing Binary Relevance for Multi-label
#'    Learning with Controlled Label Correlations Exploitation. In 13th Pacific
#'    Rim International Conference on Artificial Intelligence (pp. 91-103).
#'    Gold Coast, Australia.
#' @export
#'
#' @examples
#' \dontrun{
#' # Use SVM as base method
#' model <- ctrl(toyml)
#' pred <- predict(model, toyml)
#'
#' # Change default values and use 4 CORES
#' model <- ctrl(toyml, 'C5.0', m = 10, validation.size = 0.4,
#'               validation.threshold = 0.5, CORES = 4)
#'
#' # Set a parameters for all subproblems
#' model <- ctrl(dataset$train, 'KNN', k=5)
#' }
ctrl <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
                 m = 5, validation.size = 0.3, validation.threshold = 0.3, ...,
                 predict.params = list(), CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (!requireNamespace("FSelector", quietly = TRUE)) {
    stop(paste("There are no installed package 'FSelector' to use CTRL",
               "multi-label classifier"))
  }

  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (m <= 1) {
    stop("The number of iterations (m) must be greater than 1")
  }

  if (validation.size < 0.1 || validation.size > 0.5) {
    stop("The validation size must be between 0.1 and 0.6")
  }

  if (validation.threshold < 0 || validation.threshold > 1) {
    stop("The validation size must be between 0 and 1")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  # CTRL Model class
  ctrlmodel <- list(
    rounds = m,
    validation.size = validation.size,
    validation.threshold = validation.threshold,
    call = match.call()
  )

  # Step1 - Split validation data, train and evaluation using F1 measure (1-5)
  partitions <- c(1 - validation.size, validation.size)
  validation.set <- create_holdout_partition(mdata, partitions, "iterative")
  validation.model <- br(validation.set[[1]], base.method, ..., CORES = CORES)
  params <- list(
    object      = validation.model,
    newdata     = validation.set[[2]],
    probability = FALSE,
    CORES       = CORES
  )
  validation.prediction <- do.call('predict', c(params, predict.params))
  validation.result <- utiml_measure_labels(validation.set[[2]],
                                            validation.prediction,
                                            utiml_measure_recall)
  Yc <- names(which(validation.result >= validation.threshold))
  ctrlmodel$Y <- Yc

  # Step2 - Identify close-related labels within Yc using feature selection
  #         technique (6-10)
  classes <- mdata$dataset[mdata$labels$index][, Yc]
  labels <- utiml_renames(rownames(mdata$labels))
  ctrlmodel$R <- Rj <- utiml_lapply(labels, function(labelname) {
    formula <- as.formula(paste("`", labelname, "` ~ .", sep = ""))
    cor.labels <- unique(c(Yc, labelname))
    Aj <- mdata$dataset[, mdata$labels$index, drop = F][, cor.labels, drop = F]
    if (ncol(Aj) > 1) {
      weights <- FSelector::relief(formula, Aj)
      FSelector::cutoff.k(weights, m)
    }
  }, CORES)

  # Build models (11-17)
  D <- mdata$dataset[mdata$attributesIndexes]
  ctrlmodel$models <- utiml_lapply(labels, function(labelname) {
    data  <- create_br_data(mdata, labelname)
    Di <- prepare_br_data(data, "mldBR", base.method)
    fi <- list(create_br_model(Di, ...))
    for (k in Rj[[labelname]]) {
      data <- create_br_data(mdata, labelname, mdata$dataset[k])
      Di <- prepare_br_data(data, "mldBR", base.method)
      fi <- c(fi, list(create_br_model(Di, ...)))
    }
    names(fi) <- c(labelname, Rj[[labelname]])
    fi
  }, CORES)

  class(ctrlmodel) <- "CTRLmodel"
  ctrlmodel
}

#' Predict Method for CTRL
#'
#' This function predicts values based upon a model trained by
#' \code{\link{ctrl}}.
#'
#' @param object Object of class '\code{CTRLmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param vote.schema Define the way that binary ensemble must compute the
#'  binary predictions. (Default: \code{'maj'})
#'  The default valid options are:
#'  \describe{
#'    \code{'avg'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'maj'}{Compute the averages of probabilities},
#'    \code{'max'}{Compute the votes scaled between 0 and \code{m}
#'      (number of interations)},
#'    \code{'min'}{Compute the proportion of votes, scale data between min and
#'      max of votes}
#'    \code{'prod'}{Compute the product of all votes for each instance}
#'  }
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param CORES The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=ctrl]{CTRL}}
#' @export
#'
#' @examples
#' \dontrun{
#' # Predict SVM scores
#' model <- ctrl(toyml)
#' pred <- predict(model, toyml)
#'
#' # Predict SVM bipartitions running in 6 cores
#' pred <- predict(model, toyml, probability = FALSE, CORES = 6)
#'
#' # Using the Maximum vote schema
#' pred <- predict(model, toyml, vote.schema = 'max')
#' }
predict.CTRLmodel <- function(object, newdata, vote.schema = "maj",
                              probability = getOption("utiml.use.probs", TRUE),
                              ..., CORES = getOption("utiml.cores", 1)) {
  # Validations
  if (class(object) != "CTRLmodel") {
    stop("First argument must be an CTRLmodel object")
  }

  if (CORES < 1) {
    stop("Cores must be a positive value")
  }

  check_ensemble_vote(vote.schema, accept.null = FALSE)

  newdata <- utiml_newdata(newdata)

  # Predict initial values
  initial.prediction <- utiml_lapply(object$models, function(models) {
    predict_br_model(models[[1]], newdata, ...)
  }, CORES)
  fjk <- as.matrix(as.multilabelPrediction(initial.prediction, FALSE))

  # Predict binary ensemble values
  labels <- utiml_renames(names(object$models))
  predictions <- utiml_lapply(labels, function(labelname) {
    models <- object$models[[labelname]]
    preds <- list()
    for (label in names(models)[-1]) {
      preds[[label]] <- predict_br_model(models[[label]],
                                          cbind(newdata, fjk[, label]), ...)
    }

    if (length(preds) < 1) { #No models are found, only first prediction
      initial.prediction[[labelname]]
    }
    else {
      compute_binary_ensemble_votes(preds, vote.schema)
    }
  }, CORES)

  as.multilabelPrediction(predictions, probability)
}

#' Print CTRL model
#' @param x The ctrlmodel
#' @param ... ignored
#' @export
print.CTRLmodel <- function(x, ...) {
  cat("BR with ConTRolled Label correlation Model (CTRL)\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", 1 - x$validation.size, "/", x$validation.size,
      "train/validation size")
  cat("\n ", x$validation.threshold, "Threshold value")
  cat("\n\nPruned Labels:", length(x$Y), "\n  ")
  cat(x$Y, sep = ", ")
}
