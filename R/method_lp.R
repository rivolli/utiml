#' Label Powerset for multi-label Classification
#'
#' Create a Label Powerset model for multilabel classification.
#'
#' Label Powerset is a simple transformation method to predict multi-label data.
#' This is based on the multi-class approach to build a model where the classes
#' are each labelset.
#'
#' @family Transformation methods
#' @family Powerset
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param ... Others arguments passed to the base algorithm for all subproblems
#' @param cores Not used
#' @param seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#' @return An object of class \code{LPmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{model}{A multi-class model.}
#'   }
#' @references
#'  Boutell, M. R., Luo, J., Shen, X., & Brown, C. M. (2004). Learning
#'    multi-label scene classification. Pattern Recognition, 37(9), 1757-1771.
#' @export
#'
#' @examples
#' model <- lp(toyml, "RANDOM")
#' pred <- predict(model, toyml)
lp <- function (mdata,
                base.algorithm = getOption("utiml.base.algorithm", "SVM"), ...,
                cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  # LP Model class
  lpmodel <- list(labels = rownames(mdata$labels),
                  call = match.call(),
                  classes = mdata$labelsets)
  utiml_preserve_seed()

  lpmodel$model <- utiml_lapply(1, function (x){
    #Due the seed
    utiml_create_model(
      utiml_prepare_data(
        utiml_create_lp_data(mdata),
        "mldLP", mdata$name, "lp", base.algorithm
      ), ...
    )
  }, 1, seed)[[1]]
  utiml_restore_seed()
  class(lpmodel) <- "LPmodel"
  lpmodel
}

#' Predict Method for Label Powerset
#'
#' This function predicts values based upon a model trained by \code{\link{lp}}.
#'
#' @param object Object of class '\code{LPmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base algorithm prediction for all
#'   subproblems.
#' @param cores Not used
#' @param seed An optional integer used to set the seed. (Default:
#'   \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=lp]{Label Powerset (LP)}}
#' @export
#'
#' @examples
#' model <- lp(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.LPmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "LPmodel") {
    stop("First argument must be a LPmodel object")
  }

  newdata <- utiml_newdata(newdata)
  utiml_preserve_seed()
  result <- utiml_lapply(1, function (x){
    #Due the seed
    utiml_predict_multiclass_model(object$model, newdata, object$labels,
                                           probability, ...)
  }, 1, seed)[[1]]
  utiml_restore_seed()

  result
}

#' Print LP model
#' @param x The lp model
#' @param ... ignored
#' @export
print.LPmodel <- function(x, ...) {
  cat("Label Powerset Model\n\nCall:\n")
  print(x$call)
  cat("\n1 Model: ",length(x$classes),"classes\n")
  print(cbind.data.frame(classe=names(x$classes), instances=as.numeric(x$classes)))
}
