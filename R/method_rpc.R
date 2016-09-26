#' Ranking by pairwise comparison (RPC) for multi-label Classification
#'
#' Create a RPC model for multilabel classification.
#'
#' RPC is a simple transformation method that uses pairwise classification to
#' predict multi-label data. This is based on the one-versus-one approach to
#' build a specific model for each label combination.
#'
#' @family Transformation methods
#' @family Pairwise methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param ... Others arguments passed to the base method for all subproblems
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{RPCmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{models}{A list of the generated models, named by the label names.}
#'   }
#' @references
#'  Hullermeier, E., Furnkranz, J., Cheng, W., & Brinker, K. (2008).
#'  Label ranking by learning pairwise preferences. Artificial Intelligence,
#'  172(16-17), 1897-1916.
#' @export
#'
#' @examples
#' model <- rpc(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' }
rpc <- function(mdata, base.method = getOption("utiml.base.method", "SVM"), ...,
               cores = getOption("utiml.cores", 1),
               seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # RPC Model class
  rpcmodel <- list(labels = rownames(mdata$labels), call = match.call())

  # Create models
  labels <- utils::combn(rpcmodel$labels, 2, simplify=FALSE)
  names(labels) <- unlist(lapply(labels, paste, collapse=','))
  rpcmodel$models <- utiml_lapply(labels, function (pairwise) {
    utiml_create_model(
      utiml_prepare_data(
        utiml_create_pairwise_data(mdata, pairwise[1], pairwise[2]),
        "mldRPC", mdata$name, "rpc", base.method,
        label1=pairwise[1], label2=pairwise[2]
      ), ...
    )
  }, cores, seed)

  utiml_restore_seed()

  class(rpcmodel) <- "RPCmodel"
  rpcmodel
}

#' Predict Method for RPC
#'
#' This function predicts values based upon a model trained by
#' \code{\link{rpc}}.
#'
#' @param object Object of class '\code{RPCmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=br]{Binary Relevance (BR)}}
#' @export
#'
#' @examples
#' model <- rpc(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' }
predict.RPCmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "RPCmodel") {
    stop("First argument must be an BRmodel object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # Create models
  newdata <- utiml_newdata(newdata)
  labels <- utiml_rename(object$labels)
  predictions <- utiml_lapply(object$models, utiml_predict_binary_model,
                              newdata = newdata, ..., cores, seed)

  utiml_restore_seed()

  # Compute votes
  labels <- utils::combn(object$labels, 2, simplify=FALSE)
  votes <- matrix(0, ncol=length(object$labels), nrow=nrow(newdata),
                  dimnames = list(rownames(newdata), object$labels))
  for (i in seq(labels)) {
    votes[,labels[[i]]] <- votes[,labels[[i]]] +
      cbind(predictions[[i]]$bipartition, 1 - predictions[[i]]$bipartition)
  }

  as.mlresult(votes / length(object$labels), probability)
}

#' Print RPC model
#' @param x The br model
#' @param ... ignored
#' @export
print.RPCmodel <- function(x, ...) {
  cat("RPC Model\n\nCall:\n")
  print(x$call)
  cat("\n", length(x$models), " pairwise models\n", sep='')}
