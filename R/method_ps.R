#' Pruned Set for multi-label Classification
#'
#' Create a Pruned Set model for multilabel classification.
#'
#' Pruned Set (PS) is a multi-class transformation that remove the less common
#' classes to predict multi-label data.
#'
#' @family Transformation methods
#' @family Powerset
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param p Number of instances to prune. All labelsets that occurs p times or
#'  less in the training data is removed. (Default: 3)
#' @param strategy The strategy  (A or B) for processing infrequent labelsets.
#'    (Default: A).
#' @param b The number used by the strategy for processing infrequent labelsets.
#' @param ... Others arguments passed to the base method for all subproblems
#' @param cores Not used
#' @param seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#' @return An object of class \code{PSmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{model}{A LP model contained only the most common labelsets.}
#'   }
#' @references
#'  Read, J. (2008). A pruned problem transformation method for multi-label
#'  classification. In Proceedings of the New Zealand Computer Science Research
#'  Student Conference (pp. 143-150).
#' @export
#'
#' @examples
#' model <- ps(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' ##Change default configurations
#' model <- ps(toyml, "RF", p=4, strategy="B", b=4)
#' }
ps <- function (mdata, base.method = getOption("utiml.base.method", "SVM"),
                 p = 3, strategy = c("A", "B"), b = 2, ...,
                 cores = getOption("utiml.cores", 1),
                 seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (p < 1) {
    stop("The prunning value must be greater than 0")
  }

  strategy <- match.arg(strategy)

  if (b < 0) {
    stop("The parameter b must be greater or equal than 0")
  }

  utiml_preserve_seed()

  # PS Model class
  psmodel <- list(labels = rownames(mdata$labels),
                  p = p,
                  strategy = strategy,
                  b = b,
                  call = match.call())

  common.labelsets <- names(which(mdata$labelsets > p))
  instances <- apply(mdata$dataset[, mdata$labels$index], 1, paste, collapse='')
  original.instances <- instances %in% common.labelsets
  removed.instances <- which(!original.instances)

  labelsets <- lapply(common.labelsets, function (x) {
    as.numeric(unlist(strsplit(x, '')))
  })

  #Sort by the number of labels and then for frequency
  labelsets <- labelsets[rev(order(unlist(lapply(labelsets, sum))))]

  if (strategy == "B") {
    #Strategy B: use only subsets of size greater than b
    labelsets <- labelsets[unlist(lapply(labelsets, sum)) > b]
    b <- length(labelsets)
  }

  Si <- mdata$dataset[removed.instances, mdata$labels$index]
  has.match <- do.call(cbind, lapply(labelsets, function (ls) {
    colSums(ls == 1 & ls == t(Si)) == sum(ls)
  }))
  rm(Si)

  inst.lab <- lapply(
    lapply(split(has.match,seq(nrow(has.match))),which),
    function (lbls){
      utiml_ifelse(length(lbls) > 0, c(lbls[seq(min(length(lbls), b))]), c())
    }
  )
  rm(has.match)

  ndata <- merge_pruned_instances(mdata, removed.instances, inst.lab, labelsets)
  psmodel$model <- lp(ndata, base.method=base.method, seed=seed)

  utiml_restore_seed()
  class(psmodel) <- "PSmodel"

  psmodel
}

merge_pruned_instances <- function (mdata, removed.instances,
                                    inst.lab, labelsets) {
  #Remove instances without labelsets
  inst.idx <- which(unlist(lapply(inst.lab, length)) > 0)

  #Create the new labelsets data
  new.labelsets <- do.call(rbind, labelsets[unlist(inst.lab[inst.idx])])
  colnames(new.labelsets) <- rownames(mdata$labels)

  #Select the rows that will be modified
  rows <- rep(removed.instances[inst.idx],
              unlist(lapply(inst.lab[inst.idx], length)))

  mldr::mldr_from_dataframe(
    rbind(
      #Original instances
      mdata$dataset[-removed.instances,
                    c(mdata$attributesIndexes, mdata$labels$index)],

      #Rejected instances
      cbind.data.frame(
        mdata$dataset[rows, mdata$attributesIndexes], new.labelsets
      )
    ), seq(mdata$measures$num.inputs + 1, mdata$measures$num.attributes),
    name = mdata$name
  )
}

#' Predict Method for Pruned Set Transformation
#'
#' This function predicts values based upon a model trained by
#'  \code{\link{ps}}.
#'
#' @param object Object of class '\code{PSmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param cores Not used
#' @param seed An optional integer used to set the seed. (Default:
#'   \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=ps]{Pruned Set (PS)}}
#' @export
#'
#' @examples
#' model <- ps(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.PSmodel <- function(object, newdata,
                             probability = getOption("utiml.use.probs", TRUE),
                             ..., cores = getOption("utiml.cores", 1),
                             seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "PSmodel") {
    stop("First argument must be a PSmodel object")
  }

  predict.LPmodel(object$model, newdata, probability, ..., seed=seed)
}

#' Print PS model
#' @param x The ps model
#' @param ... ignored
#' @export
print.PSmodel <- function(x, ...) {
  cat("Pruned Set Model\n\nCall:\n")
  print(x$call)

  cat("\nPrune:", x$p, "\n")
  cat("Strategy:", x$strategy, "\n")
  cat("B value:", x$b, "\n")

  cat("\n1 LP Model:", length(x$model$classes), "classes\n")
  print(cbind.data.frame(classe=names(x$model$classes),
                         instances=as.numeric(x$model$classes)))
}
