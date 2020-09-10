#' Pruned Problem Transformation for multi-label Classification
#'
#' Create a Pruned Problem Transformation model for multilabel classification.
#'
#' Pruned Problem Transformation (PPT) is a multi-class transformation that
#' remove the less common classes to predict multi-label data.
#'
#' @family Transformation methods
#' @family Powerset
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.algorithm A string with the name of the base algorithm. (Default:
#'  \code{options("utiml.base.algorithm", "SVM")})
#' @param p Number of instances to prune. All labelsets that occurs p times or
#'  less in the training data is removed. (Default: 3)
#' @param info.loss Logical value where \code{TRUE} means discard infrequent
#'  labelsets and \code{FALSE} means reintroduce infrequent labelsets via
#'  subsets. (Default: FALSE)
#' @param ... Others arguments passed to the base algorithm for all subproblems
#' @param cores Not used
#' @param seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#' @return An object of class \code{PPTmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{model}{A LP model contained only the most common labelsets.}
#'   }
#' @references
#'  Read, J., Pfahringer, B., & Holmes, G. (2008). Multi-label classification
#'   using ensembles of pruned sets. In Proceedings - IEEE International
#'   Conference on Data Mining, ICDM (pp. 995–1000).
#'  Read, J. (2008). A pruned problem transformation method for multi-label
#'   classification. In Proceedings of the New Zealand Computer Science
#'   Research Student Conference (pp. 143-150).
#' @export
#'
#' @examples
#' model <- ppt(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' ##Change default configurations
#' model <- ppt(toyml, "RF", p=4, info.loss=TRUE)
#' }
ppt <- function (mdata,
                 base.algorithm = getOption("utiml.base.algorithm", "SVM"),
                 p = 3, info.loss = FALSE, ...,
                 cores = getOption("utiml.cores", 1),
                 seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(mdata, "mldr")) {
    stop("First argument must be an mldr object")
  }

  if (p < 1) {
    stop("The prunning value must be greater than 0")
  }

  utiml_preserve_seed()

  # PPT Model class
  pptmodel <- list(labels = rownames(mdata$labels),
                   p = p,
                   info.loss = info.loss,
                   call = match.call())

  common.labelsets <- names(which(mdata$labelsets > p))
  if (length(common.labelsets) == 0) {
    stop(paste("All labelsets appear less than", p,
               "time(s) in the training data."))
  }
  instances <- apply(mdata$dataset[, mdata$labels$index], 1, paste, collapse='')
  original.instances <- instances %in% common.labelsets

  if (info.loss || all(original.instances)) {
    #Discard instances (infromation loss)
    ndata <- create_subset(mdata, which(original.instances))
  } else {
    #No information loss
    #TODO refactory it too ugly
    labelsets <- lapply(common.labelsets, function (x) {
      as.numeric(unlist(strsplit(x, '')))
    })
    #Sort by the number of labels and then for frequency
    labelsets <- labelsets[rev(order(unlist(lapply(labelsets, sum))))]

    rem.inst <- which(!original.instances)
    Si <- mdata$dataset[rem.inst, mdata$labels$index]
    has.match <- do.call(cbind, lapply(labelsets, function (ls) {
      colSums(ls == 1 & ls == t(Si)) == sum(ls)
    }))
    rm(Si)

    inst.lab <- lapply(lapply(split(has.match,seq(nrow(has.match))),which),
                       function (lbls){
      selected <- c()
      if (length(lbls) > 0) {
        selected <- lbls[1]
        value <- labelsets[[lbls[1]]]
        for (x in lbls[-1]) {
          the.new <- utiml_ifelse(any(value + labelsets[[x]] > 1), NULL, x)
          value <- utiml_ifelse(is.null(the.new), value, value + labelsets[[x]])
          selected <- c(selected, the.new)
        }
      }
      selected
    })
    rm(has.match)

    ndata <- merge_pruned_instances(mdata, rem.inst, inst.lab,
                                    labelsets)
  }

  pptmodel$model <- lp(ndata, base.algorithm=base.algorithm, seed=seed)

  utiml_restore_seed()
  class(pptmodel) <- "PPTmodel"
  pptmodel
}

#' Predict Method for Pruned Problem Transformation
#'
#' This function predicts values based upon a model trained by
#'  \code{\link{ppt}}.
#'
#' @param object Object of class '\code{PPTmodel}'.
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
#' @seealso \code{\link[=ppt]{Pruned Problem Transformation (PPT)}}
#' @export
#'
#' @examples
#' model <- ppt(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.PPTmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (!is(object, "PPTmodel")) {
    stop("First argument must be a PPTmodel object")
  }

  predict.LPmodel(object$model, newdata, probability, ..., seed=seed)
}

#' Print PPT model
#' @param x The ppt model
#' @param ... ignored
#' @export
print.PPTmodel <- function(x, ...) {
  cat("Pruned Problem Transformation Model\n\nCall:\n")
  print(x$call)

  cat("\nPrune:", x$p, "\n")
  cat("Information loss:", ifelse(x$info.loss, "yes", "no"), "\n")

  cat("\n1 LP Model:", length(x$model$classes), "classes\n")
  print(cbind.data.frame(classe=names(x$model$classes),
                         instances=as.numeric(x$model$classes)))
}
