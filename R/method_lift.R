#' LIFT for multi-label Classification
#'
#' Create a multi-label learning with Label specIfic FeaTures (LIFT) model.
#'
#' LIFT firstly constructs features specific to each label by conducting
#' clustering analysis on its positive and negative instances, and then performs
#' training and testing by querying the clustering results.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param ratio Controll the number of clusters being retained. Must be between
#'  0 and 1. (Default: \code{0.1})
#' @param ... Others arguments passed to the base method for all subproblems
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. This is useful when
#'  the method is run in parallel. (Default: \code{options("utiml.seed", NA)})
#' @return An object of class \code{LIFTmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{models}{A list of the generated models, named by the label names.}
#'   }
#' @references
#'  Zhang, M.-L., & Wu, L. (2015). Lift: Multi-Label Learning with
#'  Label-Specific Features. IEEE Transactions on Pattern Analysis and Machine
#'  Intelligence, 37(1), 107-120.
#' @export
#'
#' @examples
#' model <- lift(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' # Runing lift with a specific ratio
#' model <- lift(toyml, "RF", 0.15)
#' }
lift <- function(mdata, base.method = getOption("utiml.base.method", "SVM"),
               ratio = 0.1, ..., cores = getOption("utiml.cores", 1),
               seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  if (ratio < 0 || ratio > 1) {
    stop("The attribbute ratio must be between 0 and 1")
  }

  utiml_preserve_seed()

  # LIFT Model class
  liftmodel <- list(labels = rownames(mdata$labels),
                    ratio = ratio, call = match.call())

  # Create models
  labels <- utiml_rename(liftmodel$labels)
  liftdata <- utiml_lapply(labels, function (label) {
    #Form Pk and Nk based on D according to Eq.(1)
    Pk <- mdata$dataset[,label] == 1
    Nk <- !Pk

    #Perform k-means on Pk and Nk, each with mk clusters as defined in Eq.(2)
    mk <- ceiling(ratio * min(sum(Pk), sum(Nk)))

    gpk <- stats::kmeans(mdata$dataset[Pk, mdata$attributesIndexes], mk)
    gnk <- stats::kmeans(mdata$dataset[Nk, mdata$attributesIndexes], mk)

    centroids <- rbind(gpk$centers, gnk$centers)
    rownames(centroids) <- c(paste("p", rownames(gpk$centers), sep=''),
                             paste("n", rownames(gnk$centers), sep=''))

    #Create the mapping φk for lk according to Eq.(3);
    rows <- seq(mdata$measures$num.instances)
    dataset <- do.call(rbind, lapply(rows, function (inst){
      instance <- mdata$dataset[inst, ]
      instancedata <- instance[mdata$attributesIndexes]
      ninst <- apply(centroids, 1, function (group) {
        stats::dist(rbind(group, instancedata))
      })
      data.frame(c(ninst, instance[label]))
    }))

    #Induce the model using the base algorithm
    model <- utiml_create_model(
      utiml_prepare_data(dataset, "mldLIFT", mdata$name, "lift", base.method),
      ...
    )

    rm(dataset)
    list(
      centroids = centroids,
      model = model
    )
  }, cores, seed)

  liftmodel$centroids <- lapply(liftdata, function (x) x$centroids)
  liftmodel$models <- lapply(liftdata, function (x) x$model)

  utiml_restore_seed()

  class(liftmodel) <- "LIFTmodel"
  liftmodel
}

#' Predict Method for LIFT
#'
#' This function predicts values based upon a model trained by
#' \code{\link{lift}}.
#'
#' @param object Object of class '\code{LIFTmodel}'.
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
#' @seealso \code{\link[=lift]{LIFT}}
#' @export
#'
#' @examples
#' model <- lift(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.LIFTmodel <- function(object, newdata,
                            probability = getOption("utiml.use.probs", TRUE),
                            ..., cores = getOption("utiml.cores", 1),
                            seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "LIFTmodel") {
    stop("First argument must be an BRmodel object")
  }

  if (cores < 1) {
    stop("Cores must be a positive value")
  }

  utiml_preserve_seed()

  # Predict models
  newdata <- utiml_newdata(newdata)
  labels <- utiml_rename(object$labels)
  predictions <- utiml_lapply(labels, function (label) {
    centroids <- object$centroids[[label]]
    rows <- seq(nrow(newdata))
    dataset <- do.call(rbind, lapply(rows, function (inst){
      instancedata <- newdata[inst, ]
      apply(centroids, 1, function (group) {
        stats::dist(rbind(group, instancedata))
      })
    }))
    rownames(dataset) <- rownames(newdata)
    utiml_predict_binary_model(object$models[[label]], dataset, ...)
  }, cores, seed)

  utiml_restore_seed()

  utiml_predict(predictions, probability)
}

#' Print BR model
#' @param x The br model
#' @param ... ignored
#' @export
print.LIFTmodel <- function(x, ...) {
  cat("LIFT Model\n\nCall:\n")
  print(x$call)
  cat("\nRatio:", x$ratio, "\n")
  cat("\n", length(x$labels), "Binary Models:\n")
  overview <- as.data.frame(cbind(label=names(x$centroids),
                                  attrs=unlist(lapply(x$centroids, nrow))))
  rownames(overview) <- NULL
  print(overview)
}
