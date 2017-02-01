#' Hierarchy Of Multilabel classifiER (HOMER)
#'
#' Create a Hierarchy Of Multilabel classifiER (HOMER).
#'
#' HOMER is an algorithm for effective and computationally efficient multilabel
#' classification in domains with many labels. It constructs a hierarchy of
#' multilabel classifiers, each one dealing with a much smaller set of labels.
#'
#' @family Transformation methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param base.method A string with the name of the base method. (Default:
#'  \code{options("utiml.base.method", "SVM")})
#' @param clusters Number maximum of nodes in each level. (Default: 3)
#' @param method The strategy used to organize the labels (create the
#'  meta-labels). The options are: "balanced", "clustering" and "random".
#'    (Default: "balanced").
#' @param iteration The number max of iterations, used by balanced or clustering
#'  methods.
#' @param ... Others arguments passed to the base method for all subproblems
#' @param cores The number of cores to parallelize the training. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. (Default:
#' \code{options("utiml.seed", NA)})
#' @return An object of class \code{HOMERmodel} containing the set of fitted
#'   models, including:
#'   \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{clusters}{The number of nodes in each level}
#'    \item{models}{The Hierarchy of BR models.}
#'   }
#' @references
#'  Tsoumakas, G., Katakis, I., & Vlahavas, I. (2008). Effective and efficient
#'   multilabel classification in domains with large number of labels. In Proc.
#'   ECML/PKDD 2008 Workshop on Mining Multidimensional Data (MMD'08)
#'   (pp. 30-44). Antwerp, Belgium.
#' @export
#'
#' @examples
#' model <- homer(toyml, "RANDOM")
#' pred <- predict(model, toyml)
#'
#' \dontrun{
#' ##Change default configurations
#' model <- homer(toyml, "RF", clusters=5, method="clustering", iteration=10)
#' }
homer <- function (mdata, base.method = getOption("utiml.base.method", "SVM"),
                   clusters = 3, method = c("balanced", "clustering", "random"),
                   iteration = 100, ..., cores = getOption("utiml.cores", 1),
                   seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(mdata) != "mldr") {
    stop("First argument must be an mldr object")
  }

  if (clusters < 1) {
    stop("The number of clusters must be greater than 1")
  }

  method <- switch (match.arg(method),
    balanced = homer_balanced_kmeans,
    clustering = homer_kmeans,
    random = homer_random
  )

  # HOMER Model class
  hmodel <- list(clusters = clusters, method = method, call = match.call())
  hmodel$labels = rownames(mdata$labels)

  utiml_preserve_seed()
  if (!anyNA(seed)) {
    set.seed(seed)
  }

  hmodel$models <- buildLabelHierarchy(mdata, base.method, method, clusters,
                                       iteration, ..., cores=cores, seed=seed)

  utiml_restore_seed()
  class(hmodel) <- "HOMERmodel"
  hmodel
}

#' Predict Method for HOMER
#'
#' This function predicts values based upon a model trained by
#'  \code{\link{homer}}.
#'
#' @param object Object of class '\code{HOMERmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Others arguments passed to the base method prediction for all
#'   subproblems.
#' @param cores The number of cores to parallelize the prediction. Values higher
#'  than 1 require the \pkg{parallel} package. (Default:
#'  \code{options("utiml.cores", 1)})
#' @param seed An optional integer used to set the seed. (Default:
#'   \code{options("utiml.seed", NA)})
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=homer]{Hierarchy Of Multilabel classifiER (HOMER)}}
#' @export
#'
#' @examples
#' model <- homer(toyml, "RANDOM")
#' pred <- predict(model, toyml)
predict.HOMERmodel <- function (object, newdata,
                              probability = getOption("utiml.use.probs", TRUE),
                              ..., cores = getOption("utiml.cores", 1),
                              seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "HOMERmodel") {
    stop("First argument must be a HOMERmodel object")
  }

  previous.value <- getOption("utiml.empty.prediction")
  options(utiml.empty.prediction = FALSE)

  newdata <- utiml_newdata(newdata)
  utiml_preserve_seed()
  if (!anyNA(seed)) {
    set.seed(seed)
  }

  prediction <- predictLabelHierarchy(object$model, newdata, ...,
                                      cores=cores, seed=seed)

  utiml_restore_seed()
  options(utiml.empty.prediction = previous.value)

  as.mlresult(prediction, probability)
}

predictLabelHierarchy <- function(node, newdata, ..., cores, seed) {
  prediction <- predict.BRmodel(node$model, newdata[, node$attributes], ...)
  bipartition <- as.bipartition(prediction)
  probability <- as.probability(prediction)

  metalabel <- paste(unlist(lapply(node$metalabels, paste, collapse="*")),
                     collapse="|")

  for(i in seq(node$metalabels)) {
    labels <- node$metalabels[[i]]
    if (length(labels) > 1) {
      child <- node$children[[i]]

      indexes <- bipartition[, i, drop=FALSE] == 1
      if (any(indexes)) {
        prediction <- predictLabelHierarchy(child, newdata[indexes, ], ...,
                                            cores=cores, seed=seed)

        new.bip <- new.prob <- as.data.frame(
          matrix(0, ncol = ncol(prediction), nrow = nrow(bipartition),
          dimnames = list(rownames(bipartition), colnames(prediction)))
        )

        new.bip[indexes, colnames(prediction)] <- as.bipartition(prediction)
        bipartition <- cbind(bipartition, new.bip)

        values <- probability[!indexes, i]
        if (length(values) > 0) {
          #TODO change this if
          new.prob[!indexes, ] <- do.call(
            cbind,
            lapply(seq(ncol(new.prob)), function (j) values)
          )
        }
        new.prob[indexes, colnames(prediction)] <- as.probability(prediction)
        probability <- cbind(probability, new.prob)
      } else {
        #Predict all instances of the meta-label as negative
        aux <- do.call(cbind, lapply(labels, function(lbl)
          bipartition[, i, drop=FALSE]))
        colnames(aux) <- labels
        bipartition <- cbind(bipartition, aux)

        aux <- do.call(cbind, lapply(labels, function(lbl)
          probability[, i, drop=FALSE]))
        colnames(aux) <- labels
        probability <- cbind(probability, aux)
      }
    } else {
      #Rename the meta-label because it is the label
      colnames(bipartition)[i] <- colnames(probability)[i] <- labels
    }
  }

 #cat(metalabel, "\n")
  multilabel_prediction(
    bipartition[, node$labels, drop=F], probability[, node$labels, drop=F]
  )
}

buildLabelHierarchy <- function (mdata, base.method, method, k, it,
                                 ..., cores, seed) {
  node <- list(labels = rownames(mdata$labels), metalabels = list())

  node$metalabels <- method(mdata, k, it)

  newls <- do.call(cbind, lapply(node$metalabels, function (u){
    as.numeric(rowSums(mdata$dataset[, u, drop=FALSE]) > 0)
  }))
  colnames(newls) <- paste('meta-lbl-', seq(node$metalabels), sep='')
  rows <- which(rowSums(newls) > 0)

  #Fix meta-label without positive instances
  if (any(colSums(newls) == 0)) {
    empty.labels <- colSums(newls) == 0
    node$metalabels <- c(node$metalabels[!empty.labels],
                         unlist(node$metalabels[empty.labels]))
    newls <- do.call(cbind, lapply(node$metalabels, function (u){
      as.numeric(rowSums(mdata$dataset[, u, drop=FALSE]) > 0)
    }))
    colnames(newls) <- paste('meta-lbl-', seq(node$metalabels), sep='')
    rows <- which(rowSums(newls) > 0)
  }

  ndata <- remove_unique_attributes(mldr_from_dataframe(
    cbind(mdata$dataset[rows, mdata$attributesIndexes], newls[rows,, drop=F]),
    mdata$measures$num.inputs + seq(length(node$metalabels)),
    name = mdata$name
  ))

  mtlbl <- paste(sapply(node$metalabels, paste, collapse='*'), collapse="|")

  node$attributes <- colnames(ndata$dataset[, ndata$attributesIndexes])
  node$model <- br(ndata, base.method, ..., cores=cores, seed=seed)
  rm(ndata)

  node$children <- lapply(node$metalabels, function (metalabels) {
    if (length(metalabels) > 1) {
      excluded.label <- node$labels[!node$labels %in% metalabels]
      ndata <- remove_unlabeled_instances(remove_labels(mdata, excluded.label))
      buildLabelHierarchy(ndata, base.method, method, k, it, ...,
                          cores=cores, seed=seed)
    } else {
      NULL
    }
  })

  node
}

homer_balanced_kmeans <- function (mdata, k, it, ...) {
  if (k >= mdata$measures$num.labels) {
    return(as.list(rownames(mdata$labels)))
  }

  dataset <- t(mdata$dataset[, mdata$labels$index])
  labels <- rownames(dataset)
  Ci <- list()
  centers <- dataset[sample(labels, k), ]
  rownames(centers) <- NULL

  for (i in seq(it)) {
    ldist <- apply(dataset, 1, function (r1) {
      apply(centers, 1, function (r2) stats::dist(rbind(r1, r2)))
    })
    has.extra <- TRUE

    while(has.extra) {
      j <- apply(ldist, 2, which.min)
      Ci <- lapply(seq(k), function (i) sort(ldist[i, which(j == i)]))

      extra <- which(unlist(lapply(Ci, length)) > ceiling(length(labels)/k))
      for (i in extra) {
        ldist[i, names(Ci[[i]])[length(Ci[[i]])]] <- Inf
      }
      has.extra <- length(extra) > 0
    }

    new.centers <- do.call(rbind, lapply(Ci, function (rows) {
      colMeans(dataset[names(rows), , drop=FALSE])
    }))

    if (all(centers == new.centers)) {
      break
    }

    centers <- new.centers
  }

  lapply(Ci, names)
}

homer_kmeans <- function (mdata, k, it, ...) {
  if (k >= mdata$measures$num.labels) {
    as.list(rownames(mdata$labels))
  } else {
    clusters <- stats::kmeans(t(mdata$dataset[, mdata$labels$index]),
                              k, iter.max = it)
    split(rownames(mdata$labels), clusters$cluster)
  }
}

homer_random <- function (mdata, k, ...) {
  split(sample(rownames(mdata$labels)),
        rep_len(seq(k), mdata$measures$num.labels))
}
