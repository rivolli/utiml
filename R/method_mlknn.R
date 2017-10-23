#' Multi-label KNN (ML-KNN) for multi-label Classification
#'
#' Create a ML-KNN classifier to predict multi-label data. It is a multi-label
#' lazy learning, which is derived from the traditional K-nearest neighbor (KNN)
#' algorithm. For each unseen instance, its K nearest neighbors in the training
#' set are identified and based on statistical information gained from the label
#' sets of these neighboring instances, the maximum a posteriori (MAP) principle
#' is utilized to determine the label set for the unseen instance.
#'
#' @family Adaptatio methods
#' @param mdata A mldr dataset used to train the binary models.
#' @param k The number of neighbors. (Default: \code{10})
#' @param s Smoothing parameter controlling the strength of uniform prior. When
#'  it is set to be 1, we have the Laplace smoothing. (Default: \code{1}).
#' @param distance The name of method used to compute the distance. See
#'  \code{\link[stats]{dist}} to the list of options.
#'  (Default: \code{"euclidian"})
#' @param ... Not used.
#' @param cores Ignored because this method does not support multi-core.
#' @param seed Ignored because this method is deterministic.
#' @return An object of class \code{MLKNNmodel} containing the set of fitted
#'  models, including:
#'  \describe{
#'    \item{labels}{A vector with the label names.}
#'    \item{prior}{The prior probability of each label to occur.}
#'    \item{posterior}{The posterior probability of each label to occur given
#'      that k neighbors have it.}
#'  }
#' @references
#'  Zhang, M.-L. L., & Zhou, Z.-H. H. (2007). ML-KNN: A lazy learning approach
#'    to multi-label learning. Pattern Recognition, 40(7), 2038–2048.
#' @export
#'
#' @examples
#' model <- mlknn(toyml, k=3)
#' pred <- predict(model, toyml)
mlknn <- function(mdata, k=10, s=1, distance="euclidean", ...,
                  cores = getOption("utiml.cores", 1),
                  seed = getOption("utiml.seed", NA)){
  # KNN Model class
  knnmodel <- list(labels = rownames(mdata$labels), call = match.call(),
                   k=k, s=s, distance=distance)

  Prior <- (s + mdata$labels$count) / (s * 2 + mdata$measures$num.instances)
  names(Prior) <- knnmodel$labels

  dm <- as.matrix(stats::dist(mdata$dataset[,mdata$attributesIndexes],
                              method=distance))
  colnames(dm) <- rownames(dm) <- seq(mdata$measures$num.instances)
  diag(dm) <- Inf
  Cx <- t(apply(dm, 1, function(dx) {
    Nx <- as.numeric(names(sort(dx)[seq(k)]))
    colSums(mdata$dataset[Nx, mdata$labels$index])
  }))

  Ck <- sapply(knnmodel$labels, function(label){
    klabel <- factor(Cx[,label], level=seq(0, k))
    has.label <- mdata$dataset[,label] == 1
    rbind(c1=table(klabel[has.label]), c0=table(klabel[!has.label]))
  }, simplify = FALSE)

  Sc <- t(do.call(rbind, lapply(Ck, rowSums)))
  Posterior <- lapply(seq(0, k), function(j){
    aux <- t(do.call(rbind, lapply(Ck, function(x) x[,j+1])))
    (s + aux) / (s * (k+1) + Sc)
  })

  knnmodel$mdata <- mdata
  knnmodel$prior <- Prior
  knnmodel$posterior <- Posterior

  class(knnmodel) <- "MLKNNmodel"
  knnmodel
}

#' Predict Method for ML-KNN
#'
#' This function predicts values based upon a model trained by \code{mlknn}.
#' '
#' @param object Object of class '\code{MLKNNmodel}'.
#' @param newdata An object containing the new input data. This must be a
#'  matrix, data.frame or a mldr object.
#' @param probability Logical indicating whether class probabilities should be
#'  returned. (Default: \code{getOption("utiml.use.probs", TRUE)})
#' @param ... Not used.
#' @param cores Ignored because this method does not support multi-core.
#' @param seed Ignored because this method is deterministic.
#' @return An object of type mlresult, based on the parameter probability.
#' @seealso \code{\link[=mlknn]{ML-KNN}}
#' @export
#'
#' @examples
#' model <- mlknn(toyml)
#' pred <- predict(model, toyml)
predict.MLKNNmodel <- function(object, newdata,
                               probability = getOption("utiml.use.probs", TRUE),
                               ..., cores = getOption("utiml.cores", 1),
                               seed = getOption("utiml.seed", NA)) {
  # Validations
  if (class(object) != "MLKNNmodel") {
    stop("First argument must be an MLKNNmodel object")
  }

  newdata <- utiml_newdata(newdata)
  train.data <- object$mdata$dataset[,object$mdata$attributesIndexes]
  train.labels <- object$mdata$dataset[,object$mdata$labels$index]

  Cx <- t(apply(newdata, 1, function(test.inst){
    dx <- apply(train.data, 1, function(train.inst){
      stats::dist(rbind(test.inst, train.inst), method=object$distance)
    })
    names(dx) <- seq(length(dx))

    Nx <- as.numeric(names(sort(dx)[seq(object$k)]))
    colSums(train.labels[Nx,])
  }))

  predictions <- sapply(object$labels, function(label){
    prior <- c(object$prior[label], 1-object$prior[label])
    names(prior) <- c("c1","c0")

    probs <- sapply(object$posterior[Cx[,label] + 1], function(item){
      item[,label]
    }) * prior

    bipartition <- abs(apply(probs , 2, which.max) - 2)
    probability <- probs[1,] / colSums(probs)

    names(bipartition) <- names(probability) <- rownames(newdata)
    utiml_binary_prediction(bipartition, probability)
  }, simplify = FALSE)

  utiml_predict(predictions, probability)
}

#' Print MLKNN model
#' @param x The mlknn model
#' @param ... ignored
#' @export
print.MLKNNmodel <- function(x, ...) {
  cat("Classifier MLKNN\n\nCall:\n")
  print(x$call)
  cat("\nk = ", k, "\nPrior positive probabilits:\n")
  print(x$prior)
}

