#Zhang, M.-L. L., & Zhou, Z.-H. H. (2007). ML-KNN: A lazy learning approach to
# multi-label learning. Pattern Recognition, 40(7), 2038–2048.
# http://doi.org/10.1016/j.patcog.2006.12.019
mlknn <- function(mdata, k, s=1, ..., cores = getOption("utiml.cores", 1),
                seed = getOption("utiml.seed", NA)){

  # KNN Model class
  knnmodel <- list(labels = rownames(mdata$labels), call = match.call(),
                   k=k, s=s)

  Prior <- (s + mdata$labels$count) / (s * 2 + mdata$measures$num.instances)
  names(Prior) <- knnmodel$labels

  dm <- as.matrix(dist(mdata$dataset[,mdata$attributesIndexes]))
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
      dist(rbind(test.inst, train.inst))
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

