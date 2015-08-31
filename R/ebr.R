ebr <- function (mdata,
                base.method = "SVM",
                m = 10,
                subsample = 0.75,
                attr.space = 0.5,
                ...,
                save.datasets = FALSE,
                SEED = -1,
                CORES = 1
) {
  #Validations
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if(m <= 1)
    stop('The number of iterations (m) must be greater than 1')

  if (subsample < 0.1 || subsample > 1)
    stop("The subset of training instances must be between 0.1 and 1 inclusive")

  if (attr.space <= 0.1 || attr.space > 1)
    stop("The attribbute space of training instances must be between 0.1 and 1 inclusive")

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BR Model class
  ebrmodel <- list()
  ebrmodel$rounds <- m
  ebrmodel$nrow <- ceiling(mdata$measures$num.instances * subsample)
  ebrmodel$ncol <- ceiling(length(mdata$attributesIndexes) * attr.space)

  if (SEED > 0) {
    ebrmodel$seed <- SEED
    set.seed(SEED)
  }

  ebrmodel$models <- lapply(1:m, function (iteration){
    ndata <- mldr_random_subset(mdata, ebrmodel$nrow, ebrmodel$ncol)
    br(ndata, base.method, ..., save.datasets = save.datasets, CORES = CORES)
  })

  ebrmodel$call <- match.call()
  class(ebrmodel) <- "EBRmodel"

  if (SEED > 0) {
    set.seed(NULL)
  }
  ebrmodel
}

predict.EBRmodel <- function (object,
                             newdata,
                             vote.schema = c("score", "majority", "prop"),
                             ...,
                             probability = TRUE,
                             CORES = 1
) {
  #Validations
  if(class(object) != 'EBRmodel')
    stop('First argument must be an EBRmodel object')

  schemas <- c("score", "majority", "prop")
  if(!vote.schema[1] %in% schemas)
    stop(paste("Vote schema value must be '", paste(schemas, collapse = "' or '"), "'", sep=""))

  if (CORES < 1)
    stop('Cores must be a positive value')

  allpreds <- lapply(model$models, function (brmodel) {
    prob <- vote.schema[1] == "score"
    predict(brmodel, newdata, ..., probability = prob, CORES = CORES)
  })

  sumtable <- allpreds[[1]]
  for (i in 2:model$rounds)
    sumtable <- sumtable + allpreds[[i]]

  avgtable <- if (vote.schema[1] == "score")
    sumtable / model$rounds
  else if (vote.schema[1] == "majority")
    utiml_normalize(sumtable, model$rounds, 0)
  else
    utiml_normalize(sumtable) #proportionally

  predictions <- apply(avgtable, 2, as.resultPrediction)
  result <- if (probability)
    sapply(predictions, function (lblres) as.numeric(as.character(lblres$probability)))
  else
    sapply(predictions, function (lblres) as.numeric(as.character(lblres$bipartition)))
  rownames(result) <- names(predictions[[1]]$bipartition)

  result
}

print.EBRmodel <- function (x, ...) {
  cat("Ensemble of Binary Relevance Model\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", x$nrow, "Instances")
  cat("\n ", x$ncol, "Attributes\n")
  if (!is.null(x$seed))
    cat("\nSeed value:", x$seed)
}
