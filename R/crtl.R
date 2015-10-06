crtl <- function (mdata,
                  base.method = "SVM",
                  m = 5,
                  validation.size = 0.33,
                  validation.threshold = 0.3,
                  ...,
                  predict.params = list(),
                  SEED = NULL,
                  CORES = 1) {
  #Validations
  if (!requireNamespace("FSelector", quietly = TRUE))
    stop('There are no installed package "FSelector" to use CRTL multi-label classifier')

  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if(m <= 1)
    stop('The number of iterations (m) must be greater than 1')

  if (validation.size < 0.1 || validation.size > 0.6)
    stop("The validation size must be between 0.1 and 0.6")

  if (validation.threshold < 0 || validation.threshold > 1)
    stop("The validation size must be between 0 and 1")

  if (CORES < 1)
    stop('Cores must be a positive value')

  #BR Model class
  crtlmodel <- list()
  crtlmodel$rounds <- m
  crtlmodel$validation.size <- validation.size
  crtlmodel$validation.threshold <- validation.threshold

  #Step1 - Split validation data, train and evaluation using F1 measure (1-5)
  validation.set <- mldr_stratified_holdout(mdata, c(1 - validation.size, validation.size), SEED)
  validation.model <- br(validation.set[[1]], base.method = base.method, ..., CORES = CORES)
  params <- list(object = validation.model, newdata = validation.set[[2]], probability = FALSE, CORES = CORES)
  validation.prediction <- do.call(predict, c(params, predict.params))
  validation.result <- utiml_measure_labels(validation.set[[2]], validation.prediction, utiml_measure_recall)
  Yc <- names(which(validation.result >= validation.threshold))
  crtlmodel$Y <- Yc

  #Step2 - Identify close-related labels within Yc using feature selection technique (6-10)
  classes <- mdata$dataset[mdata$labels$index][,Yc]
  Rj <- utiml_lapply(rownames(mdata$labels), function (labelname) {
    formula <- as.formula(paste("`", labelname, "` ~ .", sep=""))
    Aj <- mdata$dataset[mdata$labels$index][,unique(c(Yc, labelname))]
    weights <- FSelector::relief(formula, Aj)
    FSelector::cutoff.k(weights, m)
  }, CORES)
  names(Rj) <- rownames(mdata$labels)
  crtlmodel$R <- Rj

  #Build models (11-17)
  D <- mdata$dataset[mdata$attributesIndexes]
  crtlmodel$models <- utiml_lapply(rownames(mdata$labels), function (labelname) {
    Di <- br.transformation(cbind(D, mdata$dataset[labelname]), "mldBR", base.method)
    fi <- list(br.create_model(Di, ...))
    for (k in Rj[[labelname]]) {
      Di <- br.transformation(cbind(D, mdata$dataset[k], mdata$dataset[labelname]), "mldBR", base.method)
      fi <- c(fi, list(br.create_model(Di, ...)))
    }
    names(fi) <- c(labelname, Rj[[labelname]])
    fi
  }, CORES)
  names(crtlmodel$models) <- rownames(mdata$labels)

  crtlmodel$call <- match.call()
  class(crtlmodel) <- "CRTLmodel"

  crtlmodel
}

predict.CRTLmodel <- function (object,
                               newdata,
                               ...,
                               probability = TRUE,
                               CORES = 1
) {
  #Validations
  if(class(object) != 'CRTLmodel')
    stop('First argument must be an CRTLmodel object')

  if (CORES < 1)
    stop('Cores must be a positive value')

  newdata <- utiml_newdata(newdata)

  #Predict initial values
  predictions <- utiml_lapply(object$models, function (models){
    br.predict_model(models[[1]], newdata, ...)
  }, CORES)
  fjk <- as.data.frame(as.multilabelPrediction(predictions, FALSE))

  #Predict ensemble values
  allpreds <- utiml_lapply(object$models, function (models){
    preds <- list()
    for (labels in names(models)[-1])
      preds[[labels]] <- br.predict_model(models[[labels]], cbind(newdata, fjk[labels]), ...)

    preds
  }, CORES)

  #Compute votes using "majority vote" scheme (when there are a tie we use the scores values)

  browser()
}


print.CRTLmodel <- function (x, ...) {
  cat("BR with ConTRolled Label correlation Model\n\nCall:\n")
  print(x$call)
  cat("\nDetails:")
  cat("\n ", x$rounds, "Iterations")
  cat("\n ", 1- x$validation.size, "/", x$validation.size, "train/validation size")
  cat("\n ", x$validation.threshold, "Threshold value")
  if (!is.null(x$seed))
    cat("\nSeed value:", x$seed)

  cat("\n\nPruned Labels:", length(x$Y), "\n  ")
  cat(x$Y, sep = ", ")
}
