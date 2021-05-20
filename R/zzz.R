.onLoad <- function(libname, pkgname) {
  op <- options()
  op.utiml <- list(
    utiml.base.algorithm = "SVM",
    utiml.cores = 1,
    utiml.seed = NA,
    utiml.use.probs = TRUE,
    utiml.empty.prediction = FALSE,
    utiml.random = sample(1:10) #Random value
  )
  toset <- !(names(op.utiml) %in% names(op))
  if (any(toset)) options(op.utiml[toset])

  invisible()
}
