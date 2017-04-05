.onLoad <- function(libname, pkgname) {
  op <- options()
  op.utiml <- list(
    utiml.base.algorithm = "SVM",
    utiml.cores = 1,
    utiml.seed = NA,
    utiml.use.probs = TRUE,
    utiml.empty.prediction = FALSE
  )
  toset <- !(names(op.utiml) %in% names(op))
  if (any(toset)) options(op.utiml[toset])

  if (!exists('.Random.seed', envir = .GlobalEnv, inherits = FALSE)) {
    sample(c()) #Force .Random.seed creation
  }

  invisible()
}
