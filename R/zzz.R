.onLoad <- function(libname, pkgname) {
  op <- options()
  op.utiml <- list(
    utiml.base.method = "SVM",
    utiml.cores = 1,
    utiml.seed = NA,
    utiml.use.probs = TRUE,
    utiml.empty.prediction = FALSE,
    utiml.mldr_from_dataframe = mldr_from_dataframe
  )
  toset <- !(names(op.utiml) %in% names(op))
  if (any(toset)) options(op.utiml[toset])

  unlockBinding("mldr_from_dataframe", as.environment("package:mldr"))
  assign("mldr_from_dataframe", utiml_from_dataframe, "package:mldr")

  unlockBinding("mldr_from_dataframe", getNamespace("mldr"))
  assign("mldr_from_dataframe", utiml_from_dataframe, getNamespace("mldr"))

  invisible()
}
