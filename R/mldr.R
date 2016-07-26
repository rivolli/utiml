#' Fix the mldr dataset to use factors
#'
#' @param mdata A mldr dataset.
#'
#' @return A mldr object
#' @export
#'
#' @examples
#' toyml <- mldata(toyml)
mldata <- function (mdata) {
  # Change character attributes to factors
  attrs <- which(
    sapply(mdata$dataset[, mdata$attributesIndexes], class) == "character"
  )
  mdata$dataset[,attrs] <- as.data.frame(
    apply(mdata$dataset[, attrs, drop=FALSE], 2, as.factor)
  )

  mdata
}
