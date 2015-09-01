#' Create a subset of a dataset
#'
#' @param mdata A \code{mldr} objetc dataset
#' @param rows A vector with the row indexes (instance indexes)
#' @param cols A vector with the column indexes (attribute indexes)
#'
#' @return A new mldr subset
#' @export
#'
#' @examples
#' #Return a emotion subset with the first 40 examples and only with two attributes
#' rows <- 1:40
#' cols <- c("Mean_Acc1298_Mean_Mem40_Centroid", "Mean_Acc1298_Mean_Mem40_MFCC_10")
#' mldr_subset(emotions, rows, cols)
mldr_subset <- function (mdata, rows, cols) {
  mldr_from_dataframe(
    cbind(mdata$dataset[rows, cols], mdata$dataset[rows, rownames(mdata$labels)]),
    labelIndices = (length(cols) + 1):(length(cols)+1):(length(cols)+mdata$measures$num.labels),
    name = mdata$name
  )
}

#' Create a random subset of a dataset
#'
#' @param mdata A \code{mldr} objetc dataset
#' @param num.rows The number of expected rows (instances)
#' @param num.cols The number of expected columns (attributes)
#'
#' @return A new mldr subset
#' @export
#'
#' @examples
#' #Return a emotion subset with 100 examples and 40 predictive attributes
#' mldr_random_subset(emotions, 100, 40)
mldr_random_subset <- function (mdata, num.rows, num.cols) {
  rows <- sample(mdata$measures$num.instances, num.rows)
  cols <- sample(mdata$attributesIndexes, num.cols)
  mldr_subset(mdata, rows, cols)
}

#' Create a Binary MultiLabel Data
#'
#' @param dataset A data.frame with the data (the last column must be the class column)
#' @param classname The name of specific class of the object
#' @param base.method The name of the base method that will process this dataset
#'
#' @return A list with data, labelname, labelindex and methodname.
#'    This list has three classes: mltransformation, baseMETHODNAME and a specific name
#' @export
#'
#' @examples
#' ...
#' tbl <- binary_transformation(dataframe, "mldBR", "SVM")
#' ...
binary_transformation <- function (dataset, classname, base.method) {
  label <- colnames(dataset)[length(dataset)]

  #Convert the class column as factor
  dataset[,label] <- as.factor(dataset[,label])

  #Create data
  dataset <- list(data = dataset, labelname = label, labelindex = ncol(dataset), methodname = base.method)
  class(dataset) <- c(classname, paste("base", base.method, sep=''), "mltransformation")

  dataset
}

utiml_normalize <- function (data, max.val=NULL, min.val=NULL) {
  if (is.null(max.val))
    max.val <- max(data)
  if (is.null(min.val))
    min.val <- min(data)
  (data-min.val)/(max.val-min.val)
}
