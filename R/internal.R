#' Conditional value selection
#'
#' @param test an object which can be coerced to logical mode.
#' @param yes object that will be returned when the test value is true.
#' @param no object that will be returned when the test value is false
#' @return The respective value yes or no based on test value. This is an
#' alternative way to use a single logical value for avoid the real if/else for
#' choice lists, matrices and other composed data.
#'
#' @seealso \code{\link{ifelse}}
#'
#' @examples
#' utiml_ifelse(TRUE, dataframe1, dataframe2) ## dataframe1
#' utiml_ifelse(length(my.list) > 10, my.list[1:10], my.list)
utiml_ifelse <- function(test, yes, no) {
  list(yes, no)[c(test, !test)][[1]]
}

#' Select the suitable method lapply or mclaplly
#'
#' @param mylist a list to iterate.
#' @param myfnc The function to be applied to each element of the mylist.
#' @param cores The number of cores to use. If 1 use lapply oterwise use
#'    mclapply.
#' @param ... Extra arguments to myfnc.
#' @return A list with the results of the specified method.
#'
#' @examples
#' utiml_lapply(c(4,9,27), sqrt, 1) #use lapply
#' utiml_lapply(c(4,9,27), sqrt, 3) #use mclapply
utiml_lapply <- function(mylist, myfnc, cores, ...) {
  if (requireNamespace("parallel", quietly = TRUE)) {
    parallel::mclapply(mylist,
                       myfnc,
                       mc.cores = min(cores, length(mylist)),
                       # When FALSE the allocation occurs on demand
                       mc.preschedule = length(mylist) / cores > 2,
                       ...)
  }
  else {
    lapply(mylist, myfnc, ...)
  }
}

#' Internal normalize data function
#'
#' @param data a set of numbers.
#' @param max.val The maximum value to normalize. If NULL use the max value
#'   present in the data. (default: \code{NULL})
#' @param min.val The minimum value to normalize. If NULL use the min value
#'   present in the data (default: \code{NULL})
#' @return The normalized data
#'
#' @examples
#' utiml_normalize(c(1,2,3,4,5))
#' #--> 0 0.25 0.5 0.75 1
#'
#' utiml_normalize(c(1,2,3,4,5), 10, 0)
#' #--> 0.1 0.2 0.3 0.4 0.5
utiml_normalize <- function(data, max.val = NULL, min.val = NULL) {
  max.val <- ifelse(is.null(max.val), max(data, na.rm = TRUE), max.val)
  min.val <- ifelse(is.null(min.val), min(data, na.rm = TRUE), min.val)
  (data - min.val)/(max.val - min.val)
}

#' Return the newdata to a data.frame or matrix
#'
#' @param newdata The data.frame or mldr data
#' @return A dataframe or matrix containing only dataset
#'
#' @examples
#' test <- emotions$dataset[,emotions$attributesIndexes]
#' all(test == utiml_newdata(emotions)) # TRUE
#' all(test == utiml_newdata(test)) # TRUE
utiml_newdata <- function(newdata) {
  UseMethod("utiml_newdata")
}

#' @describeIn utiml_newdata Return the data in the original format
utiml_newdata.default <- function(newdata) {
  newdata
}

#' @describeIn utiml_newdata Return the dataset from the mldr dataset
utiml_newdata.mldr <- function(newdata) {
  newdata$dataset[, newdata$attributesIndexes]
}


#' Rename the list using the names values or its own content
#'
#' @param X A list
#' @param names The list names, If empty the content of X is used
#' @return A list with the new names
#' @export
#'
#' @examples
#' utiml_renames(c("a", "b", "c"))
#' ## c(a="a", b="b", c="c")
#'
#' utiml_renames(c(1, 2, 3), c("a", "b", "c"))
#' ## c(a=1, b=2, c=3)
utiml_renames <- function (X, names = NULL) {
  names(X) <- utiml_ifelse(is.null(names), X, names)
  X
}

#' Define if two sets are equals independently of the order of the elements
#'
#' @param a A list
#' @param b Other list
#' @return Logical value where TRUE the sets are equals and FALSE otherwise.
#' @examples
#' utiml_is_equal_sets(c(1, 2, 3), c(3, 2, 1))
#' ## TRUE
#'
#' utiml_is_equal_sets(c(1, 2, 3), c(1, 2, 3, 4))
#' ## FALSE
utiml_is_equal_sets <- function (a, b) {
  length(setdiff(union(a, b), intersect(a, b))) == 0
}
