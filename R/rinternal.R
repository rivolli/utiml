#
# This file contains internal functions of general purpose
# All functions here are not related about multi-label concepts
# The functions are sorted in alphabetical order
#

#' Conditional value selection
#'
#' @param test an object which can be coerced to logical mode.
#' @param yes object that will be returned when the test value is true.
#' @param no object that will be returned when the test value is false
#'
#' @return The respective value yes or no based on test value. This
#'  is an alternative way to use a single logical value for avoid
#'  the real if/else for choice lists, matrices and other composed data.
#'
#' @seealso \code{\link{ifelse}}
#' @export
#'
#' @examples
#' utiml_ifelse(TRUE, dataframe1, dataframe2) #dataframe1
#' utiml_ifelse(length(my.list) > 10, my.list[1:10], my.list)
utiml_ifelse <- function (test, yes, no) {
  list(yes, no)[c(test, !test)][[1]]
}

#' Select the suitable method lapply or mclaplly
#'
#' @param mylist a list to iterate.
#' @param myfnc The function to be applied to each element of the mylist.
#' @param cores The number of cores to use. If 1 use lapply oterwise use
#'    mclapply.
#' @param ... Extra arguments to myfnc.
#'
#' @return A list with the results of the specified method.
#' @export
#'
#' @examples
#' utiml_lapply(c(4,9,27), sqrt, 1) #use lapply
#' utiml_lapply(c(4,9,27), sqrt, 3) #use mclapply
utiml_lapply <- function (mylist, myfnc, cores, ...) {
  if (cores == 1)
    lapply(mylist, myfnc, ...)
  else
    parallel::mclapply(mylist, myfnc, mc.cores=min(cores, length(mylist)), ...)
}

#' @title Internal normalize data function
#'
#' @param data a set of numbers.
#' @param max.val The maximum value to normalize. If NULL use the max value
#'   present in the data. (default: \code{NULL})
#' @param min.val The minimum value to normalize. If NULL use the min value
#'   present in the data (default: \code{NULL})
#'
#' @return The normalized data
#' @export
#'
#' @examples
#' utiml_normalize(c(1,2,3,4,5))
#' #--> 0 0.25 0.5 0.75 1
#'
#' utiml_normalize(c(1,2,3,4,5), 10, 0)
#' #--> 0.1 0.2 0.3 0.4 0.5
utiml_normalize <- function (data, max.val=NULL, min.val=NULL) {
  max.val <- ifelse(is.null(max.val), max(data), max.val)
  min.val <- ifelse(is.null(min.val), min(data), min.val)
  (data-min.val) / (max.val-min.val)
}
