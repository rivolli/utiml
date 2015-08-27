#' utiml: Utilities for Multi-Label Learning
#'
#' @description
#' The utiml package provides the follow categories of important functions:
#' \enumerate{
#'  \item Transformation methods
#' }
#'
#' @section 1. Transformation methods:
#' These functions contain the main methods for predict multi-label data
#'    using transformation approaches
#'  \itemize{
#'    \item \code{\link[=br]{Binary Relevance}}
#'    \item Classifier Chains
#'  }
#'
#' @section 1.1 Base methods:
#' The default implementation has support to the specific base methods:
#' \itemize{
#'  \item \emph{Support Vector Machines (SVM)} - requires the '\pkg{e1071}'
#'    package
#'  \item \emph{Decision tree C4.5} (J48 implementation) - requires the
#'    '\pkg{RWeka}' package
#'  \item \emph{Decision tree C5.0} - requires the '\pkg{C50}' package
#'  \item \emph{Random Forest}, a ensemble of decision trees - requires the
#'    '\pkg{randomForest}' package
#'  \item \emph{Naive Bayes} - requires the '\pkg{e1071}' package
#'  \item \emph{k-Nearest Neighbors (kNN)} - requires the '\pkg{class}' package
#' }
#' New base methods can be added, to use  other methods see \link{TODO}
#'  documentation.
#'
#' @author
#' \itemize{
#'  \item Adriano Rivolli <rivolli@@usp.br>
#'  \item Andre C. P. L. F. de Carvalho <andre@@icmc.usp.br>
#' }
#'
#' @docType package
#' @name utiml
NULL
