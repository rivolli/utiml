% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/method_dbr.R
\name{predict.DBRmodel}
\alias{predict.DBRmodel}
\title{Predict Method for DBR}
\usage{
\method{predict}{DBRmodel}(
  object,
  newdata,
  estimative = NULL,
  probability = getOption("utiml.use.probs", TRUE),
  ...,
  cores = getOption("utiml.cores", 1),
  seed = getOption("utiml.seed", NA)
)
}
\arguments{
\item{object}{Object of class '\code{DBRmodel}'.}

\item{newdata}{An object containing the new input data. This must be a
matrix, data.frame or a mldr object.}

\item{estimative}{A matrix containing the bipartition result of other
multi-label classification algorithm or an mlresult object with the
predictions.}

\item{probability}{Logical indicating whether class probabilities should be
returned. (Default: \code{getOption("utiml.use.probs", TRUE)})}

\item{...}{Others arguments passed to the base algorithm prediction for all
subproblems.}

\item{cores}{The number of cores to parallelize the training. Values higher
than 1 require the \pkg{parallel} package. (Default:
\code{options("utiml.cores", 1)})}

\item{seed}{An optional integer used to set the seed. This is useful when
the method is run in parallel. (Default: \code{options("utiml.seed", NA)})}
}
\value{
An object of type mlresult, based on the parameter probability.
}
\description{
This function predicts values based upon a model trained by \code{dbr}.
In general this method is a restricted version of
\code{\link{predict.BRPmodel}} using the 'NU' strategy.
}
\details{
As new feature is possible to use other multi-label classifier to predict the
estimate values of each label. To this use the prediction argument to inform
a result of other multi-label algorithm.
}
\examples{
\donttest{
# Predict SVM scores
model <- dbr(toyml)
pred <- predict(model, toyml)

# Passing a specif parameter for SVM predict algorithm
pred <- predict(model, toyml, na.action = na.fail)

# Using other classifier (EBR) to made the labels estimatives
estimative <- predict(ebr(toyml), toyml)
model <- dbr(toyml, estimate.models = FALSE)
pred <- predict(model, toyml, estimative = estimative)
}
}
\references{
Montanes, E., Senge, R., Barranquero, J., Ramon Quevedo, J., Jose Del Coz,
   J., & Hullermeier, E. (2014). Dependent binary relevance models for
   multi-label classification. Pattern Recognition, 47(3), 1494-1508.
}
\seealso{
\code{\link[=dbr]{Dependent Binary Relevance (DBR)}}
}
