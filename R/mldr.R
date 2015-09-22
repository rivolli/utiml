#' @title Create distinct partitions of a multi-label dataset
#' @family mldr
#' @family sampling
#' @description This method creates multi-label dataset for
#'  train, test, validation or other proposes. The number of
#'  partitions is defined in \code{partitions} parameter.
#'  The instances are used in only one partition of divistion.
#'  Use the SEED parameter to obtain the same result again.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param partitions A list of percentages or a single value.
#'  The sum of all values does not be greater than 1. If a
#'  single value is informed then the complement of them is
#'  applied to generated the second partition. If two or more
#'  values are informed and the sum of them is lower than 1
#'  the partitions will be generated with the informed proportion.
#'  (default: \code{c(0.7, 0.3)})
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again. (default: \code{NULL})
#'
#' @return A list with at least two datasets sampled as specified
#'  in partitions parameter.
#' @export
#'
#' @examples
#' # Create two partitions with 70% and 30% for training and test
#' datasets <- mldr_random_holdout(emotions)
#'
#' # The same result can be obtained as:
#' datasets <- mldr_random_holdout(emotions, 0.7)
#' print(datasets[[1]]$measures)
#' print(datasets[[2]]$measures)
#'
#' # Using a SEED and split the dataset in the half
#' datasets <- mldr_random_holdout(emotions, 0.5, SEED = 12)
#'
#' # Split the dataset in three parts
#' datasets <- mldr_random_holdout(emotions, c(0.70, 0.15, 0.15))
mldr_random_holdout <- function (mdata, partitions = c(0.7, 0.3), SEED = NULL) {
  # Validations
  if (sum(partitions) > 1)
    stop("The sum of partitions can not be greater than 1")

  if (length(partitions) == 1)
    partitions[2] <- 1 - partitions[1]

  # Generate random sequence
  if (!is.null(SEED)) {
    set.seed(SEED)
    rows <- sample(1:mdata$measures$num.instances)
    set.seed(NULL)
  }
  else
    rows <- sample(1:mdata$measures$num.instances)

  # Calculate the indexes and limit the last index to avoid rounds mistakes
  idx <- c(0, cumsum(round(mdata$measures$num.instances * partitions)))
  idx[length(idx)] <- round(sum(mdata$measures$num.instances * partitions))

  ldata <- list()
  for (i in 1:length(partitions))
    ldata[[i]] <- mldr_subset(mdata, rows[(idx[i]+1):idx[i+1]], mdata$attributesIndexes)

  ldata
}

#' @title Create stratified partitions of a multi-label dataset
#' @family mldr
#' @family sampling
#' @description This method creates multi-label dataset for
#'  train, test, validation or other proposes using stratified
#'  approach based on labelsets distribution. The number of
#'  partitions is defined in \code{partitions} parameter.
#'  The instances are used in only one partition of divistion.
#'  Use the SEED parameter to obtain the same result again.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param partitions A list of percentages or a single value.
#'  The sum of all values does not be greater than 1. If a
#'  single value is informed then the complement of them is
#'  applied to generated the second partition. If two or more
#'  values are informed and the sum of them is lower than 1
#'  the partitions will be generated with the informed proportion.
#'  (default: \code{c(0.7, 0.3)})
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again. (default: \code{NULL})
#'
#' @return A list with at least two datasets sampled as specified
#'  in partitions parameter.
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011).
#'  On the stratification of multi-label data. In Proceedings of the
#'  Machine Learning and Knowledge Discovery in Databases - European
#'  Conference, ECML PKDD (pp. 145–158).
#'
#' @export
#'
#' @examples
#' # Create two partitions with 70% and 30% for training and test
#' datasets <- mldr_stratified_holdout(emotions)
#'
#' # The same result can be obtained as:
#' datasets <- mldr_stratified_holdout(emotions, 0.7)
#' print(datasets[[1]]$measures)
#' print(datasets[[2]]$measures)
#'
#' # Using a SEED and split the dataset in the half
#' datasets <- mldr_stratified_holdout(emotions, 0.5, SEED = 12)
#'
#' # Split the dataset in three parts
#' datasets <- mldr_stratified_holdout(emotions, c(0.70, 0.15, 0.15))
mldr_stratified_holdout <- function (mdata, partitions = c(0.7, 0.3), SEED = NULL) {
  # Validations
  if (sum(partitions) > 1)
    stop("The sum of partitions can not be greater than 1")

  if (length(partitions) == 1)
    partitions[2] <- 1 - partitions[1]

  if (!is.null(SEED))
    set.seed(SEED)

  # Splits
  ldata <- lapply(utiml_labelset_stratification(mdata, partitions), function (fold) {
    mldr_subset(mdata, fold, mdata$attributesIndexes)
  })

  if (!is.null(SEED))
    set.seed(NULL)

  ldata
}

#' @title Create iterative stratified partitions of a multi-label dataset
#' @family mldr
#' @family sampling
#' @description This method creates multi-label dataset for
#'  train, test, validation or other proposes using interative
#'  stratified algorithm, that is based on labels proportions.
#'  The number of partitions is defined in \code{partitions} parameter.
#'  The instances are used in only one partition of divistion.
#'  Use the SEED parameter to obtain the same result again.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param partitions A list of percentages or a single value.
#'  The sum of all values does not be greater than 1. If a
#'  single value is informed then the complement of them is
#'  applied to generated the second partition. If two or more
#'  values are informed and the sum of them is lower than 1
#'  the partitions will be generated with the informed proportion.
#'  (default: \code{c(0.7, 0.3)})
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again. (default: \code{NULL})
#'
#' @return A list with at least two datasets sampled as specified
#'  in partitions parameter.
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011).
#'  On the stratification of multi-label data. In Proceedings of the
#'  Machine Learning and Knowledge Discovery in Databases - European
#'  Conference, ECML PKDD (pp. 145–158).
#'
#' @export
#'
#' @examples
#' # Create two partitions with 70% and 30% for training and test
#' datasets <- mldr_iterative_stratification_holdout(emotions)
#'
#' # The same result can be obtained as:
#' datasets <- mldr_iterative_stratification_holdout(emotions, 0.7)
#' print(datasets[[1]]$measures)
#' print(datasets[[2]]$measures)
#'
#' # Using a SEED and split the dataset in the half
#' datasets <- mldr_iterative_stratification_holdout(emotions, 0.5, SEED = 12)
#'
#' # Split the dataset in three parts
#' datasets <- mldr_iterative_stratification_holdout(emotions, c(0.70, 0.15, 0.15))
mldr_iterative_stratification_holdout <- function (mdata, partitions = c(0.7, 0.3), SEED = NULL) {
  # Validations
  if (sum(partitions) > 1)
    stop("The sum of partitions can not be greater than 1")

  if (length(partitions) == 1)
    partitions[2] <- 1 - partitions[1]

  if (!is.null(SEED))
    set.seed(SEED)

  # Splits
  ldata <- lapply(utiml_iterative_stratification(mdata, partitions), function (fold) {
    mldr_subset(mdata, fold, mdata$attributesIndexes)
  })

  if (!is.null(SEED))
    set.seed(NULL)

  ldata
}

#' @title Get the multi-labels datasets for k-fold Cross Validation
#' @family mldr
#' @family sampling
#' @description This is a simple way to use k-fold cross validation.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param kfold An object of class \code{mldr_kfolds}, this is obtained
#'  from use of some kfold method.
#' @param n The number of fold to separated train and test subsets.
#' @param has.validation Logical value that indicate if a validation
#'  dataset will be used. (defaul: \code{FALSE})
#'
#' @return A list contained train and test dataset:
#'  \describe{
#'    \code{train}{The mldr dataset with train examples, that inclue all
#'      examples except those that are in test and validation samples}
#'    \code{test}{The mldr dataset with test examples, defined by the
#'      number of the fold}
#'    \code{validation}{Optionally, only if \code{has.validation = TRUE}.
#'      The mldr dataset with validation examples}
#'  }
#'
#' @export
#'
#' @examples
#' library(utiml)
#' folds <- mldr_random_kfold(emotions, 10)
#'
#' # Using the first iteration
#' dataset <- mldr_getfold(emotions, folds, 1)
#' classifier <- br(dataset$train)
#' result <- predict(classifier, dataset$test)
#'
#' # All iterations
#' for (i in 1:10) {
#'    dataset <- mldr_getfold(emotions, folds, i)
#'    #dataset$train
#'    #dataset$test
#' }
#'
#' # Using validation
#' dataset <- mldr_getfold(emotions, folds, 10, TRUE)
#' # dataset$train, dataset$test, #dataset$validation
mldr_getfold <- function (mdata, kfold, n, has.validation = FALSE) {
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (class(kfold) != "mld#' # 2 folds
#' mldr_random_kfold(emotions, 2)#' # 2 folds
#' mldr_random_kfold(emotions, 2)r_kfolds")
    stop("Second argument must be an 'mldr_kfolds' object")

  if (n < 1 || n > kfold$k)
    stop(cat("The 'n' value must be between 1 and", kfold$k))

  folds <- kfold$fold[-n]
  if (has.validation) {
    i <- n == length(folds)
    v <- c(1, n)[c(i, !i)]
    folds <- folds[-v]
  }
  ldata <- list()
  ldata$train <- mldr_subset(mdata, unlist(folds), mdata$attributesIndexes)
  ldata$test <- mldr_subset(mdata, kfold$fold[[n]], mdata$attributesIndexes)

  if (has.validation)
    ldata$validation <- mldr_subset(mdata, kfold$fold[[v]], mdata$attributesIndexes)

  ldata
}

#' @title Generate random k folds for multi-label data
#' @family mldr
#' @family sampling
#' @description Use this method to generate random sampling for multi-labels
#'   datasets. This may generate folds with differents proportions of labels.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param k The number of folds. (default: 10)
#' @param SEED A single value, interpreted as an integer to allow obtain the
#'   same results again. (default: \code{NULL})
#'
#' @return An object of type \code{mldr_kfolds}. This is a list with
#'  k elements, where each element contains a list of row indexes based
#'  on the original dataset.
#'
#' @seealso \code{\link{mldr_getfold}}
#' @export
#'
#' @examples
#' # 2 folds
#' folds <- mldr_random_kfold(emotions, 2)
#'
#' # 10 folds
#' folds <- mldr_random_kfold(emotions, 10)
mldr_random_kfold <- function (mdata, k = 10, SEED = NULL) {
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  k <- as.integer(k)
  if (!is.null(SEED)) {
    set.seed(SEED)
    rows <- sample(1:mdata$measures$num.instances)
    set.seed(NULL)
  }
  else
    rows <- sample(1:mdata$measures$num.instances)

  kf <- list(k=k)
  kf$fold <- split(rows, ceiling(seq_along(rows)/k))
  class(kf) <- "mldr_kfolds"

  kf
}

#' @title Generate stratified k folds for multi-label data
#' @family mldr
#' @family sampling
#' @description Use this method to generate stratified sampling for multi-labels
#'   datasets. This method use the labelsets to compute the partitions
#'   distributions, however some specific labelsets can not occurs in all folds.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param k The number of folds. (default: 10)
#' @param SEED A single value, interpreted as an integer to allow obtain the
#'   same results again. (default: \code{NULL})
#'
#' @return An object of type \code{mldr_kfolds}. This is a list with
#'  k elements, where each element contains a list of row indexes based
#'  on the original dataset.
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the
#'  stratification of multi-label data. In Proceedings of the Machine
#'  Learningand Knowledge Discovery in Databases - European Conference,
#'  ECML PKDD (pp. 145–158).
#'
#' @seealso \code{\link{mldr_getfold}}
#' @export
#'
#' @examples
#' # 2 folds
#' folds <- mldr_stratified_kfold(emotions, 2)
#'
#' # 10 folds
#' folds <- mldr_stratified_kfold(emotions, 10)
mldr_stratified_kfold <- function (mdata, k = 10, SEED = NULL) {
  if (!is.null(SEED))
    set.seed(SEED)

  kf <- list(k=k)
  kf$fold <- utiml_labelset_stratification(mdata, rep(1/k, k))
  class(kf) <- "mldr_kfolds"

  if (!is.null(SEED))
    set.seed(NULL)

  kf
}

#' @title Generate labels based stratified k folds for multi-label data
#' @family mldr
#' @family sampling
#' @description Use this method to generate stratified sampling for multi-labels
#'   datasets using the iterative stratified algorithm. This method use the
#'   labels distributions to compute the partitions proportions, however some
#'   specific label can not occurs in all folds.
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param k The number of folds. (default: 10)
#' @param SEED A single value, interpreted as an integer to allow obtain the
#'   same results again. (default: \code{NULL})
#'
#' @return An object of type \code{mldr_kfolds}. This is a list with
#'  k elements, where each element contains a list of row indexes based
#'  on the original dataset.
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the
#'  stratification of multi-label data. In Proceedings of the Machine
#'  Learningand Knowledge Discovery in Databases - European Conference,
#'  ECML PKDD (pp. 145–158).
#'
#' @seealso \code{\link{mldr_getfold}}
#' @export
#'
#' @examples
#' # 2 folds
#' folds <- mldr_iterative_stratification_kfold(emotions, 2)
#'
#' # 10 folds
#' folds <- mldr_iterative_stratification_kfold(emotions, 10)
mldr_iterative_stratification_kfold <- function (mdata, k = 10, SEED = NULL) {
  if (!is.null(SEED))
    set.seed(SEED)

  kf <- list(k=k)
  kf$fold <- utiml_iterative_stratification(mdata, rep(1/k, k))
  class(kf) <- "mldr_kfolds"

  if (!is.null(SEED))
    set.seed(NULL)

  kf
}

#' @title Create a subset of a dataset
#' @family mldr
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

#' @title Create a random subset of a dataset
#' @family mldr
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
