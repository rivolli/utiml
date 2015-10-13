#' @title Create a holdout partition based in the specified method
#' @family sampling
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param partitions A list of percentages with partitions sizes.
#' @param partition.names a vector with the partition names.
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again.
#' @param holdout.method The method to split the data.
#'
#' @return A list with at least two datasets sampled as specified
#'  in partitions parameter.
#' @export
#'
#' @examples
#' utiml_holdout(mdata, partitions, partition.names, SEED, utiml_random_split)
utiml_holdout <- function (mdata, partitions, partition.names, SEED, holdout.method) {
  # Validations
  if (sum(partitions) > 1)
    stop("The sum of partitions can not be greater than 1")

  if (!is.null(SEED))
    set.seed(SEED)

  partitions <- utiml_ifelse(length(partitions) == 1, c(partitions, 1 - partitions), partitions)

  # Split data
  ldata <- do.call(holdout.method, list(mdata = mdata, partitions = partitions))

  if (!is.null(SEED))
    set.seed(NULL)

  names(ldata) <- partition.names
  ldata
}

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
#' @param partition.names a vector with the partition names (optional).
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again. (default: \code{NULL}, optional)
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
mldr_random_holdout <- function (mdata, partitions = c(0.7, 0.3), partition.names = NULL, SEED = NULL) {
  utiml_holdout(mdata, partitions, partition.names, SEED, function (mdata, partitions){
    lapply(utiml_random_split(mdata, partitions), function (fold) {
      mldr_subset(mdata, fold, mdata$attributesIndexes)
    })
  })
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
#' @param partition.names a vector with the partition names (optional).
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
#' datasets <- mldr_stratified_holdout(emotions, 0.7)partition.names = NULL,
#' print(datasets[[1]]$measures)
#' print(datasets[[2]]$measures)
#'
#' # Using a SEED and split the dataset in the half
#' datasets <- mldr_stratified_holdout(emotions, 0.5, SEED = 12)
#'
#' # Split the dataset in three parts
#' datasets <- mldr_stratified_holdout(emotions, c(0.70, 0.15, 0.15))
mldr_stratified_holdout <- function (mdata, partitions = c(0.7, 0.3), partition.names = NULL, SEED = NULL) {
  utiml_holdout(mdata, partitions, partition.names, SEED, function (mdata, partitions){
    lapply(utiml_labelset_stratification(mdata, partitions), function (fold) {
      mldr_subset(mdata, fold, mdata$attributesIndexes)
    })
  })
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
#' @param partition.names a vector with the partition names (optional).
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
mldr_iterative_stratification_holdout <- function (mdata, partitions = c(0.7, 0.3), partition.names = NULL, SEED = NULL) {
  utiml_holdout(mdata, partitions, partition.names, SEED, function (mdata, partitions){
    lapply(utiml_iterative_stratification(mdata, partitions), function (fold) {
      mldr_subset(mdata, fold, mdata$attributesIndexes)
    })
  })
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
#' # Using the first iterationrows <- sample(1:mdata$measures$num.instances)rows <- sample(1:mdata$measures$num.instances)
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

  if (class(kfold) != "mldr_kfolds")
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

#' @title Create the k partitions of k-fold
#' @family sampling
#'
#' @param mdata A dataset of class \code{\link[mldr]{mldr}}.
#' @param k The number of folds.
#' @param SEED A single value, interpreted as an integer to allow
#'  obtain the same results again.
#' @param kfold.method The method to split the data.
#'
#' @return An object of type mldr_kfolds
#' @export
#'
#' @examples
#' utiml_kfold(mdata, 10, SEED, utiml_random_split)
utiml_kfold <- function (mdata, k, SEED, kfold.method) {
  if(class(mdata) != 'mldr')
    stop('First argument must be an mldr object')

  if (!is.null(SEED))
    set.seed(SEED)

  kf <- list(k=k)
  kf$fold <- do.call(kfold.method, list(mdata = mdata, r = rep(1/k, k)))
  class(kf) <- "mldr_kfolds"

  if (!is.null(SEED))
    set.seed(NULL)

  kf
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
  utiml_kfold(mdata, k, SEED, utiml_random_split)
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
  utiml_kfold(mdata, k, SEED, utiml_labelset_stratification)
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
  utiml_kfold(mdata, k, SEED, utiml_iterative_stratification)
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

#' @title Random split of a dataset
#'
#' @param mdata A mldr dataset.
#' @param r Desired proportion of examples in each subset r1, . . . rk.
#'
#' @return A list with k disjoint indexes subsets S1, . . .Sk.
#' @export
#'
#' @examples
#' utiml_random_split(emotions, c(0.6, 0.2, 0.2))
utiml_random_split <- function (mdata, r) {
  index <- c()
  amount <- round(mdata$measures$num.instances * r)

  dif <- mdata$measures$num.instances - sum(amount)
  for (i in 1:abs(dif))
    amount[i] <- amount[i] + sign(dif)

  for (i in 1:length(amount))
    index <- c(index, rep(i, amount[i]))

  split(sample(1:mdata$measures$num.instances), index)
}

#' @title Internal Iterative Stratification
#' @description Create the indexes using the Iterative Stratification
#'   algorithm.
#'
#' @param mdata A mldr dataset.
#' @param r Desired proportion of examples in each subset r1, . . . rk.
#'
#' @return A list with k disjoint indexes subsets S1, . . .Sk.
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the
#'  stratification of multi-label data. In Proceedings of the Machine
#'  Learningand Knowledge Discovery in Databases - European Conference,
#'  ECML PKDD (pp. 145–158).
#'
#' @export
#'
#' @examples
#' # Create 3 partitions for train, validation and test
#' indexes <- utiml_iterative_stratification(emotions, c(0.6,0.1,0.3))
#'
#' # Create a stratified 10-fold
#' indexes <- utiml_iterative_stratification(emotions, rep(0.1,10))
utiml_iterative_stratification <- function (mdata, r) {
  D <- 1:mdata$measures$num.instances
  S <- lapply(1:length(r), function (i) integer())

  # Calculate the desired number of examples at each subset
  cj <- round(mdata$measures$num.instances * r)
  dif <- mdata$measures$num.instances - sum(cj)
  if (dif != 0)
    cj[1:abs(dif)] <- cj[1:abs(dif)] + c(1, -1)[c(dif>0, dif<0)]

  # Calculate the desired number of examples of each label at each subset
  cji <- trunc(sapply(mdata$labels$count, function (di) di * r))
  colnames(cji) <- rownames(mdata$labels)

  while (length(D) > 0) {
    # Find the label with the fewest (but at least one) remaining examples,
    Dl <- apply(mdata$dataset[D, mdata$labels$index], 2, function (col) as.numeric(names(which(col == 1))))
    Di <- unlist(lapply(Dl, length))
    l <- names(which.min(Di[Di>0]))

    for (ex in Dl[[l]]) {
      # Find the subset(s) with the largest number of desired examples for this
      # label, breaking ties by considering the largest number of desired examples
      m <- which(cji[which.max(cji[,l]),l] == cji[,l])
      if (length(m) > 1) {
        m <- intersect(m, which(cj[m[which.max(cj[m])]] == cj))
        if (length(m) > 1)
          m <- sample(m)[1]
      }

      S[[m]] <- c(S[[m]], ex)
      D <- D[D != ex]

      # Update desired number of examples
      i <- which(mdata$dataset[ex, mdata$labels$index] == 1)
      cji[m, i] <- cji[m, i] - 1
      cj[m] <- cj[m] - 1
    }
  }

  S
}

#' @title Labelsets Stratification
#' @description Create the indexes using the Labelsets Stratification
#'   approach.
#'
#' @param mdata A mldr dataset
#' @param r Desired proportion of examples in each subset, r1, . . . rk
#'
#' @return A list with k disjoint indexes subsets S1, . . .Sk
#'
#' @references Sechidis, K., Tsoumakas, G., & Vlahavas, I. (2011). On the
#'  stratification of multi-label data. In Proceedings of the Machine
#'  Learningand Knowledge Discovery in Databases - European Conference,
#'  ECML PKDD (pp. 145–158).
#'
#' @export
#'
#' @examples
#' # Create 3 partitions for train, validation and test
#' indexes <- utiml_labelset_stratification(emotions, c(0.6,0.1,0.3))
#'
#' # Create a stratified 10-fold
#' indexes <- utiml_labelset_stratification(emotions, rep(0.1,10))
utiml_labelset_stratification <- function (mdata, r) {
  D <- sample(mdata$measures$num.instances)
  S <- lapply(1:length(r), function (i) integer())
  labelsets <- apply(mdata$dataset[,mdata$labels$index], 1, paste, collapse = "")

  # Calculate the desired number of examples of each labelset at each subset
  cji.aux <- sapply(mdata$labelsets, function (di) di * r)
  cji <- trunc(cji.aux)
  dif <- cji.aux - cji
  rest <- round(apply(dif, 1, sum))
  for (ls in rev(names(mdata$labelsets))) {
    s <- sum(dif[,ls])
    if (s > 0) {
      for (i in 1:s) {
        fold <- which.max(rest)
        rest[fold] <- rest[fold] - 1
        cji[fold, ls] <- cji[fold, ls] + 1
      }
    }
  }

  for (ex in D) {
    ls <- labelsets[ex]
    fold <- which.max(cji[,ls])
    if (cji[fold, ls] > 0) {
      S[[fold]] <- c(S[[fold]], ex)
      cji[fold, ls] <- cji[fold, ls] - 1
    }
  }

  S
}

