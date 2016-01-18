# SIMPLE MEASURES -------------------------------------------------------------

#' Return the number of attributes
#' @param data A dataset
#' @return integer
get_num_att <- function(data) {
  dim(data)[2] - 1
}

#' Return the number of samples (instances)
#' @param data A dataset
#' @return integer
get_num_samples <- function(data) {
  dim(data)[1]
}

#' Return the dimensionality of dataset (attributes / samples)
#' @param data A dataset
#' @return numeric
get_dim <- function (data) {
  get_num_att(data) / get_num_samples(data)
}

#' Return the number of classes values
#' @param data A dataset
#' @return integer
get_num_classes <- function(data) {
  nlevels(data[,ncol(data)])
}

#' Return the proportion of positive class
#' @param data A dataset
#' @return number between 0 and 1
get_prop_posclass <- function (data) {
  prop.classes <- prop.table(table(data[,ncol(data)]))
  as.numeric(prop.classes["1"])
}

#' Return the proportion of binary attributes
#'
#' Binary attributes are those containing two distinct values
#' @param data A dataset
#' @return number between 0 and 1
get_prop_binatt <- function (data) {
  length(which(lapply(apply(data[,-ncol(data)], 2, unique), length) == 2)) /
    get_num_att(data)
}

get_general_metafeatures <- function (data) {
  list(
    attributes     = get_num_att(data),
    samples        = get_num_samples(data),
    dimensionality = get_dim(data),
    binaryattr     = get_prop_binatt(data),
    positiveclass  = get_prop_posclass(data)
  )
}

# STATISTICAL -----------------------------------------------------------------
#' Standart deviation of attributes
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_sd <- function (data) {
  sds <- apply(data[, -ncol(data)], 2, sd)
  get_min_max_mean_sd(sds, 'sd')
}

#' Coefficient of variation
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_varcoef <- function (data) {
  varcoef <- apply(data[, -ncol(data)], 2, function (col) {
    sd(col) / mean(col)
  })
  get_min_max_mean_sd(varcoef, 'varcoef')
}

#' Covariance
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_covariance <- function (data) {
  matcov <- abs(cov(data[, -ncol(data)]))
  covariances <- c()
  col <- ncol(matcov)
  for (i in seq(col - 1)) {
    covariances <- c(covariances, matcov[i, seq(i + 1, col)])
  }
  get_min_max_mean_sd(covariances, 'covariances')
}

#' Linear correlation coefficient
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_lincorr <- function (data) {
  matcov <- abs(cov(data[, -ncol(data)]))
  sds <- apply(data[, -ncol(data)], 2, sd)
  corr <- c()
  col <- ncol(matcov)
  for (i in seq(col - 1)) {
    for (j in seq(i + 1, col)) {
      corr <- c(corr, matcov[i, j] / sqrt(sds[i] * sds[j]))
    }
  }
  get_min_max_mean_sd(corr, 'lincorr')
}

#' Skewness of attributes
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_skewness <- function (data) {
  skewness <- apply(data[, -ncol(data)], 2, e1071::skewness)
  get_min_max_mean_sd(abs(skewness), 'skewness')
}

#' Skewness mean of a dataset
#'
#' This is the mean of classes skewness, where classes skewness are the mean of
#' attributes skewness related with each class. This use only the values related
#' with each class.
#'
#' @param data A dataset
#' @return numeric value of skewness
#' #TODO@references
get_skewness <- function(data) {
  classes <- c(0, 1) # -- change it to multi-class supports --
  num_att <- get_num_att(data)
  skew <- 0.0
  for (class in classes) {
    s <- 0.0
    n <- 0.0
    att_data_class <- get_column_of_class(data, seq(num_att), class)
    for (col in seq(num_att)) {
      v <- e1071::skewness(att_data_class[, col])
      if (!is.nan(v) && !is.na(v)) {
        # -- NaN e.g. if the attribute has equal values for one class --
        s <- s + abs(v)
        n <- n + 1.0
      }
    }
    if(n > 0.0) { # -- 0 e.g. if one class with only one smaple
      skew <- skew + (s / n)
    }
  }

  skew / (length(classes))
}

#' Kurtosis of attributes
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_stat_kurtosis <- function (data) {
  kurtosis <- apply(data[, -ncol(data)], 2, e1071::kurtosis)
  get_min_max_mean_sd(abs(kurtosis), 'kurtosis')
}

#' Kurtosis mean of dataset
#'
#' This is the mean of classes kurtosis, where classes kurtosis are the mean of
#' attributes kurtosis related with each class. This use only the values related
#' with each class.
#'
#' @param data A dataset
#' @return numeric value of kurtosis
#' #TODO@references
get_kurtosis <- function(data) {
  classes <- c(0, 1) # -- change it to multi-class supports --
  num_att <- get_num_att(data)
  kurtosis <- 0.0
  for (class in classes) {
    s <- 0.0
    n <- 0.0
    att_data_class <- get_column_of_class(data, seq(num_att), class)
    for (col in seq(num_att)) {
      v <- e1071::kurtosis(att_data_class[, col])
      if (!is.nan(v) && !is.na(v)) {
        # -- NaN e.g. if the attribute has equal values for one class --
        s <- s + abs(v)
        n <- n + 1.0
      }
    }

    if (n > 0.0) { # -- 0 e.g. if one class with only one smaple
      kurtosis <- kurtosis + (s / n)
    }
  }

  (kurtosis / (length(classes))) + 3.0
}

#- Cost matrix indicator
#- Number of outliers

get_statistical_metafeatures <- function (data) {
  c(
    get_stat_sd(data),
    get_stat_varcoef(data),
    get_stat_covariance(data),
    get_stat_lincorr(data),
    skewness = get_skewness(data),
    get_stat_skewness(data),
    kurtosis = get_kurtosis(data),
    get_stat_kurtosis(data)
  )
}

# INFORMATION-THEORETIC -------------------------------------------------------
#' Normalized class entropy
#'
#' @param data A dataset
#' @return numeric value of the normalized entropy
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_normalized_classentropy <- function (data) {
  get_normalized_entropy(data[, ncol(data)])
}

#' Normalized attribute entropy
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_normalized_attentropy <- function (data) {
  entropies <- apply(data[, -ncol(data)], 2, get_normalized_entropy)
  get_min_max_mean_sd(entropies, 'att_norm_ent')
}

#' Attribute entropy
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' #TODO@references
get_att_entropy <- function (data) {
  entropies <- apply(infotheo::discretize(data[, -ncol(data)]), 2,
                     infotheo::entropy)
  get_min_max_mean_sd(entropies, 'att_ent')
}

#' Joint entropy of class and attribute
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_joint_entropy <- function (data) {
  labels <- data[, ncol(data)]
  entropies <- apply(data[, -ncol(data)], 2, function (column) {
    column <- infotheo::discretize(column, disc="equalwidth")[, 1]
    infotheo::entropy(paste(column, labels, sep='_'))
  })
  get_min_max_mean_sd(entropies, 'joint_ent')
}

#' Mutual information of class and attribute
#'
#' @param data A dataset
#' @return numeric vector with min, max, mean and sd values
#' @references
#'  Castiello, C., Castellano, G., & Fanelli, A. M. (2005). Meta-data:
#'  Characterization of Input Features for Meta-learning. Modeling Decisions
#'  for Artificial Intelligence, 3558, 457–468.
get_mutual_information <- function (data) {
  labels <- data[ncol(data)]
  mutinf <- apply(data[, -ncol(data)], 2, function (column) {
    column <- infotheo::discretize(column, disc="equalwidth")
    infotheo::mutinformation(column, labels)
  })
  get_min_max_mean_sd(mutinf, 'mut_inf')
}


get_informationtheoretic_metafeatures <- function (data) {
  #TODO optimize this function calling the infotheo::discretize a single time
  class.ent <- infotheo::entropy(data[ncol(data)])
  att.ent <- get_att_entropy(data)
  mutinf <- get_mutual_information(data)

  mutifmean <- as.numeric(mutinf["mut_inf_mean"])
  attentmean <- as.numeric(att.ent["att_ent_mean"])

  c(
    class_entropy = get_normalized_classentropy(data),
    att.ent,
    get_normalized_attentropy(data),
    get_joint_entropy(data),
    mutinf,
    equivalent_num_att = class.ent / mutifmean,
    noise_signal_ratio = (attentmean - mutifmean) / mutifmean
  )
}

# UTILITIES -------------------------------------------------------------------

#' Calculate the min, max, mean and standart deviation measures for a list
#'
#' @param x A list
#' @param name The base name
#' @return
#' get_min_max_mean_sd(c(1,2,3,4,5), 'test')
#' ## test_min  test_max test_mean   test_sd
#' ## 1.000000  5.000000  3.000000  1.581139
get_min_max_mean_sd <- function(x, name) {
  m <- numeric(0)
  m[paste(name, "min", sep="_")] = min(x, na.rm = TRUE)
  m[paste(name, "max", sep="_")] = max(x, na.rm = TRUE)
  m[paste(name, "mean", sep="_")] = mean(x, na.rm = TRUE)

  s <- sd(x, na.rm = TRUE)
  if(is.na(s)) {
    s <- 0
  }
  m[paste(name, "sd", sep="_")] <- s
  m
}

#' Return the column values related with the class
#'
#' @param data A dataset
#' @param column The column name or number
#' @param class The class value
#'
#' @return A list with the column values that are related with the class value
get_column_of_class <- function(data, column, class) {
  data[data[ncol(data)] == class, column]
}

#' Return the normalized entropy of a column
#'
#' @param column a data column
#'
#' @return A numeric value with the normalized entropy
get_normalized_entropy <- function(column) {
  column <- infotheo::discretize(column, disc = "equalwidth")[, 1]
  n <- nlevels(as.factor(column))

  infotheo::entropy(column) / log(n)
}
