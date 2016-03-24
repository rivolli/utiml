#' Phi Correlation Coefficient
#'
#' Calculate all labels phi correlation coefficient. This is a specialized
#' version of the Pearson product moment correlation coefficient for categorical
#' variables with two values, also called dichotomous variables. This is also
#' called of Pearson product moment Correlation Coefficient (PCC)
#'
#' @param mdata A mldr multi-label dataset
#' @return A matrix with all labels correlation coefficient. The rows and
#'  columns have the labels and each value are the correlation between the
#'  labels. The main diagonal have the 1 value that represents the correlation
#'  of a label with itself.
#' @references
#' Tsoumakas, G., Dimou, A., Spyromitros, E., Mezaris, V., Kompatsiaris, I., &
#'  Vlahavas, I. (2009). Correlation-based pruning of stacked binary relevance
#'  models for multi-label learning. In Proceedings of the Workshop on Learning
#'  from Multi-Label Data (MLD'09) (pp. 22-30).
#' @seealso \code{\link[=mbr]{Meta-BR (MBR or 2BR)}}
#' @export
#'
#' @examples
#' result <- calculate_labels_correlation(toyml)
#'
#' # Get the phi coefficient between the labels 'y1' and 'y2'
#' result['y1', 'y2']
#'
#' # Get all coefficients of a specific label
#' result[4, -4]
calculate_labels_correlation <- function(mdata) {
  label.names <- rownames(mdata$labels)
  classes <- mdata$dataset[, mdata$labels$index]
  q <- length(label.names)
  cor <- matrix(nrow = q, ncol = q, dimnames = list(label.names, label.names))
  for (i in 1:q) {
    for (j in i:q) {
      confmat <- table(classes[, c(i, j)])
      A <- as.numeric(confmat["1", "1"])
      B <- as.numeric(confmat["1", "0"])
      C <- as.numeric(confmat["0", "1"])
      D <- as.numeric(confmat["0", "0"])
      cor[i, j] <- abs((A * D - B * C) / sqrt(as.numeric(A + B) * (C + D) *
                                                (A + C) * (B + D)))
      cor[j, i] <- cor[i, j]
    }
  }
  cor
}

#' Calculate the Information Gain for each pair of labels
#'
#' @param mdata A mldr dataset containing the label information.
#' @return A matrix where the rows and columns represents the labels.
#' @references
#'  Alali, A., & Kubat, M. (2015). PruDent: A Pruned and Confident Stacking
#'   Approach for Multi-Label Classification. IEEE Transactions on Knowledge
#'   and Data Engineering, 27(9), 2480-2493.
#' @export
#'
#' @examples
#' calculate_labels_information_gain(toyml)
calculate_labels_information_gain <- function (mdata) {
  entropy <- function (prob) {
    res <- c(0, -prob * log2(prob) - (1 - prob) * log2(1 - prob))
    zero <- prob == 0 || prob == 1
    res[c(zero, !zero)]
  }

  labelnames <- rownames(mdata$labels)
  classes <- mdata$dataset[,mdata$labels$index]
  q <- length(labelnames)
  ig <- matrix(nrow = q, ncol = q, dimnames = list(labelnames, labelnames))
  for (i in 1:q) {
    for (j in i:q) {
      Hya <- entropy(mdata$labels$freq[i])
      hasJ <- classes[j] == 1
      Hyab <- mdata$labels$freq[j] *
        entropy(sum(classes[hasJ, i] == 1) / sum(hasJ)) +
        (1 - mdata$labels$freq[j]) *
        entropy(sum(classes[classes[j] == 0, i] == 1) / sum(!hasJ))

      ig[i,j] <- Hya  - Hyab
      ig[j,i] <- ig[i,j]
    }
    ig[i,i] <- 0
  }
  ig
}
