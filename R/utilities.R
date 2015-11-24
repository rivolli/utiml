#' @title Phi Correlation Coefficient
#' @family labels correlation
#' @description Calculate all labels phi correlation coefficient.
#' This is a specialized version of the Pearson product moment
#' correlation coefficient for categorical variables with two
#' values, also called dichotomous variables.
#' This is also called of Pearson product moment Correlation Coefficient (PCC)
#'
#' @param mdata Object of class \code{\link[mldr]{mldr}}, a multi-label dataset
#'
#' @return A matrix with all labels phi correlation coefficient. The rows and
#' columns have the labels and the values are the phi value. The main diagonal
#' have the 1 value that represents the correlation of a label with itself.
#'
#' @references
#'  Tsoumakas, G., Dimou, A., Spyromitros, E., Mezaris, V., Kompatsiaris, I., &
#'    Vlahavas, I. (2009). Correlation-based pruning of stacked binary relevance models
#'    for multi-label learning. In Proceedings of the Workshop on Learning from
#'    Multi-Label Data (MLD’09) (pp. 22–30).
#'
#' @export
#'
#' @examples
#' library(utiml)
#' result <- labels_correlation_coefficient(emotions)
#'
#' # Get the phi coefficient between the labels 'happy-pleased' and 'quiet-still'
#' result['happy-pleased', 'quiet-still']
#'
#' # Get all coefficients of a specific label
#' result[1, ]
labels_correlation_coefficient <- function(mdata) {
    labelnames <- rownames(mdata$labels)
    classes <- mdata$dataset[, mdata$labels$index]
    q <- length(labelnames)
    cor <- matrix(nrow = q, ncol = q, dimnames = list(labelnames, labelnames))
    for (i in 1:q) {
        for (j in i:q) {
            confmat <- table(classes[, c(i, j)])
            A <- as.numeric(confmat["1", "1"])
            B <- as.numeric(confmat["1", "0"])
            C <- as.numeric(confmat["0", "1"])
            D <- as.numeric(confmat["0", "0"])
            cor[i, j] <- abs((A * D - B * C)/sqrt(as.numeric(A + B) * (C + D) * (A + C) * (B + D)))
            cor[j, i] <- cor[i, j]
        }
    }
    cor
} 
