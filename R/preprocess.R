#' @title Fill sparce dataset with 0 or '' values
#' @description Transform a sparce dataset filling values, if there is a numeric
#' column but with text value change this column to numerical.
#'
#' @param mdata The mldr dataset to be filled.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_fill_sparce_data(emotions)
mldr_fill_sparce_data <- function(mdata) {
    is.letter <- function(x) grepl("[[:alpha:]]", x)
    dataset <- data.frame(row.names = rownames(mdata$dataset))
    dataset <- cbind(dataset, lapply(mdata$dataset[, 1:mdata$measures$num.attributes], function(col) {
        if (sum(!complete.cases(col)) > 0) {
            # Has NA value
            if (is.numeric(col)) {
                # Numeric value - fill with 0
                col[is.na(col)] <- 0
            } else if (any(is.letter(col))) {
                # Text value - fill with ''
                col <- as.character(col)
                col[is.na(col)] <- ""
            } else {
                # Text but with numeric values - convert to numeric and fill with 0
                col <- as.numeric(as.character(col))
                col[is.na(col)] <- 0
            }
        }
        col
    }))

    mldr::mldr_from_dataframe(dataset, mdata$labels$index, mdata$name)
}

#' @title Normalize dataset attributes
#' @description Normalize all numerical attributes to values between
#'  0 and 1.
#'
#' @param mdata The mldr dataset to be normalized.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_normalize(emotions)
mldr_normalize <- function(mdata) {
    data <- mdata$dataset[seq(mdata$measures$num.attributes)]
    for (col in mdata$attributesIndexes) {
        if (is.numeric(data[, col])) {
            data[col] <- utiml_normalize(data[col])
        }
    }
    mldr::mldr_from_dataframe(data, mdata$labels$index, mdata$name)
}

#' @title Remove unique attributes
#' @description Remove the attributes that have a single value.
#'  Observation: NA is considered a value.
#'
#' @param mdata The mldr dataset to remove.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_remove_unique_attributes(emotions)
mldr_remove_unique_attributes <- function(mdata) {
    labelsIndexes <- c()
    attributesIndexes <- c()
    for (col in seq(mdata$measures$num.attributes)) {
        if (col %in% mdata$labels$index) {
            attributesIndexes <- c(attributesIndexes, col)
            labelsIndexes <- c(labelsIndexes, length(attributesIndexes))
        } else {
            if (length(unique(mdata$dataset[, col])) > 1) {
                attributesIndexes <- c(attributesIndexes, col)
            }
        }
    }
    mldr::mldr_from_dataframe(mdata$dataset[attributesIndexes], labelsIndexes, mdata$name)
}

#' @title Remove examples without labels
#' @description Remove the examples that there are not labels.
#'
#' @param mdata The mldr dataset to remove.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_remove_unlabeled_instances(emotions)
mldr_remove_unlabeled_instances <- function(mdata) {
    labelset <- rep(0, mdata$measures$num.labels)
    rows <- !apply(mdata$dataset[mdata$labels$index] == labelset, 1, all)
    cols <- seq(mdata$measures$num.attributes)
    mldr::mldr_from_dataframe(mdata$dataset[rows, cols], mdata$labels$index, mdata$name)
}

#' @title Remove unusual or very common labels
#' @description Remove the labels that have smaller or higher examples based on
#'  a specific threshold value.
#'
#' @param mdata The mldr dataset to remove.
#' @param t Threshold value.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_remove_labels(emotions)
mldr_remove_labels <- function(mdata, t = 1) {
    labelsIndexes <- c()
    for (col in mdata$labels$index) {
        tbl <- table(mdata$dataset[col])
        if (length(tbl) > 1 && all(tbl > t)) {
            labelsIndexes <- c(labelsIndexes, col)
        }
    }

    if (length(labelsIndexes) <= 1)
        stop("The pre process procedure result in a single label")

    dataset <- mdata$dataset[sort(c(mdata$attributesIndexes, labelsIndexes))]
    mldr::mldr_from_dataframe(dataset, which(colnames(dataset) %in% rownames(mdata$labels)), mdata$name)
}

#' @title Replace nominal attributes
#' @description Replace the nominal attributes by binary attributes.
#'
#' @param mdata The mldr dataset to remove.
#' @param ordinal.attributes Not yet, but it will be used to specify which attributes need to be replaced.
#'
#' @return a new mldr object.
#' @export
#'
#' @examples
#' mldr_replace_nominal_attributes(emotions)
mldr_replace_nominal_attributes <- function(mdata, ordinal.attributes = list()) {
    # TODO ordinal.attributes
    replace_nominal_column <- function(column, column.name = "", type = 1) {
        column <- as.factor(column)
        symbols <- levels(column)
        result <- {
        }
        if (length(symbols) == 2 && type == 1 && 0 %in% symbols && 1 %in% symbols) {
            result <- cbind(result, as.double(column == 1))
            names <- column.name
        } else {
            for (i in 1:(length(symbols) - type)) result <- cbind(result, as.double(column == symbols[i]))
            names <- paste(column.name, symbols[1:(length(symbols) - type)], sep = "_")
        }
        if (column.name != "")
            colnames(result) <- names

        result
    }

    dataset <- data.frame(row.names = rownames(mdata$dataset))
    labelIndexes <- c()
    for (col in seq(mdata$measures$num.attributes)) {
        if (is.numeric(mdata$dataset[, col])) {
            dataset <- cbind(dataset, mdata$dataset[col])
            if (col %in% mdata$labels$index) {
                labelIndexes <- c(labelIndexes, ncol(dataset))
            }
        } else {
            dataset <- cbind(dataset, replace_nominal_column(mdata$dataset[, col], colnames(mdata$dataset[col])))
        }
    }

    mldr::mldr_from_dataframe(dataset, labelIndexes, mdata$name)
}
