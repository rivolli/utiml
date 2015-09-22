mldr_remove_unique_attributes <- function (mdata) {
  attributesIndexes <- which(apply(mdata$dataset[mdata$attributesIndexes], 2, function (col) length(unique(col)) > 1))
  dataset <- cbind(mdata$dataset[attributesIndexes], mdata$dataset[mdata$labels$index])
  mldr_from_dataframe(dataset, (length(attributesIndexes) + 1):ncol(dataset), mdata$name)
}

mldr_remove_unusual_labels <- function (mdata, t = 1) {
  labelsIndexes <- which(apply(mdata$dataset[mdata$labels$index], 2, function (col) {
    tbl <- table(col)
    length(tbl) > 1 && all(tbl > t)
  })) + mdata$measures$num.attributes - mdata$measures$num.labels

  dataset <- cbind(mdata$dataset[mdata$attributesIndexes], mdata$dataset[labelsIndexes])
  mldr_from_dataframe(dataset, (1 + ncol(dataset) - length(labelsIndexes)):ncol(dataset), mdata$name)
}

mldr_remove_unlabeled_instances <- function (mdata) {
  labelset <- rep(0, mdata$measures$num.labels)
  rows <- !apply(mdata$dataset[mdata$labels$index] == labelset, 1, all)
  cols <- c(mdata$attributesIndexes, mdata$labels$index)
  mldr_from_dataframe(mdata$dataset[rows, cols], mdata$labels$index, mdata$name)
}

mldr_fill_sparce_data <- function (mdata) {
  dataset <- data.frame(row.names = rownames(mdata$dataset))
  dataset <- cbind(dataset, lapply(mdata$dataset[,1:mdata$measures$num.attributes], function (col){
    if (sum(!complete.cases(col)) > 0) {
      #Has NA value
      if (is.numeric(col)) {
        #Numeric value - fill with 0
        col[is.na(col)] <- 0
      }
      else if (sum(!is.na(as.numeric(col[!is.na(col)]))) > 0) {
        #Text but with numeric values - convert to numeric and fill with 0
        col[is.na(col)] <- "0"
        col <- as.numeric(col)
      }
      else {
        #Text value - fill with ""
        col[is.na(col)] <- ""
      }
    }
    col
  }))

  mldr_from_dataframe(dataset, mdata$labels$index, mdata$name)
}

mldr_normalize <- function(mdata) {
  data <- mdata$dataset[c(mdata$attributesIndexes, mdata$labels$index)]
  for (col in mdata$attributesIndexes) {
    if (is.numeric(data[,col])) {
      min_v = min(data[col])
      max_v = max(data[col])
      d <- (max_v - min_v)
      if(is.na(d) || d == 0) {
        d <- 1.0
      }
      data[col] <- (data[col] - min_v) / d
    }
  }
  mldr_from_dataframe(data, mdata$labels$index, mdata$name)
}

mldr_replace_nominal_attributes <- function(mdata, ordinal.attributes = list()) {
  #TODO ordinal.attributes
  dataset <- data.frame(row.names=rrownames(mdata$dataset))
  for(i in mdata$attributesIndexes) {
    dataset <- if (is.numeric(mdata$dataset[,i]))
       cbind(dataset, mdata$dataset[i])
    else
      dataset <- cbind(dataset, mldr_replace_nominal_column(mdata$dataset[,i], colnames(mdata$dataset[i])))
  }

  classIndexes <- (1 + ncol(dataset)):(ncol(dataset)+mdata$measures$num.labels)
  mldr_from_dataframe(cbind(result, mdata$dataset[mdata$labels$index]), classIndexes, mdata$name)
}

mldr_replace_nominal_column <- function(column, column.name = '', type = 1) {
  column <- as.factor(column)
  symbols <- levels(column)
  result <- {}
  if (length(symbols) == 2 && type == 1 && 0 %in% symbols && 1 %in% symbols) {
    result <- cbind(result, as.double(column == 1))
    names <- column.name
  }
  else {
    for (i in 1:(length(symbols)-type))
      result <- cbind(result, as.double(column == symbols[i]))
    names <- paste(column.name, symbols[1:(length(symbols)-type)], sep='_')
  }
  if (column.name != '')
    colnames(result) <- names

  result
}
