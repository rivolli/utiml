
"[.mldr" <- function(mldrObject, rowFilter = T) {
    rowFilter <- substitute(rowFilter)
    rows <- eval(rowFilter, mldrObject$dataset, parent.frame())
    newDataset <- mldrObject$dataset[rows, seq(mldrObject$measures$num.attributes)]
    
    mldr_from_dataframe(newDataset, labelIndices = mldrObject$labels$index, name = mldrObject$name)
} 
