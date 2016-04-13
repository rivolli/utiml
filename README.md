# utiml: UTIlities for Multi-label Learning
[![Travis-CI Build Status](https://travis-ci.org/rivolli/utiml.svg?branch=master)](https://travis-ci.org/rivolli/utiml)

The utiml package is a framework to support multi-label processing, like Mulan 
on Weka. 

The main methods available on this package are organized in the groups:
- Classification methods
- Evaluation methods
- Pre-process utilities
- Sampling methods
- Threshold methods

# Instalation
The installation process is similar to other packages available on CRAN:
```r
install.packages("utiml")
```

This will also install [mldr](https://cran.r-project.org/package=mldr).
To run the examples in this document, you also need to install the packages:
```r
# Base classifiers (SVM and Random Forest)
install.packages(c("e1071", "randomForest"))
```

## Install via github (development version)
```r
devtools::install_github("rivolli/utiml")
```

# Multi-label Classification
## Running Binary Relevance Method
```{r}
library(utiml)

# Create two partitions (train and test) of toyml multi-label dataset
ds <- create_holdout_partition(toyml, c(train=0.65, test=0.35))

# Create a Binary Relevance Model using e1071::svm method
brmodel <- br(ds$train, "SVM", seed=123)

# Predict
prediction <- predict(brmodel, ds$test)

# Show the predictions
head(as.bipartition(prediction))
head(as.ranking(prediction))

# Apply a threshold
newpred <- rcut_threshold(prediction, 2)

# Evaluate the models
result <- multilabel_evaluate(ds$tes, prediction, "bipartition")
thresres <- multilabel_evaluate(ds$tes, newpred, "bipartition")

# Print the result
print(round(cbind(Default=result, RCUT=thresres), 3))
```

## Running Ensemble of Classifier Chains
```{r}
library(utiml)

# Create three partitions (train, val, test) of emotions dataset
partitions <- c(train = 0.6, val = 0.2, test = 0.2)
ds <- create_holdout_partition(emotions, partitions, method="iterative")

# Create an Ensemble of Classifier Chains using Random Forest (randomForest package)
eccmodel <- ecc(ds$train, "RF", m=3, cores=parallel::detectCores(), seed=123)

# Predict
val <- predict(eccmodel, ds$val, cores=parallel::detectCores())
test <- predict(eccmodel, ds$test, cores=parallel::detectCores())

# Apply a threshold
thresholds <- scut_threshold(val, ds$val, cores=parallel::detectCores())
new.val <- fixed_threshold(val, thresholds)
new.test <- fixed_threshold(test, thresholds)

# Evaluate the models
measures <- c("subset-accuracy", "F1", "hamming-loss", "macro-based") 

result <- cbind(
  Test = multilabel_evaluate(ds$tes, test, measures),
  TestWithThreshold = multilabel_evaluate(ds$tes, new.test, measures),
  Validation = multilabel_evaluate(ds$val, val, measures),
  ValidationWithThreshold = multilabel_evaluate(ds$val, new.val, measures)
)

print(round(result, 3))
```

More examples and details are available on functions documentations and vignettes, please refer to the documentation.
