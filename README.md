# utiml: UTIlities for Multi-label Learning
[![Travis-CI Build Status](https://travis-ci.org/rivolli/utiml.svg?branch=master)](https://travis-ci.org/rivolli/utiml)

The utiml package is a framework to support multi-label processing, like Mulan 
on Weka. It is simple to use and extend, then this tutorial explain the main 
topics related with the utiml package.

# Geting started

Load the library:
```{r}
library(utiml)
```

Next, we want to stratification the dataset in two partitions (train and test), 
containing 65% and 35% of instances respectively, then we can do:
```{r}
ds <- create_holdout_partition(toyml, c(train=0.65, test=0.35), "iterative")
```

Now, the `ds` object has two elements `ds$train` and `ds$test`, where the first will
be used to create a model and the second to test the model. For example, using the 
*Binary Relevance* multi-label method with the base classifier *Random Forest*, 
we can do:
```{r}
brmodel <- br(ds$train, "RF", seed=123)
prediction <- predict(brmodel, ds$test)
```

The `prediction` is an object of class `mlresult` that contains the probability (also called confidence or score)
and the bipartitions values:
```{r}
head(as.bipartition(prediction))
head(as.probability(prediction))
head(as.ranking(prediction))
```

A threshold strategy can be applied and generate a refined prediction:
```{r}
newpred <- rcut_threshold(prediction, 2)
```

Now we can evaluate the model and compare if the use of MCUT threshold improve the results:
```{r}
result <- multilabel_evaluate(ds$tes, prediction, "bipartition")
thresres <- multilabel_evaluate(ds$tes, newpred, "bipartition")

measures <- c("accuracy", "F1", "precision", "recall", "subset-accuracy")
round(cbind(Default=result, RCUT=thresres), 3)
```

