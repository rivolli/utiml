# Changelog

## utiml 0.1.4

### New Features
* MLKNN algorithm
* ranking-loss baseline
* label problem evaluation measures

### Bug fixes
* Stratification sampling to support instances without labels

## utiml 0.1.3
### Major changes
* Change `multilabel_evaluation` to also return the label measures

### Bug fixes
* Bugfix in `brplus` because the newfeatures were using different levels
* Fix `baseline` using hamming-loss to prevent empty label prediction
* Fix empty prediction when all labels have the same probability

### Minor changes
* Fix type mistakes in documentation

## utiml 0.1.2

### Major changes
* change base.method parameter name for base.algorithm

### Bug fixes
* Bugfix in `homer` to deal with labels without intances and to predict instances 
   based on the meta-label scores
* Refactory of merge_mlconfmat
* Ensure reproducibility in all cases


## utiml 0.1.1
New multi-label transformation methods including pairwise and multiclass 
approaches. Some fixes from previous version.

### Major changes
* lcard threshold calibration
* Use categorical attributes in multilabel datasets and methods
* LIFT multi-label classification method
* RPC multi-label classification method
* CRL multi-label classification method
* LP multi-label classification method
* RAkEL multi-label classification method
* BASELINE multi-label classification method
* PPT multi-label classification method
* PS multi-label classification method
* EPS multi-label classification method
* HOMER multi-label classification method

### Minor changes
* Add Empty Model as base method to fix training labels with few examples
* `multilabel_confusion_matrix` accepts a data.frame or matrix with the predicitons
* Change EBR and ECC to use threshold calibration
* Include empty.prediction configuration to enable/disable empty predictions

### Bug fixes
* Majority Ensemble Predictions Votes
* Majority Ensemble Predictions Probability
* Base method not found message error
* Base method support any attribute names
* Normalize data ignore attributes with a single value
* MBR support labels without positive examples
* Fix average precision and coverage measures to support instances without labels

## utiml 0.1.0

First release of **utiml**:

* Classification methods: `Binary Relevance (BR)`; `BR+`; `Classifier Chains`;
  `ConTRolled Label correlation exploitation (CTRL)`; `Dependent Binary Relevance (DBR)`;
  `Ensemble of Binary Relevance (EBR)`; `Ensemble of Classifier Chains (ECC)`;
  `Meta-Binary Relevance (MBR or 2BR)`; `Nested Stacking (NS)`; 
  `Pruned and Confident Stacking Approach (Prudent)`; and, `Recursive Dependent Binary Relevance (RDBR)`
* Evaluation methods: Create a multi-label confusion matrix and multi-label measures
* Pre-process utilities: fill sparse data; normalize data; remove attributes; 
   remove labels; remove skewness labels; remove unique attributes; 
   remove unlabeled instances; and, replace nominal attributes
* Sampling methods: Create subsets of multi-label dataset; 
   create holdout and k-fold partitions; and, stratification methods
* Threshold methods: Fixed threshold; MCUT; PCUT; RCUT; SCUT; and, subset correction
* Synthetic dataset: toyml
