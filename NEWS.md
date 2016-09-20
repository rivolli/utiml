# Changelog

## utiml 0.1.1.9020
Development version (only available via github install)

### Major changes
* lcard threshold calibration
* Use categorical attributes in multilabel datasets and methods
* LIFT multi-label classification method

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
* Pre-process utilities: fill sparce data; normalize data; remove attributes; 
   remove labels; remove skewness labels; remove unique attributes; 
   remove unlabeled instances; and, replace nominal attributes
* Sampling methods: Create subsets of multi-label dataset; 
   create holdout and k-fold partitions; and, stratification methods
* Threshold methods: Fixed threshold; MCUT; PCUT; RCUT; SCUT; and, subset correction
* Synthetic dataset: toyml
