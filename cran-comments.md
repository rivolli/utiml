## Test environments
* local Ubuntu install, R 3.2.1
* Ubuntu on Travis CI, R 3.2.4
* win-builder (devel and release)

## R CMD check results
There were no ERRORs or WARNINGs.

There was 1 NOTE:

* Maintainer: ‘Adriano Rivolli <rivolli@utfpr.edu.br>’
New submission
Possibly mis-spelled words in DESCRIPTION:
  Multi (3:22, 8:14)
  multi (9:58)
  
I tried 'multi-label', 'multilabel' and 'multi label' but this note persists 
(only on win-builder).

## Downstream dependencies
There are no downstream dependencies.
