The package was removed from CRAN due to the issue: 
  "The errors in r-devel are from recent changes which lock the base environment
   and its namespace, so that one can no longer create new bindings there." 
It was caused by some tests and internal procedures that changed .Random.seed. 
These problems are fixed now. 
For particular reasons, I could not fix the issue on time.

## Test environments
* local Debian 10 install, R 4.0.3
* Ubuntu Linux 16.04.6 LTS on Travis CI, R version 4.0.2
* win-builder (devel and release)

## R CMD check results
There were no ERRORs and WARNINGs.
There is a NOTE about mis-spelled words and package archived on CRAN.

## Downstream dependencies
There are no downstream dependencies.
