context("CV")

test_that("Test CV returns", {
  res1 <- cv(toyml, br, base.algorithm="RANDOM", cv.folds=10)
  expect_length(res1, 22)

  res2 <- cv(toyml, br, base.algorithm="RANDOM", cv.folds=3, cv.results = TRUE)
  expect_length(res2, 2)
  expect_named(res2, c("multilabel", "labels"))
  expect_equal(dim(res2$multilabel), c(3, length(res1)))
  expect_named(res2$labels, rownames(toyml$labels))
  expect_equal(nrow(res2$labels[[1]]), 3)
})

test_that("Test CV seed", {
  res1 <- cv(toyml, dbr, base.algorithm="RANDOM", cv.folds=4, cv.seed = 123)
  res2 <- cv(toyml, dbr, base.algorithm="RANDOM", cv.folds=4, cv.seed = 123)
  expect_equal(res1, res2)
  set.seed(123)
  res3 <- cv(toyml, dbr, base.algorithm="RANDOM", cv.folds=4)
  expect_equal(res1, res3)

  res4 <- cv(toyml, dbr, base.algorithm="RANDOM", cv.folds=4,
             cv.sampling="iterative", cv.seed = 123)
  expect_false(isTRUE(all.equal(res1, res4)))
})
