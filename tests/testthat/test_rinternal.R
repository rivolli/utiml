context("R Internal tests")

test_that("Normalize", {
  expect_equal(utiml_normalize(1:3), c(0.0, 0.5, 1.0))
  expect_equal(utiml_normalize(1:5), c(0.0, 0.25, 0.5, 0.75, 1.0))
  expect_equal(utiml_normalize(c(1,2,3,4,5), 10, 0), 1:5/10)
})

test_that("Ifelse", {
  c1 <- rep(1, 10)
  c2 <- 1:10
  expect_equal(utiml_ifelse(T, c1, c2), c1)
  expect_equal(utiml_ifelse(F, c1, c2), c2)
  expect_equal(utiml_ifelse(T, c1, NA), c1)
  expect_equal(utiml_ifelse(F, c1, NA), NA)
  expect_equal(utiml_ifelse(T, NA, c2), NA)
  expect_equal(utiml_ifelse(F, NA, c2), c2)
  expect_null(utiml_ifelse(NA, c1, c2))
})
