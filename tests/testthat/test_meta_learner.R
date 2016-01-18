context("Meta-learner tests")

dataml <- toyml$dataset[,names(toyml$attributes)]
brdata <- toyml$dataset[,c(toyml$attributesIndexes, toyml$labels[1,"index"])]
brdata[,ncol(brdata)] <- as.factor(brdata[,ncol(brdata)])

test_that("Simple measures", {
  expect_equal(get_num_att(brdata), toyml$measures$num.inputs)
  expect_equal(get_num_samples(brdata), toyml$measures$num.instances)
  expect_equal(get_dim(brdata), toyml$measures$num.inputs /
                 toyml$measures$num.instances)
  expect_equal(get_num_classes(brdata), 2)
  expect_equal(get_prop_posclass(brdata), toyml$labels[1,"freq"])
  expect_equal(get_prop_binatt(dataml), (toyml$measures$num.labels - 1) /
                 (ncol(dataml)-1))
})

test_that("Statistical measures", {
  expect_equal(length(get_stat_sd(brdata)), 4)
  expect_equal(length(get_stat_varcoef(brdata)), 4)
  expect_equal(length(get_stat_covariance(brdata)), 4)
  expect_equal(length(get_stat_lincorr(brdata)), 4)
  expect_equal(length(get_stat_skewness(brdata)), 4)
  expect_equal(length(get_stat_kurtosis(brdata)), 4)
  expect_equal(length(get_skewness(brdata)), 1)
  expect_equal(length(get_kurtosis(brdata)), 1)
})
