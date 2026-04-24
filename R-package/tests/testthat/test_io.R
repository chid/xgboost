context("Test model IO.")

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
train <- agaricus.train
test <- agaricus.test

test_that("load/save raw works", {
  nrounds <- 8
  booster <- xgb.train(
    data = xgb.DMatrix(train$data, label = train$label, nthread = 1),
    nrounds = nrounds,
    params = xgb.params(
      objective = "binary:logistic",
      nthread = 2
    )
  )

  json_bytes <- xgb.save.raw(booster, raw_format = "json")
  ubj_bytes <- xgb.save.raw(booster, raw_format = "ubj")

  from_json <- xgb.load.raw(json_bytes)
  from_ubj <- xgb.load.raw(ubj_bytes)

  json2ubj <- xgb.save.raw(from_json, raw_format = "ubj")
  ubj2ubj <- xgb.save.raw(from_ubj, raw_format = "ubj")

  expect_equal(json2ubj, ubj2ubj)
})

test_that("saveRDS preserves C and R attributes", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = xgb.params(nthread = 1, max_depth = 2),
    nrounds = 5
  )
  attributes(model)$my_attr <- "qwerty"
  xgb.attr(model, "c_attr") <- "asdf"

  fname <- file.path(tempdir(), "xgb_model.Rds")
  saveRDS(model, fname)
  model_new <- readRDS(fname)

  expect_equal(attributes(model_new)$my_attr, attributes(model)$my_attr)
  expect_equal(xgb.attr(model, "c_attr"), xgb.attr(model_new, "c_attr"))
})

test_that("R serializers keep C config", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = list(
      tree_method = "approx",
      nthread = 1,
      max_depth = 2
    ),
    nrounds = 3
  )
  model_new <- unserialize(serialize(model, NULL))
  expect_equal(
    xgb.config(model)$learner$gradient_booster$gbtree_train_param$tree_method,
    xgb.config(model_new)$learner$gradient_booster$gbtree_train_param$tree_method
  )
  expect_equal(variable.names(model), variable.names(model_new))
})

test_that("xgb.Booster objects are read-only list wrappers", {
  data(mtcars)
  model <- xgboost(mtcars[, -1], mtcars$mpg, nthreads = 1, nrounds = 3)

  expect_error(model[1] <- list(model[[1]]), "read-only")
  expect_error(model[[1]] <- model[[1]], "read-only")
  expect_error(model[[2]] <- 1, "read-only")
  expect_error(model$foo <- 1, "read-only")
  expect_error(names(model) <- "foo", "read-only")
  expect_error(unname(model), "read-only")
  expect_error({
    alt_obj <- unclass(model)
    alt_obj[[1]] <- 1
  }, "read-only")

  bad <- list(ptr = model[[1]], extra = 1)
  class(bad) <- class(model)

  expect_error(predict(bad, as.matrix(mtcars[, -1])), "corrupted|blank 'externalptr'")
})

test_that("data.table accepts xgb.Booster in list columns and copies", {
  data(mtcars)
  x_issue <- mtcars[, -1]
  model_issue <- xgboost(x_issue, mtcars$mpg, nthreads = 1, nrounds = 3)
  dt_issue <- data.table::data.table(model = list(model_issue))
  expect_s3_class(dt_issue$model[[1]], "xgb.Booster")
  expect_equal(
    predict(dt_issue$model[[1]], as.matrix(x_issue)),
    predict(model_issue, as.matrix(x_issue))
  )

  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  model <- xgb.train(
    data = xgb.DMatrix(x, label = y, nthread = 1),
    params = xgb.params(nthread = 1, max_depth = 2),
    nrounds = 3
  )

  dt <- data.table::data.table(model = list(model))
  expect_s3_class(dt$model[[1]], "xgb.Booster")
  expect_equal(
    predict(dt$model[[1]], x),
    predict(model, x)
  )

  dt_copy <- data.table::copy(dt)
  expect_s3_class(dt_copy$model[[1]], "xgb.Booster")
  expect_equal(
    predict(dt_copy$model[[1]], x),
    predict(model, x)
  )

  fname <- file.path(tempdir(), "xgb_bst_dt.Rds")
  saveRDS(dt, fname)
  dt_new <- readRDS(fname)

  expect_s3_class(dt_new$model[[1]], "xgb.Booster")
  expect_equal(
    predict(dt_new$model[[1]], x),
    predict(model, x)
  )
})
