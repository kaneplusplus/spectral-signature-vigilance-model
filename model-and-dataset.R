library(torch)
library(luz)

# The data set acesses the tensors in the fd data set.
SSFD = dataset(
  name = "SSFD",
  initialize = function(fd) {
    self$fd = fd
  },
  .getitem = function(i) {
    # Return the spatial signature tensor and the vigilance status.
    list(
      x = self$fd$tensor[[i]], # |> torch_flatten(start_dim = 2), 
      y = torch_tensor(as.double(self$fd$vigilant[i] == "vigilant"))
    )
  },
  .length = function() {
    nrow(self$fd)
  }
)

# The deep learner.
SSMod = nn_module(
  "SSMod",
  initialize = function() {
    # The network layers are inspired by AlexNet.
    # The first layer is a convolutional nn that works across the 3 channels.
    # The second is max pooling of the output.
    # Third "layer" is a fully connected layer, dropout, sigmoid, and then a
    #   linear layer.
    # A final, fully connected layer that is a single output 1 is vigilant
    # 0 is non-vigilant.
    self$xl = nn_conv2d(3, 1, kernel_size = 3)
    self$mp = nn_max_pool2d(kernel_size = 3)
    self$embed = nn_sequential(
      nn_linear(1824, 1824),
      nn_dropout(),
      nn_sigmoid(),
      nn_linear(1824, 256),
    )
    self$reduce = nn_linear(256, 1)
  },
  mod_embed = function(x) {
    # You can get the embedding of the spectral signature by pushing the 
    # data through all of the layers up through the penultimate one.
    x |>
      self$xl() |>
      self$mp() |>
      torch_flatten(start_dim = 2) |>
      self$embed()
  },
  forward = function(x) {
    ret = x |>
      self$mod_embed() |>
      self$reduce() |>
      (\(x) 1. / (1. + exp(-x)))()
  }
)

# Get the cross-validated output.
make_cv_model = function(
  fd,
  xv_num,
  epochs, 
  train_batch_size,
  num_workers = 0) {
  done = FALSE

  # Train the model.
  tryCatch({
    model = SSMod |>
      setup(
        loss = nn_bce_loss(),
        optimizer = optim_adam,
        metrics = luz_metric_set(
          metrics = luz_metric_binary_accuracy()
        )
      ) |>
      set_opt_hparams(lr = 1e-5) |>
      fit(
        dataloader(
          SSFD(fd |> filter(xv != xv_num)), 
          batch_size = train_batch_size, 
          shuffle = TRUE,
          num_workers = num_workers,
          worker_packages = c("torch", "arrow", "dplyr", "lubridate", "purrr"),
          worker_globals = c("fd")
        )
        ,
        epochs = epochs,
        valid_data = dataloader(
          SSFD(fd |> filter(xv == xv_num)),
          shuffle = FALSE,
          num_workers = num_workers,
          worker_packages = c("torch", "arrow", "dplyr", "lubridate", "purrr"),
          worker_globals = c("fd")
        ) ,
        verbose = FALSE
      )
      done = TRUE
    }, error = function(e) {browser(); gc(); cat("Trying again\n")}
  )

  # Get the out-of-sample prediction accuarcy by calling predict on the
  # validation data (where the cross validation number is fd$xv).
  ret = fd |> filter(xv == xv_num) 
  ret$pred = predict(model, SSFD(ret)) |> 
      as.numeric()

  # Output the accuracy for this training-validation set.
  av = accuracy_vec(
    truth = factor(as.character(as.integer(ret$vigilant == "vigilant")), levels = c("0", "1")),
    estimate = factor(as.character(round(ret$pred)), levels = c("0", "1"))
  )
  cat("Acc ", av, "\n")
  ret
}
