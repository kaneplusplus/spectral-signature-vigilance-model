library(dplyr)
library(future)
library(purrr)
library(tibble)
library(torch)
library(tidyr)
library(ggplot2)
library(yardstick)

print(Sys.time())

# Load the model and dataset 
source("model-and-dataset.R")

torch_manual_seed(1)

training_batch_size = 128 #64
#num_folds = 20 # 187
num_epochs = 300

# `fd` holds the participant identifier (eid) and path to the tensor file
# for a 5 minute bout.
fd = tibble(
  fn = system("ls five-minute-day-tensors", intern = TRUE) |>
    sprintf("five-minute-day-tensors/%s", ...=_),
  eid= fn |> 
    strsplit("/") |>
    map_chr(~.x[2]) |>
    strsplit("-") |>
    map_chr(~.x[1]) |>
    as.integer(),
  id = fn |>
    strsplit("/") |>
    map_chr(~.x[2]) |>
    gsub(".pq", "", x = _),
)

# `x` holds the (propensity-score) matched participants.
x = readRDS("matched-vigilance.rds") |>
  filter(eid %in% fd$eid)

# join the tensor paths with the participant data.
fd = left_join(fd, x, by = "eid")

# Load the tensors, put them in a list column in the `fd` tibble.
fd$tensor = map(fd$fn, ~ torch_load(.x))

# Spectral Signature five minute intervals by day.
device = "mps"

# Loss is binary cross entropy with an epsilon to take care of the cases
# of numerical stability
my_loss = function(input, target) {
  target = target$reshape(target$shape[-2])
  eps = 1e-8
  ret = torch_mean(
    -(target * log(input + eps) + (1-target) * log(1 - input + eps))
  )
  if (any(is.nan(as.numeric(ret$to(device = "cpu"))))) {
    ret = torch_scalar_tensor(10., device = "mps")
  }
  ret
}

for (num_folds in c(10, 50, 100, 150, 187)) {
  # Nest by the eid, give each eid a cross-validation fold (xv), which will
  # be the row number mod the number of folds plus 1.
  # cross validation.
  fdn = fd |>
    group_nest(eid) |>
    mutate(xv = (row_number() %% num_folds) + 1)

  set.seed(1)

  # Shuffle the rows of the nested data frame and then unnest.
  fdn$xv = sample(fdn$xv)
  fdl = unnest(fdn, data)

  # For each of the fols, create the nn-model, build the model and get the
  # out-of-sample predictions, and the truth.
  cv_res = map_dfr(
    seq_len(num_folds),
  #  1:10,
    ~ {
      mm = make_cv_model(fdl, .x, epochs = num_epochs, training_batch_size)
      mm$truth = mm$vigilant == "vigilant"
      gc()
      mm
    }
  )

  # Turn the "truth" into a numeric value.
  cv_res$truth = as.numeric(cv_res$vigilant == "vigilant")

  # Write the estimates and truths for us to present later.
  saveRDS(cv_res, sprintf("cv_res-%s-folds.rds", num_folds))
  readr::write_csv(cv_res, sprintf("cv_res-%s.csv", num_folds))

  # Show the balanced accuracy.
  print(paste("folds", num_folds))
  library(yardstick)
  bal_accuracy_vec(
    factor(as.integer(cv_res$truth), levels = c("0", "1")), 
    factor(round(cv_res$pred), levels = c("0", "1"))
  ) |> print()
  accuracy_vec(
    factor(as.numeric(cv_res$truth), levels = c("0", "1")), 
    factor(round(cv_res$pred), levels = c("0", "1"))
  ) |> print()
}

print(Sys.time())
