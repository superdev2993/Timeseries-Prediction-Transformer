#Initialize NumerAPI - the offical Python API client for Numerai
from numerapi import NumerAPI
import json
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from numerai_tools.scoring import numerai_corr, correlation_contribution




napi = NumerAPI()

#list the datasets and available versions
all_datasets = napi.list_datasets()
dataset_versions = list(set(d.split('/')[0] for d in all_datasets))
print("Available versions:\n", dataset_versions)

#set data versions to one of the latest datasets
DATA_VERSION = "v5.2"

#Print all files available for download for our version
current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]
print("Available", DATA_VERSION, "files:\n", current_version_files)

#download the feature metadata file
napi.download_dataset(f"{DATA_VERSION}/features.json")

#read the metadata and display
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))

for metadata in feature_metadata:
    print(metadata, len(feature_metadata[metadata]))

feature_sets = feature_metadata["feature_sets"]
# for feature_set in ["small", "medium","all"]:
#     print(feature_set, len(feature_sets[feature_set]))

#Define our feature set
feature_set = feature_sets["small"]
#use "medium" or "all" for better performance. Requires more RAM.
#features = feature_metadata["feature_sets"]["medium"]
#features = feature_metadata["feature_sets"]["all"]

#Download the training data - this will take a few minutes
napi.download_dataset(f"{DATA_VERSION}/train.parquet")

#Load only the "medium" feature set to
#use the "all" feature set to use all features
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era", "target"] + feature_set
)

#Downsample to every 4th era to reduce memory usage and speedup model training (suggested for Colab free tier)
#Comment out the line below to use all the data
train = train[train["era"].isin(train["era"].unique()[::4])]

#Plot the number of rows per era
train.groupby("era").size().plot(
    title="Number of rows per era",
    figsize=(5,3),
    xlabel="Era"
)

#Plot density histogram of the target
train["target"].plot(
    kind="hist",
    title="Target",
    figsize=(5,3),
    xlabel="Value",
    density=True,
    bins=50
)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3))
first_era = train[train["era"] == train["era"].unique()[0]]
last_era = train[train["era"] == train["era"].unique()[-1]]
last_era[feature_set[-1]].plot(
    title="5 equal bins",
    kind="hist",
    density=True,
    bins=50,
    ax = ax1
)
first_era[feature_set[-1]].plot(
    title="missing data",
    kind="hist",
    density=True,
    bins = 50,
    ax =ax2
)


# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
model = lgb.LGBMRegressor(
  n_estimators=2000,
  learning_rate=0.01,
  max_depth=5,
  num_leaves=2**5-1,
  colsample_bytree=0.1
)
# We've found the following "deep" parameters perform much better, but they require much more CPU and RAM
# model = lgb.LGBMRegressor(
#     n_estimators=30_000,
#     learning_rate=0.001,
#     max_depth=10,
#     num_leaves=2**10,
#     colsample_bytree=0.1
#     min_data_in_leaf=10000,
# )

# This will take a few minutes 🍵
model.fit(
  train[feature_set],
  train["target"]
)


# Download validation data - this will take a few minutes
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

# Load the validation data and filter for data_type == "validation"
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type", "target"] + feature_set
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
# Comment out the line below to use all the data (slower and higher memory usage, but more accurate evaluation)
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate predictions against the out-of-sample validation features
# This will take a few minutes 🍵
validation["prediction"] = model.predict(validation[feature_set])
validation[["era", "prediction", "target"]]
print(validation)


# Download and join in the meta_model for the validation eras
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]

# Compute the per-era corr between our predictions and the target values
per_era_corr = validation.groupby("era").apply(
    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
)

# Compute the per-era mmc between our predictions, the meta model, and the target values
per_era_mmc = validation.dropna().groupby("era").apply(
    lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
)


# Plot the per-era correlation
per_era_corr.plot(
  title="Validation CORR",
  kind="bar",
  figsize=(8, 4),
  xticks=[],
  legend=False,
  snap=False
)
per_era_mmc.plot(
  title="Validation MMC",
  kind="bar",
  figsize=(8, 4),
  xticks=[],
  legend=False,
  snap=False
)

# Compute performance metrics
corr_mean = per_era_corr.mean()
corr_std = per_era_corr.std(ddof=0)
corr_sharpe = corr_mean / corr_std
corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

mmc_mean = per_era_mmc.mean()
mmc_std = per_era_mmc.std(ddof=0)
mmc_sharpe = mmc_mean / mmc_std
mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

pd.DataFrame({
    "mean": [corr_mean, mmc_mean],
    "std": [corr_std, mmc_std],
    "sharpe": [corr_sharpe, mmc_sharpe],
    "max_drawdown": [corr_max_drawdown, mmc_max_drawdown]
}, index=["CORR", "MMC"]).T