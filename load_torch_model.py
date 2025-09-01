import numpy as np
import pandas as pd
from datetime import datetime, timedelta
 
my_df = pd.read_csv('/home/frinksserver/Documents/timesfm_torch_exp/high_value_revenue_836.csv')
 
 
# Create the `ds` column and rename other columns
# Take the first 4 digits of F_YEAR and convert to string
my_df['F_YEAR_cleaned'] = my_df['F_YEAR'].astype(str).str[:4]
# Pad FWEEK_IN_YEAR with a leading zero if needed
my_df['FWEEK_IN_YEAR_padded'] = my_df['FWEEK_IN_YEAR'].astype(str).str.zfill(2)
 
# Create the 'ds' (date) column
#my_df['ds_string'] = my_df['F_YEAR_cleaned'] + '-' + my_df['FWEEK_IN_YEAR_padded'] + '-1'
#my_df['ds'] = pd.to_datetime(my_df['ds_string'], format='%Y-%W-%w')
def fiscal_week_to_date(f_year, f_week):
    """
    Convert F_YEAR (financial year starting in April) and FWEEK_IN_YEAR
    into a proper datetime (first day of that week).
    """
    # Start of fiscal year = April 1 of F_YEAR
    start_date = datetime(int(f_year), 4, 1)
 
    # Add (f_week-1) weeks
    return start_date + timedelta(weeks=int(f_week)-1)
 
# Apply to your dataframe
my_df['ds'] = my_df.apply(lambda row: fiscal_week_to_date(row['F_YEAR_cleaned'], row['FWEEK_IN_YEAR']), axis=1)
 
 
#indices_to_drop = my_df.groupby('Material').tail(1).index
#my_df = my_df.drop(indices_to_drop)
my_df.rename(columns={'Material': 'unique_id', 'weekly_Order_Quantity': 'y'}, inplace=True)
 
# Assuming your DataFrame is named 'my_df'
my_df.drop(columns=['min_week', 'max_week'], inplace=True)
input_df = my_df[['unique_id', 'ds', 'y']].copy()
 
 
import timesfm
 
tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=9,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          path = "/home/frinksserver/Documents/Main_Times_fm/timesfm_torch/torch_model.ckpt",
          huggingface_repo_id= None),
  )
 
print("model_loading done!")
 
 
 
# Create empty lists to store the data
train_list = []
holdout_list = []
 
# Get a list of all unique materials
unique_ids = input_df['unique_id'].unique()
 
for uid in unique_ids:
    # Filter the DataFrame for the current unique ID and sort by date
    df_for_split = input_df[input_df['unique_id'] == uid].sort_values('ds')
 
    # Get the holdout data (the last 4 weeks)
    #holdout_df = df_for_split.tail(5)
    #holdout_list.append(holdout_df)
 
    # Get the training data (all but the last 4 weeks)
   # train_df = df_for_split.head(len(df_for_split) - 5)

    train_list.append(df_for_split)
 
# Concatenate the lists back into DataFrames
train_df = pd.concat(train_list)
#holdout_df = pd.concat(holdout_list)
 
print("Training Data:")
print(train_df.tail())
#print("\nHoldout Data:")
#print(holdout_df.head())
 
import numpy as np
 
# Assuming you have already created the `train_df` and `holdout_df`
# from the previous step.
 
# Perform the forecast on the training data
forecast_df = tfm.forecast_on_df(
    inputs=train_df,
    freq='W',
    value_name='y',
    num_jobs=-1,
)
 
# Rename the forecast column to avoid confusion
#forecast_df.rename(columns={'y': 'y_forecast'}, inplace=True)
 
# Merge the forecast with the actual holdout data for comparison
#validation_df = pd.merge(
  #  holdout_df,
   # forecast_df[['unique_id', 'ds', 'y_forecast']],
  #  on=['unique_id', 'ds'],
   # how='left'
#)
 
# Calculate the Absolute Percentage Error for each row, handling division by zero
def calculate_mape_error(actual, forecast):
    # Avoid division by zero by checking if the actual value is 0
    return np.where(actual != 0, np.abs((actual - forecast) / actual) * 100, 0)
 
###)
 
# Calculate the Mean Absolute Percentage Error (MAPE) for each unique_id
#mape_per_id = validation_df.groupby('unique_id')['abs_percentage_error'].mean()
 
#print("\nMean Absolute Percentage Error (MAPE) per Material:")
#print(mape_per_id)

# Save as CSV
forecast_df.to_csv("forecast_results_aug_sep.csv", index=False)
 
 
 