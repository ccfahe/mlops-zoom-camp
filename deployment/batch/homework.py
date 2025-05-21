import pickle
import pandas as pd
import sklearn

!wget -P data https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet

year = 2023
month = 3
taxi_type = 'yellow'

input_file = f'data/yellow_tripdata_2023-03.parquet'
output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


!mkdir -p output/yellow

with open('model/model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

!mkdir model

df = read_data(input_file)
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df.head()

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)

y_pred

y_pred.std()

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

!ls -lh output/yellow
