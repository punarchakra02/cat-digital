import pandas as pd

csv_path = '/home/spongyshaman/Documents/Caterpillar/equipment_rentals.csv'

df = pd.read_csv(csv_path)
if 'realized_price' in df.columns:
    df = df.drop(columns=['realized_price'])
df.to_csv(csv_path, index=False)