import pandas as pd

# create a sample DataFrame
df = pd.DataFrame({'col1': ['a', 'b', 'c$', 'd'], 'col2': [1, 2, 3, 4]})

# display the original DataFrame
print('Original DataFrame:')
print(df)

# create a boolean mask to select rows that contain the special character
mask = df['col1'].str.contains('\$')

# select the rows to delete
rows_to_delete = df[mask].index

# drop the selected rows
df = df.drop(rows_to_delete)

# display the updated DataFrame
print('Updated DataFrame:')
print(df)