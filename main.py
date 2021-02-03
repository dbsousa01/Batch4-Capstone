from data_processing import processing as pc

df = pc.load_data()
df = pc.build_outcome_label(df)
print(df["label"].unique())

