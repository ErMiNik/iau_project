from pathlib import Path
folder = Path("./070")

data = pd.read_csv(folder / "connection.csv", parse_dates=["ts"])

# EDA
 ## Target ( Y, dependent value, predictor) MWRA column

# kukni properties
    ## shape, columns, data types, missing values, duplicates
## ts -> datetime     pd.to_datetime(date["ts"], format = "$Y-....")
## dt["ts"].dt.wday()
## imei -> string ??   astype("object/string")
# skeweness and kurtosis 
# STANDARD DEVIATION
# Standard error

#### Data Quality Issues
# data type
# missing values
dt[""].isna()  # bool  .isna().sum()

# 2 approache to keep or not to keep that is the question
# drop missin values
# df.loc[~df["ts"].isna(), :].copy()    ~ means NOT

# Imputing of missing values
# normally dist -> mean()
# non - median()
# KNN K-nearest neighbour 

# dubpicates
# full duplicates -> full rows
df.duplicated()   #  df.loc[df.dupllicate(), :]]

df.duplicated(subset = "imei")   # keep = first/last

# oputliers
# clipping  -> 25%*1.5 IQR lower clip
# upper clip: 75%*1.5IQR

# substitution
# np.where(condition, True,Fale)  np.wher(df[] <= 5%, var_5%, df[])
# pd.case_when() mutliple conditions pandas 2.2 +
# np.select([list of conditions], [list of instruction], other = "Other"/100)

# Loc filtering  df2 = df.loc[(df["imei"] > imei_var_5%) | (df["imei"] < imey_var_95%), : ]     | OR   () |& ()

### column characteristics
## numeric
# range and shape  -> anomalies outliers
## character/ string
# value_counts()
# grouping variables .groupby([]).agg()

# df.merge(df2, on=[key columns], how = "inner").merge()


## Relationships
# Visually 
# corr / scatterplot
# Boxplot / ANOVA normnal dist / Man-Whitney non-nmormally dist   p < 0.05
# Chi Square  -> 2 categorical values

# Hypotrhesis

# Null Hypothesis: No relationship
# Alpha value = Significance level => 0.05

# Alternative: There is a relationship
# Pearsons if normal  / Spearsons rank correlation if non-normal
# HIst or Formal test of normality
# Corelation / ANOVA + post-hock test Tukey's post hoc test
# p value  => p < 0.05  => statistically significant reject null hypothesis
    # p > 0.05 cannot reject null hyp



#df["ts"]

#df.groupby("city")["mwra"].agg("mean","min")
print(df.shape)

print(df.info()) # ts -> datetime

df.describe()

df["c.katana"].plot.hist(bins=30)
plt.show()