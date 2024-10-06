# 1. EDA
- celkova analyza dát
    - shape, columns, data types, missing values, duplicates, unique values
    - .info(), .shape, .shape[0] - .dropna().shape[0], .isnull(), .isna(), .unique(),

- analyza atributov
    - numeric
        - range and shape -> anomalies outliers
    - character string
        - value_count()
        - grouping variables .groupby([]).agg()
    - column types
    - attribute types
        - continuous (numeric)
        - discrete (categorical) - nominal vs. ordinal
    - univariate analysis
        - continuous - descriptive statistics (average, median, ...), distributions
        - categorical - number of unique values, frequency of their occurrence
    - bivariate analysis - pair analysis
        - continuous-continuous - dependence, correlation
        - continuous-categorical - difference in the value of the continous attribute depending on the category
        - categorical-categorical - table, ratio of frequency of values
- univariate
    - .describe(), mean, median mode, variance, standard deviation, range, quartile, percentile, inter quartal range, boxplot, histogram, standard error, distribution (.displot()), skewness, kurtosis
    - categorical attributes
        - column graph (more than 3-4 values), pie chart
- bivariate
    - scatter plot, pairplot, correlation, heatmap

# 2. identifikacia problemov a cistenie
- identifikacia
    - column data types, missing values, duplicates, outliers, other problems
- cistenie a riesenie
    - type conversion
    - clipping if NaN / doplnit missing values (mean, median, KNN)
    - duplikaty, prist na to o aky duplikat ide a podla toho sa rozhodnut.
    - outliers, bud clipping alebo substitute (tak ako missing values)

# 3.
- Null hypothesis
- Alpha value = Signigicance level
- Pearsons if normal  / Spearsons rank correlation if non-normal
- Hist or Formal test of normality
- Corelation / ANOVA + post-hock test Tukey's post hoc test
- p value  => p < 0.05  => statistically significant reject null hypothesis
    - p > 0.05 cannot reject null hyp