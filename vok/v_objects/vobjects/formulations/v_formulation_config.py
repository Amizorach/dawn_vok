class FormulationConfig:


    aggregation_functions_info = [
        {
            "key": "min",
            "name": "Minimum Value",
            "syntax_directives": [
                "Computes the smallest value in the dataset.",
                "Helps to determine the lower bound of the data values.",
                "Often used in filtering or threshold determination.",
                "Handles missing values according to the library's rules.",
                "Applies to numeric, date, or other comparable data types."
            ]
        },
        {
            "key": "max",
            "name": "Maximum Value",
            "syntax_directives": [
                "Identifies the largest value in the dataset.",
                "Determines the upper bound of the data range.",
                "Useful for detecting outliers on the higher end.",
                "Supports various data types such as numbers and dates.",
                "Often paired with min to assess the span of the data."
            ]
        },
        {
            "key": "mean",
            "name": "Mean (Average)",
            "syntax_directives": [
                "Calculates the arithmetic average of the values.",
                "Sums all values and divides by the count of elements.",
                "Sensitive to outliers which can skew the result.",
                "A common measure of central tendency.",
                "Requires numerical input for meaningful computation."
            ]
        },
        {
            "key": "median",
            "name": "Median Value",
            "syntax_directives": [
                "Finds the middle value of a sorted dataset.",
                "Offers a robust central tendency measure less affected by outliers.",
                "Ideal for skewed distributions in comparative analysis.",
                "Provides a single, representative value when data are ordered.",
                "Often used alongside the mean for distribution insights."
            ]
        },
        {
            "key": "sum",
            "name": "Sum of Values",
            "syntax_directives": [
                "Adds all values in the dataset to return a total.",
                "Useful for calculating cumulative totals or overall impact.",
                "Requires numeric data to function correctly.",
                "Often used in conjunction with count or average computations.",
                "Helps in financial, statistical, and operational analyses."
            ]
        },
        {
            "key": "count",
            "name": "Count of Values",
            "syntax_directives": [
                "Tallies the number of non-null elements in the dataset.",
                "Useful for assessing the volume of valid observations.",
                "Ignores missing or null values during computation.",
                "Foundation for ratio and average calculations.",
                "Helps in evaluating data completeness."
            ]
        },
        {
            "key": "std",
            "name": "Standard Deviation",
            "syntax_directives": [
                "Calculates the standard deviation, a measure of dispersion.",
                "Quantifies the average distance of each value from the mean.",
                "Helps assess variability and consistency within the data.",
                "Sensitive to extreme values which can inflate the measure.",
                "Frequently used in statistical quality control and risk assessment."
            ]
        },
        {
            "key": "var",
            "name": "Variance",
            "syntax_directives": [
                "Computes the variance, representing the spread of data.",
                "Measures the average squared deviation from the mean.",
                "Provides insight into data volatility or consistency.",
                "Can be calculated as a biased or unbiased estimator.",
                "Often used together with standard deviation for comprehensive analysis."
            ]
        },
        {
            "key": "prod",
            "name": "Product of Values",
            "syntax_directives": [
                "Multiplies all values in the dataset to yield a product.",
                "Useful for calculating cumulative multiplicative effects.",
                "Sensitive to zero: a single zero can nullify the product.",
                "Requires numeric inputs for valid computation.",
                "Commonly applied in growth rates or compounded interest calculations."
            ]
        },
        {
            "key": "first",
            "name": "First Value",
            "syntax_directives": [
                "Extracts the first element in the dataset or group.",
                "Often used with time series or ordered data to indicate a starting value.",
                "Does not perform any transformation beyond selection.",
                "Helps capture the initial state of a dataset.",
                "Returns the same data type as the original element."
            ]
        },
        {
            "key": "last",
            "name": "Last Value",
            "syntax_directives": [
                "Retrieves the final element from the dataset or group.",
                "Used in sequential or time-series analysis to denote an endpoint.",
                "Focuses solely on the end value without additional computation.",
                "Assists in end-of-period evaluations or trend analysis.",
                "Maintains the original format and data type of the element."
            ]
        },
        {
            "key": "nunique",
            "name": "Unique Count",
            "syntax_directives": [
                "Counts the number of unique, non-null values present.",
                "Useful for determining the diversity of categorical data.",
                "Ignores duplicate entries to provide a distinct count.",
                "Helps in data quality and exploratory analysis.",
                "Supports identifying uniqueness in sets of data."
            ]
        },
        {
            "key": "mode",
            "name": "Mode (Most Frequent)",
            "syntax_directives": [
                "Identifies the most frequently occurring value(s) in a dataset.",
                "May return multiple values if there is a tie for frequency.",
                "Provides insight into the central tendency for categorical data.",
                "Useful when the most common occurrence is more meaningful than an average.",
                "Often employed in statistical summaries and anomaly detection."
            ]
        },
        {
            "key": "quantile",
            "name": "Quantile Value",
            "syntax_directives": [
                "Computes the value below which a specified percentage of observations fall.",
                "Flexible tool to calculate percentiles (e.g., 25th, 50th, 75th percentiles).",
                "Assists in understanding the distribution and spread of data.",
                "Frequently used for risk management and outlier analysis.",
                "Requires a quantile parameter to specify the desired fraction."
            ]
        },
        {
            "key": "skew",
            "name": "Skewness",
            "syntax_directives": [
                "Measures the asymmetry of the data distribution.",
                "Indicates if the data are skewed to the left (negative) or right (positive).",
                "Helps in understanding the shape and balance of the dataset.",
                "Critical for choosing the proper statistical analysis method.",
                "Sensitive to extreme values and outliers."
            ]
        },
        {
            "key": "kurt",
            "name": "Kurtosis",
            "syntax_directives": [
                "Calculates kurtosis, indicating the 'tailedness' of the distribution.",
                "Assesses whether the data have heavy tails or light tails compared to a normal distribution.",
                "Useful in financial and risk analysis to predict the likelihood of extreme outcomes.",
                "Provides additional insight beyond variance and standard deviation.",
                "Typically reported as excess kurtosis relative to a normal distribution."
            ]
        }
    ]
