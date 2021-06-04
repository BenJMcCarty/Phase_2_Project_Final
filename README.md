# Selling Seattle: Impactful Features on House Sell Price 

**When selling your house, one of the highest priorities is getting the best price for it.** You may be a real estate agent looking to advise homeowners on the price range for their home, or to give potential buyers an idea of what prices to expect for particular features (like waterfront locations or multiple bedrooms). Or perhaps you are a homeowner who is interested in doing some renovations to improve the house's value. Either way, you want to know what features have the greatest impact on that value.

But how do you *really* know what features are best? How can you be sure that your intuition or expectations are based on fact and not assumptions? **Turn to the facts for answers!**

---

My project used data from house sales in King County, WA. to determine some of the top features determining the sell price of a home.

* **First, I explored the data using the Pandas package to review the data itself.** I reviewed the different features included in my dataset and their respective values; the descriptive statistics for the numerical data; and the overall size and shape of the data (how many rows and columns).

* **Then, I used Seaborn to plot the data and a fitted linear regression model for each feature against the sell price.** These visualizations helped differentiate which features to treat as *categorical* variables (e.g. features that would be one of a select number of options) versus *continuous* variables (which would have less of a rigid range of values).

* **In order to give more depth to my data, I engineered four new features based on the original features and data.** I determined the age of each house at sale; whether or not a home was renovated; how many years since a renovation; and whether or not a house had a basement.

* **I wanted to make sure all of the features and their data were relevant to apply to my future model.** I performed correlational comparisons to determine which features were too closely related (indicating multicollinearity), which would affect my modeling process later on.

* **To test my assumptions that certain features would have a statistically significant impact on the price, I performed a few statistical tests.** I performed an ANOVA test on our "condition" feature (comparing each ranking of condition against the others) and I ran a two-sample t-test on the "waterfront" and "was_renovated" features as well.

# ** Setting Up the Tools and Data**

## Imports


```python
# Data Handling
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling - Statsmodels
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import statsmodels.api as sms

# Modeling - SKLearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.feature_selection import RFECV

# Settings
%matplotlib inline
plt.style.use('seaborn-talk')
pd.set_option('display.max_columns', None)
```

## Defining Functions

I created several functions to help review, clean, test, and evaluate my data.

### Data Cleaning and Visualizations

> The following functions enable the user to filter a Pandas series and return a boolean index to use for filtering out the outliers. Once saved to a new variable, the user can perform further filtering by using the new variable to slice the dataframe to be filtered.

#### ƒ: `find_outliers_z`


```python
## Check using z-score - sensitive to outliers

def find_outliers_z(data):
    """Detects outliers using the Z-score>3 cutoff.
    Returns a boolean Series where True=outlier
    
    Source: https://github.com/jirvingphd/dsc-phase-2-project/blob/main/functions_SG.py
    """
    
    zFP = np.abs(stats.zscore(data))
    zFP = pd.Series(zFP, index=data.index)
    idx_outliers = zFP > 3
    return idx_outliers
```

#### ƒ: `find_outliers_IQR`


```python
## Check using IQR - less sensitive to outliers

def find_outliers_IQR(data):
    """
    * Takes a series sliced from a dataframe
    * Detects outliers using the 1.5*IQR thresholds.
    * Returns a boolean Series where True=outlier

    Source: https://github.com/jirvingphd/dsc-phase-2-project/blob/main/functions_SG.py
    """
    
    res = data.describe()
    q1 = res['25%']
    q3 = res['75%']
    thresh = 1.5*(q3-q1)
    idx_outliers =(data < (q1-thresh)) | (data > (q3+thresh))
    return idx_outliers
```

#### ƒ: `feature_vis`


```python
def feature_vis(data, x, y = 'price', discrete = False, kde = True):
    '''-----
    * Requires a DataFrame and a column name to process.
    * Keyword arguments specify that the target variable will be "price"
    for this case.
    * For future use, redefine function without predetermined y-value, or 
    reassign.
    
    --
    
    * Args:
        * Data: Pandas DataFrame; data source
        * x: str; column index to specify data
    
    * Kwargs:
        * y = "price"
        * discrete = False
        * kde = true
        
    -----'''
    
    ## Print the slice of the original DataFrame for easy viewing
    
    print(df[x].value_counts().sort_index())
  
    ## Create two plots via Seaborn: one scatter plot with regression line,
    ## then a histogram of the data (with KDE if specified
    
    fig, axs = plt.subplots(ncols=2, figsize= (12,6))
    
    sns.regplot(data=data, x=x, y=y, ax=axs[0])
    sns.histplot(data=data, x=x, discrete=discrete, kde=kde, ax=axs[1])
    
    fig.suptitle(f'{x.title()} vs. {y.title()}', fontsize=16)
    plt.tight_layout();
    
    return
```

#### ƒ: `filter_outliers`


```python
def filter_outliers(data):
    '''------
    
    * Removes outliers from data via "find_outliers_IQR" and saves filtered
    values to the dataframe
    
    ---
    
    * Arg:
        * Data: slice of a dataframe for a specific column header
    
    ------
    '''
   
    idx_out = find_outliers_IQR(data)
 
    cleaned = df[~idx_out]

    print(f'There were {idx_out.sum()} outliers.')
    
    return cleaned
```

#### ƒ: `remove_outliers`


```python
def remove_outliers(data, x):

    idx_out = find_outliers_IQR(data[x])
 
    df_clean = df[~idx_out].copy()
    
    return df_clean
```

#### ƒ: `show_cleaned_vis`


```python
def show_cleaned_vis(data, x, y = 'price', discrete = False, kde = True):
    '''-----
    
    * Combines functions to filter outliers and to create the feature 
        visualizations.
    * Requres 'find_outliers_IQR' and 'feature_vis' to be defined.
    * Returns filtered data and two visualizations - Seaborn regression plot
        and histplot.
    
    ---
    
    * Args:
        * Data: Pandas DataFrame; data source
        * x: str; column index to specify data
    * Kwargs
    
    -----'''
    
    ### Filter outliers first
    
    idx_out = find_outliers_IQR(data[x])
 
    df_cleaned = df[~idx_out].copy()

    print(f'There were {idx_out.sum()} outliers.')
    
    ### Plot Data
    
    
    df_cleaned[x].value_counts().sort_index()
        
    fig, axs = plt.subplots(ncols=2, figsize= (12,6))
    
    sns.regplot(data=df_cleaned, x=x, y=y, ax=axs[0])
    sns.histplot(data=df_cleaned, x=x, discrete=discrete, kde=kde, ax=axs[1])
    
    fig.suptitle(f'{x.title()} vs. {y.title()}', fontsize=16)
    plt.tight_layout();
    
    return #df_cleaned
```

### Creating Function for T-Testing

#### ƒ: `ttest_review`


```python
def ttest_review(sample_1, sample_2, alpha=.05):
    '''------
    * Runs a t-test on two samples; prints whether or not they are significant,
    and returns p-value as a variable called "p-value."
    * Requires two data samples and an alpha value.
    
    ----
    
    * Args: two data samples for t-test
    * Kwargs: alpha=.05
    
    -----
    '''
    
    result = stats.ttest_ind(sample_1, sample_2)
    crit_val, p_val = result
    
    ## Creating interpretation based on p-value results.

    if p_value < .05:
        print(f'The feature "waterfront" is statistically significant with a p-value of {p_val}.')

    else:
         print(f'The feature "waterfront" is not statistically significant with a p-value of {p_val}.')
    
    return p_val
```

### Creating a Correlation Matrix

#### ƒ: `"corr_val"`


```python
# Create "corr_val" function

def corr_val(df,figsize=(15,15),cmap="OrRd",):
    
    # Calculate correlations
    corr = df.corr()
       
    # Create a mask of the same size as our correlation data
    mask = np.zeros_like(corr)
    
    # Set the upper values of the numpy array to "True" to ignore them
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=figsize)
    
    # Mask=mask to hide the upper-right half of values (otherwise mirrored)
    sns.heatmap(corr, annot=True,cmap="Reds",mask=mask)
    return fig, ax
```

### Creating and Evaluating Models

#### ƒ: `create_model`


```python
def create_model(data, cont, cat):
    '''------
    
    * Creates a linear regression model using Statsmodels OLS
    * Save the 
    
    ---
    
    * Args:
        * Data: dataframe as data source
        * Cont: list of continuous variables
        * Cat: list of categorical variables
        * unique_id: UNIQUE name for the resulting model
    
    -----
    '''

    cont_features = '+'.join(cont)

    cat_features = '+'.join([f'C({x})' for x in cat])

    f = f'price~+{cont_features}+{cat_features}'

    print(f)

    model = smf.ols(formula=f, data=data).fit()
   
    diagnose_model(model)
    
    return model
```

#### ƒ: `diagnose_model`

>* Create a function to:
>  * Display the summary details of the model
>  * Create a scatter plot of the predictions
>    * Used for determining homoscedasticity
>  * Create a Q-Q plot of the residuals of the model
>    * Used to determine the normality of the residuals



```python
def diagnose_model(model, figsize=(10,5)):
    """ ---
    
    Argument:
        * model: provide the linear regression model for diagnostics
    
    Keyword Argument:
        * figsize: default (10,5); can increase/decrease for larger/smaller
    ---
    
    * Display the summary details of the provided model
    * Create two scatter plots to test assumptions of linearity
        * Predictions: verifying homoscedasticity (no cone-shapes)
        * Residuals: confirming normal distribution of residuals
    ---
    
    """
    display(model.summary())
    
    fig, axes = plt.subplots(ncols=2, figsize=figsize)

    axes[0].scatter(model.predict(), model.resid)
    axes[0].axhline()
    axes[0].set_xlabel('Model Predictions')
    axes[0].set_ylabel('Model Residuals')
    axes[0].set_title('Testing for Homoscedasticity')

    sms.graphics.qqplot(data=model.resid, fit=True, line = "45", ax=axes[1])
    
    plt.tight_layout()
    
    return
```

#### ƒ: `plot_param_coef`

>* Create a function to:
>  * Get the model's coefficients as a series
>  * Plot a figure to show the coefficients in descending order



```python
def plot_param_coef(model, kind = 'barh', figsize = (10,5)):
    ''' ---
    
    * Plotting a figure to visualize parameter coefficients
    
    ---
    
    * Args:
        * Model: linear regression model details to plot
        
    * Kwargs:
        * Kind (default 'barh'): allows different types of plots
        * Size (default (10,10)): allows for different sizes
    ---
    
    '''
    # Plotting figure to visualize parameter coefficients

    ## Getting coefficients as a Series
    params = model.params[1:]
    params.sort_values(inplace=True)

    plt.figure(figsize=figsize) # Used if large number of params
    ax = params.plot(kind=kind)
    ax.axvline()
    ax.set_xlabel('Coefficient')
    ax.set_ylabel('Features')
    ax.set_title('Comparing Feature Coefficients')
    
    plt.tight_layout()
    
    return
```

#### ƒ: `plot_p_values`


```python
def plot_p_values(model, kind = 'barh', size = None, alpha = .05):
    ''' ---
    
    * Plots a figure to visualize parameter p-values exceeding stated alpha.
    
    ---
    
    * Args:
        * Model: linear regression model details to plot
        
    * Kwargs:
        * Kind (default 'barh'): allows different types of plots
        * Size (default None): allows for different sizes
    ---
    
    '''
    
    pv = model.pvalues[1:]
    pv_high = pv[pv > alpha]
    pv_low = pv[pv <= alpha]
    pv_high.sort_values(ascending=False, inplace=True)
    
    if len(pv_high) > 0:
        plt.figure(figsize=size) # Used if large number of params
        ax = pv_high.plot(kind=kind)
        ax = pv_low.plot(kind=kind)
        ax.axvline()
        plt.suptitle(f'P-Values')
        
    if len(pv_low) > 0:
        plt.figure(figsize=size) # Used if large number of params
        ax = pv_low.plot(kind=kind)
        ax.axvline()
        plt.suptitle(f'P-Values Below {alpha}')        
        
#     else:
#         print(f'There are no p-values above {alpha}.')
        
    plt.tight_layout()
    
    return
```

#### ƒ: `review_model`


```python
def review_model(model):
    '''---
    
    Combines earlier functions into one all-purpose function for reviewing
    model performance.
    
    ---
    
    Arg:
        * model: Specify model to review.
        
    ---'''
    
    diagnose_model(model)
    
    plot_param_coef(model)
    
    plot_p_values(model)
    
    return    
```

## Reading Data


```python
df= pd.read_csv('kc_house_data_train.csv', index_col=0)
```

# **Exploring Fresh Data**

## Basic Overviews

The goal for this section is to get a broad idea of the data before I start any cleaning or feature engineering.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2591820310</td>
      <td>20141006T000000</td>
      <td>365000.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>2070</td>
      <td>8893</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>2070</td>
      <td>0</td>
      <td>1986</td>
      <td>0</td>
      <td>98058</td>
      <td>47.4388</td>
      <td>-122.162</td>
      <td>2390</td>
      <td>7700</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7974200820</td>
      <td>20140821T000000</td>
      <td>865000.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>2900</td>
      <td>6730</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>1830</td>
      <td>1070</td>
      <td>1977</td>
      <td>0</td>
      <td>98115</td>
      <td>47.6784</td>
      <td>-122.285</td>
      <td>2370</td>
      <td>6283</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7701450110</td>
      <td>20140815T000000</td>
      <td>1038000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3770</td>
      <td>10893</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>11</td>
      <td>3770</td>
      <td>0</td>
      <td>1997</td>
      <td>0</td>
      <td>98006</td>
      <td>47.5646</td>
      <td>-122.129</td>
      <td>3710</td>
      <td>9685</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9522300010</td>
      <td>20150331T000000</td>
      <td>1490000.0</td>
      <td>3</td>
      <td>3.50</td>
      <td>4560</td>
      <td>14608</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>12</td>
      <td>4560</td>
      <td>0</td>
      <td>1990</td>
      <td>0</td>
      <td>98034</td>
      <td>47.6995</td>
      <td>-122.228</td>
      <td>4050</td>
      <td>14226</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9510861140</td>
      <td>20140714T000000</td>
      <td>711000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2550</td>
      <td>5376</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2550</td>
      <td>0</td>
      <td>2004</td>
      <td>0</td>
      <td>98052</td>
      <td>47.6647</td>
      <td>-122.083</td>
      <td>2250</td>
      <td>4050</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 17290 entries, 0 to 17289
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   id             17290 non-null  int64  
     1   date           17290 non-null  object 
     2   price          17290 non-null  float64
     3   bedrooms       17290 non-null  int64  
     4   bathrooms      17290 non-null  float64
     5   sqft_living    17290 non-null  int64  
     6   sqft_lot       17290 non-null  int64  
     7   floors         17290 non-null  float64
     8   waterfront     17290 non-null  int64  
     9   view           17290 non-null  int64  
     10  condition      17290 non-null  int64  
     11  grade          17290 non-null  int64  
     12  sqft_above     17290 non-null  int64  
     13  sqft_basement  17290 non-null  int64  
     14  yr_built       17290 non-null  int64  
     15  yr_renovated   17290 non-null  int64  
     16  zipcode        17290 non-null  int64  
     17  lat            17290 non-null  float64
     18  long           17290 non-null  float64
     19  sqft_living15  17290 non-null  int64  
     20  sqft_lot15     17290 non-null  int64  
    dtypes: float64(5), int64(15), object(1)
    memory usage: 2.9+ MB
    

**DF Columns to Convert**

I could convert the 'date' feature datatype from object to datetime or I may drop it instead.


```python
df.isnull().sum()
```




    id               0
    date             0
    price            0
    bedrooms         0
    bathrooms        0
    sqft_living      0
    sqft_lot         0
    floors           0
    waterfront       0
    view             0
    condition        0
    grade            0
    sqft_above       0
    sqft_basement    0
    yr_built         0
    yr_renovated     0
    zipcode          0
    lat              0
    long             0
    sqft_living15    0
    sqft_lot15       0
    dtype: int64




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.729000e+04</td>
      <td>1.729000e+04</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>1.729000e+04</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
      <td>17290.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.565502e+09</td>
      <td>5.407395e+05</td>
      <td>3.372470</td>
      <td>2.111943</td>
      <td>2081.464604</td>
      <td>1.524340e+04</td>
      <td>1.490312</td>
      <td>0.007981</td>
      <td>0.238519</td>
      <td>3.408502</td>
      <td>7.654425</td>
      <td>1789.306015</td>
      <td>292.158589</td>
      <td>1970.792019</td>
      <td>83.806304</td>
      <td>98078.193175</td>
      <td>47.560058</td>
      <td>-122.214258</td>
      <td>1987.986698</td>
      <td>12873.475824</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.874656e+09</td>
      <td>3.733190e+05</td>
      <td>0.939346</td>
      <td>0.770476</td>
      <td>920.018539</td>
      <td>4.230462e+04</td>
      <td>0.538909</td>
      <td>0.088985</td>
      <td>0.775229</td>
      <td>0.651296</td>
      <td>1.174718</td>
      <td>829.265107</td>
      <td>443.151874</td>
      <td>29.343516</td>
      <td>400.329376</td>
      <td>53.607949</td>
      <td>0.138412</td>
      <td>0.140857</td>
      <td>684.802635</td>
      <td>27227.437583</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.500000e+04</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>290.000000</td>
      <td>5.720000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>290.000000</td>
      <td>0.000000</td>
      <td>1900.000000</td>
      <td>0.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>659.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.114701e+09</td>
      <td>3.210000e+05</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1430.000000</td>
      <td>5.081250e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1200.000000</td>
      <td>0.000000</td>
      <td>1951.000000</td>
      <td>0.000000</td>
      <td>98033.000000</td>
      <td>47.471200</td>
      <td>-122.329000</td>
      <td>1490.000000</td>
      <td>5111.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.903650e+09</td>
      <td>4.500000e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1920.000000</td>
      <td>7.642000e+03</td>
      <td>1.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1560.000000</td>
      <td>0.000000</td>
      <td>1974.000000</td>
      <td>0.000000</td>
      <td>98065.000000</td>
      <td>47.571600</td>
      <td>-122.230000</td>
      <td>1840.000000</td>
      <td>7622.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.301150e+09</td>
      <td>6.450000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2550.000000</td>
      <td>1.072575e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>2214.500000</td>
      <td>560.000000</td>
      <td>1996.000000</td>
      <td>0.000000</td>
      <td>98118.000000</td>
      <td>47.677900</td>
      <td>-122.126000</td>
      <td>2360.000000</td>
      <td>10101.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>7.700000e+06</td>
      <td>33.000000</td>
      <td>8.000000</td>
      <td>13540.000000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>13.000000</td>
      <td>9410.000000</td>
      <td>4820.000000</td>
      <td>2015.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>6210.000000</td>
      <td>858132.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Overview Summary

The dataset contains 21 colums of data, most of which seem useful for evaluations and modeling, but "id" and "date" are not really usable in their current state.

I do not have any null values to handle. Additionally, the data types seem to be relatively accurate; the only exception being the dates being treated as an object vs. datetime.

I could convert the date for later calculations, engineering, or modeling. For now, I will leave it as-is.

## Exploring Features

Now I will be comparing each individual feature against "price" to determine if a feature is continuous or categorical.

I will check the regressions and histograms for each feature; if a regression plot shows a flatter regression line or if the histogram is not a normal distribution, I will treat that feature as a categorical feature.

As part of the function to create the plots, I include my function to ignore any outliers for the purpose of the visualizations. I will *not* change the data itself, though.

### Bedrooms


```python
show_cleaned_vis(df,"bedrooms", discrete=True, kde = False)
```

    There were 451 outliers.
    


    
![png](output_55_1.png)
    


***
**Observations**
>* There is a slight linear regression between the number of bedrooms and the sell price, indicated by the positive slope of the regression line.
>
>* The histogram slightly resembles a normal distribution, although the first bin is lower than I would expect for a normal distribution.

***
**TO-DO**
>* Use Bedrooms as a continuous variable.
***

### Bathrooms


```python
show_cleaned_vis(df,"bathrooms", discrete=True, kde=False)
```

    There were 209 outliers.
    


    
![png](output_58_1.png)
    


***
**Observations**
>* Clear linear relationship between bathrooms and price.
>* The histogram/normal distribution isn't the strongest, but I feel it is good enough.
***
**TO-DO**
>* Use as continuous variable.
*** 

### sqft_living


```python
show_cleaned_vis(df,"sqft_living")
```

    There were 454 outliers.
    


    
![png](output_61_1.png)
    


***
**Observations**

>* Clear linear relationship and normal distribution.

***

**TO-DO**

>* Use as continuous variable


### sqft_lot


```python
show_cleaned_vis(df,"sqft_lot")
```

    There were 1963 outliers.
    


    
![png](output_64_1.png)
    


***
**Observations**

>* Neither normally distributed nor having a linear relationship with price.
***

**TO-DO**

>* Treat as categorical variable.

* **

### floors


```python
show_cleaned_vis(df,"floors", kde=False)
```

    There were 0 outliers.
    


    
![png](output_67_1.png)
    


***
**Observations**

>* Linear relationship, but not normally distributed
>
>* Most homes had one or two bedrooms.

***
**TO-DO**

>* Treat as continuous variable due to regression
 
***

### waterfront


```python
feature_vis(df,"waterfront", discrete=True, kde=False)
```

    0    17152
    1      138
    Name: waterfront, dtype: int64
    


    
![png](output_70_1.png)
    


***
**Observations**

>* Graphs are misleading
>* Most properties are non-waterfront
>* Clear linear relationship, but not normally distributed

***
**TO-DO**

>* Treat as categorical - only two options (0 or 1).
 
***

### view


```python
feature_vis(df,"view", discrete=True, kde=False)
```

    0    15571
    1      264
    2      775
    3      410
    4      270
    Name: view, dtype: int64
    


    
![png](output_73_1.png)
    


***
**Observations**

>* Higher "view" rating, higher price
>* Most have '0' view
>* A few extreme outliers in pricing

***
**TO-DO**

>* Treat as categorical
 
***

### condition


```python
show_cleaned_vis(df,"condition", discrete=True, kde= False)
```

    There were 24 outliers.
    


    
![png](output_76_1.png)
    


***
**Observations**

>* Largest number of homes sold were in condition 3
>* Very few sold in 1s, 2s
>* Price outliers in 4.0 area, some slight outliers in 2 and 3
>* No linear relationship, not normal distribution.

***
**TO-DO**

>* Treat as categorical
 
***

### grade


```python
show_cleaned_vis(df,"grade", kde=False)
```

    There were 1512 outliers.
    


    
![png](output_79_1.png)
    


**Observations**

>* Clear linear trend - as grade increases, so does price
>
>* Largest range of grades is 6-9
---
**TO-DO**

>* Treat as continuous variable


### sqft_above


```python
show_cleaned_vis(df,"sqft_above")
```

    There were 498 outliers.
    


    
![png](output_82_1.png)
    


---

**Observations**

>* Outliers impacting accuracy of linear regression, but still seems relatively linear.
>
>* Distribution skewed left, but normal
---
**TO-DO**

>* Treat as continuous
---

### sqft_basement


```python
show_cleaned_vis(df,"sqft_basement")
```

    There were 396 outliers.
    


    
![png](output_85_1.png)
    



**Observations**

>* Lots of 0 sqft basements -  shows no basement at all on property
>
>* scattered outliers; poor regression due to 0s
---
**TO-DO**

>* Treat as continuous if used for modeling


### yr_built


```python
show_cleaned_vis(df,"yr_built")
```

    There were 0 outliers.
    


    
![png](output_88_1.png)
    


---
**Observations**

>* Non-linear trend from this regression plot
>
>* Seems like may houses built between 1940 - 1970, then major boom in early 2000s.
>
>* Some significant outliers with price, but not affecting regression
---

**TO-DO**

>* Compare to year sold - how old was the house at sale?
>
>* Treat as categorical due to regression plot
---

### yr_renovated


```python
feature_vis(df,'yr_renovated', kde=False)
```

    0       16564
    1934        1
    1940        2
    1944        1
    1945        3
            ...  
    2011        8
    2012        9
    2013       30
    2014       71
    2015       12
    Name: yr_renovated, Length: 69, dtype: int64
    


    
![png](output_91_1.png)
    



**Observations**

>* Number of houses not renovated significantly outweighs the number renovated
>
>* Graphs indicate that an engineered feature for whether or not the house was renovated would be worthwhile.
---
**TO-DO**

>* Create new categorical feature - renovated or not

### zipcode


```python
show_cleaned_vis(df,"zipcode", discrete=True, kde=False)
```

    There were 0 outliers.
    


    
![png](output_94_1.png)
    


---
**Observations**

>* No linear relationship; not normally distributed
---
**TO-DO**

>* Treat as categorical variable
>
>* Perform OneHotEncoding to create new features for modeling.
---

### lat


```python
show_cleaned_vis(df,"lat")
```

    There were 1 outliers.
    


    
![png](output_97_1.png)
    


**Observations**

>* No linear relationship and not normally distributed.

---
**TO-DO**

>* Treat as continuous if used for modeling.

### long


```python
show_cleaned_vis(df,"long")
```

    There were 206 outliers.
    


    
![png](output_100_1.png)
    


**Observations**

>* No linear relationship and not normally distributed.

---
**TO-DO**

>* Treat as continuous if used for modeling.

### sqft_living15


```python
show_cleaned_vis(df,"sqft_living15")
```

    There were 438 outliers.
    


    
![png](output_103_1.png)
    


---
**Observations**

>* Most properties sold with sqft between 1500/2000 
>
>* Price follows linear trend and is normally distributed, although skewed right.
---
**TO-DO**

>* Treat as continuous variable
---

### sqft_lot15


```python
show_cleaned_vis(df,"sqft_lot15")
```

    There were 1765 outliers.
    


    
![png](output_106_1.png)
    


---
**Observations**

>* No clear linear trend

---
**TO-DO**
>* Treat as categorical
---

# **Feature Engineering**

## Creating `'yrs_old_sold'`

I created this feature to differentiate between houses that were built recently versus older houses.

In order to determine this feature, I need to determine the year the house was sold first.

### Determine `'year_sold'`


```python
## Pull the year from the "date" column
df['year_sold'] = df['date'].map(lambda x: x[:4])

## Change it from a string to an integer datatype
df['year_sold'] =  df['year_sold'].map(lambda x: int(x))
```


```python
## Confirm that it was created
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sold'],
          dtype='object')




```python
## Review the values to ensure they are only the four-digit years
df['year_sold'].value_counts()
```




    2014    11703
    2015     5587
    Name: year_sold, dtype: int64



### Calculate `'y_old_sold'`


```python
## Calculating the age of the house at the time of sale
df['y_old_sold'] = df['year_sold'] - df['yr_built']
df['y_old_sold'].describe()

## Minimum age is -1 due to a house being sold before it was finished being built
```




    count    17290.000000
    mean        43.531116
    std         29.345345
    min         -1.000000
    25%         18.000000
    50%         40.000000
    75%         63.000000
    max        115.000000
    Name: y_old_sold, dtype: float64




```python
## Confirming creation
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sold',
           'y_old_sold'],
          dtype='object')




```python
## Confirming number of house entries
df['y_old_sold'].value_counts().sort_index()
```




    -1        7
     0      350
     1      232
     2      141
     3      123
           ... 
     111     41
     112     24
     113     23
     114     55
     115     22
    Name: y_old_sold, Length: 117, dtype: int64




```python
## Reviewing the descriptive stats for the data
df['y_old_sold'].describe()
```




    count    17290.000000
    mean        43.531116
    std         29.345345
    min         -1.000000
    25%         18.000000
    50%         40.000000
    75%         63.000000
    max        115.000000
    Name: y_old_sold, dtype: float64



## `'was_renovated'`

I created this feature to determine whether or not a house was renovated. If a house was renovated, what impact would that have on the price? 


```python
## Using the year that the home was renovated to deterine whether or not the home was renovated
reno_y_n = np.where(df['yr_renovated']>0, 1, 0 )
df = df.assign(was_renovated = reno_y_n)
```


```python
## Checking values
df['was_renovated'].value_counts()
```




    0    16564
    1      726
    Name: was_renovated, dtype: int64




```python
## Confirming column was created
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sold', 'y_old_sold',
           'was_renovated'],
          dtype='object')



## `"yrs_since_reno"`

If a house was renovated, how long ago was the renovation? Would more newly-renovated houses increase price?


```python
## Using new "was_renovated" feature to calculate how recently the house was renovated at the time of sale
reno = df[df['was_renovated'] == 1]

difference = reno['year_sold'] - reno['yr_renovated']

difference
```




    11       17
    33       31
    67       25
    186      29
    227      29
             ..
    17229    16
    17235    10
    17239    22
    17248    20
    17259    15
    Length: 726, dtype: int64




```python
## Confirming column creation within the dataframe
df.columns
```




    Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
           'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
           'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'year_sold', 'y_old_sold',
           'was_renovated'],
          dtype='object')




```python
## Ensuring there are no null values in the new feature and replacing any with zeroes
df = df.assign(yrs_since_reno = difference)

df['yrs_since_reno'].fillna(0, inplace=True)

df['yrs_since_reno'].isnull().sum()

df['yrs_since_reno'].describe()

## Minimum age is -1 due to a house being sold before it was finished being built
```




    count    17290.000000
    mean         0.772470
    std          4.888679
    min         -1.000000
    25%          0.000000
    50%          0.000000
    75%          0.000000
    max         80.000000
    Name: yrs_since_reno, dtype: float64



## "`has_bsmnt`"

I noticed that there were fewer houses with a value for "sqft_basement" during my data exploration. I am curious if the presence or absence of a basement would have any impact.


```python
## Determining whether or not a house has a basement based on the square footage
df['has_bsmnt'] = np.where(df['sqft_basement'] > 0, 1, 0)

# Reviewing the results
display(df['has_bsmnt'].describe(), df['has_bsmnt'].value_counts())
```


    count    17290.000000
    mean         0.393638
    std          0.488570
    min          0.000000
    25%          0.000000
    50%          0.000000
    75%          1.000000
    max          1.000000
    Name: has_bsmnt, dtype: float64



    0    10484
    1     6806
    Name: has_bsmnt, dtype: int64


# **Correlations**

## Determining Correlations with Price


```python
## Determining each feature's relationship with price

df_corr = df.drop(['price', 'id', 'lat','long'], axis=1).corrwith(df['price']).sort_values(ascending=False)
display(df_corr[0:5],df_corr[-6:-1])
```


    sqft_living      0.703540
    grade            0.666215
    sqft_above       0.607760
    sqft_living15    0.585567
    bathrooms        0.524335
    dtype: float64



    yrs_since_reno    0.058501
    yr_built          0.056450
    condition         0.036099
    year_sold        -0.002334
    zipcode          -0.048958
    dtype: float64


## Determining Multicollinearity


```python
## Get the correlation matrix for the data (without the target)
corr = df.drop('price',axis=1).corr()
corr.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>year_sold</th>
      <th>y_old_sold</th>
      <th>was_renovated</th>
      <th>yrs_since_reno</th>
      <th>has_bsmnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1.00</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>-0.13</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>-0.00</td>
      <td>0.01</td>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>-0.00</td>
      <td>0.02</td>
      <td>-0.01</td>
      <td>-0.14</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.51</td>
      <td>0.57</td>
      <td>0.03</td>
      <td>0.17</td>
      <td>-0.01</td>
      <td>0.08</td>
      <td>0.03</td>
      <td>0.35</td>
      <td>0.47</td>
      <td>0.30</td>
      <td>0.15</td>
      <td>0.01</td>
      <td>-0.15</td>
      <td>-0.00</td>
      <td>0.12</td>
      <td>0.39</td>
      <td>0.03</td>
      <td>-0.01</td>
      <td>-0.15</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.00</td>
      <td>0.51</td>
      <td>1.00</td>
      <td>0.76</td>
      <td>0.09</td>
      <td>0.50</td>
      <td>0.07</td>
      <td>0.19</td>
      <td>-0.12</td>
      <td>0.66</td>
      <td>0.69</td>
      <td>0.28</td>
      <td>0.51</td>
      <td>0.05</td>
      <td>-0.20</td>
      <td>0.02</td>
      <td>0.22</td>
      <td>0.57</td>
      <td>0.09</td>
      <td>-0.03</td>
      <td>-0.51</td>
      <td>0.05</td>
      <td>-0.01</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>sqft_living</th>
      <td>-0.01</td>
      <td>0.57</td>
      <td>0.76</td>
      <td>1.00</td>
      <td>0.17</td>
      <td>0.36</td>
      <td>0.11</td>
      <td>0.28</td>
      <td>-0.06</td>
      <td>0.76</td>
      <td>0.88</td>
      <td>0.44</td>
      <td>0.32</td>
      <td>0.05</td>
      <td>-0.20</td>
      <td>0.05</td>
      <td>0.23</td>
      <td>0.76</td>
      <td>0.18</td>
      <td>-0.03</td>
      <td>-0.32</td>
      <td>0.05</td>
      <td>0.01</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>sqft_lot</th>
      <td>-0.13</td>
      <td>0.03</td>
      <td>0.09</td>
      <td>0.17</td>
      <td>1.00</td>
      <td>-0.00</td>
      <td>0.02</td>
      <td>0.07</td>
      <td>-0.01</td>
      <td>0.11</td>
      <td>0.18</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>-0.13</td>
      <td>-0.08</td>
      <td>0.23</td>
      <td>0.14</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>-0.05</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>floors</th>
      <td>0.02</td>
      <td>0.17</td>
      <td>0.50</td>
      <td>0.36</td>
      <td>-0.00</td>
      <td>1.00</td>
      <td>0.03</td>
      <td>0.03</td>
      <td>-0.26</td>
      <td>0.46</td>
      <td>0.53</td>
      <td>-0.25</td>
      <td>0.48</td>
      <td>0.01</td>
      <td>-0.06</td>
      <td>0.05</td>
      <td>0.13</td>
      <td>0.28</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>-0.48</td>
      <td>0.01</td>
      <td>-0.00</td>
      <td>-0.26</td>
    </tr>
    <tr>
      <th>waterfront</th>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.07</td>
      <td>0.11</td>
      <td>0.02</td>
      <td>0.03</td>
      <td>1.00</td>
      <td>0.41</td>
      <td>0.01</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>-0.02</td>
      <td>0.09</td>
      <td>0.03</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>0.09</td>
      <td>0.03</td>
      <td>-0.00</td>
      <td>0.02</td>
      <td>0.09</td>
      <td>0.10</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>view</th>
      <td>0.01</td>
      <td>0.08</td>
      <td>0.19</td>
      <td>0.28</td>
      <td>0.07</td>
      <td>0.03</td>
      <td>0.41</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>0.25</td>
      <td>0.16</td>
      <td>0.28</td>
      <td>-0.05</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>-0.08</td>
      <td>0.28</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.09</td>
      <td>0.08</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>condition</th>
      <td>-0.01</td>
      <td>0.03</td>
      <td>-0.12</td>
      <td>-0.06</td>
      <td>-0.01</td>
      <td>-0.26</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>-0.14</td>
      <td>-0.16</td>
      <td>0.17</td>
      <td>-0.36</td>
      <td>-0.06</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>-0.11</td>
      <td>-0.09</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>0.36</td>
      <td>-0.06</td>
      <td>-0.00</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>0.00</td>
      <td>0.35</td>
      <td>0.66</td>
      <td>0.76</td>
      <td>0.11</td>
      <td>0.46</td>
      <td>0.09</td>
      <td>0.25</td>
      <td>-0.14</td>
      <td>1.00</td>
      <td>0.76</td>
      <td>0.17</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>-0.18</td>
      <td>0.12</td>
      <td>0.20</td>
      <td>0.72</td>
      <td>0.12</td>
      <td>-0.04</td>
      <td>-0.45</td>
      <td>0.01</td>
      <td>-0.03</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>sqft_above</th>
      <td>-0.01</td>
      <td>0.47</td>
      <td>0.69</td>
      <td>0.88</td>
      <td>0.18</td>
      <td>0.53</td>
      <td>0.08</td>
      <td>0.16</td>
      <td>-0.16</td>
      <td>0.76</td>
      <td>1.00</td>
      <td>-0.05</td>
      <td>0.42</td>
      <td>0.02</td>
      <td>-0.26</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>0.73</td>
      <td>0.19</td>
      <td>-0.03</td>
      <td>-0.42</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>-0.21</td>
    </tr>
    <tr>
      <th>sqft_basement</th>
      <td>-0.00</td>
      <td>0.30</td>
      <td>0.28</td>
      <td>0.44</td>
      <td>0.01</td>
      <td>-0.25</td>
      <td>0.08</td>
      <td>0.28</td>
      <td>0.17</td>
      <td>0.17</td>
      <td>-0.05</td>
      <td>1.00</td>
      <td>-0.13</td>
      <td>0.07</td>
      <td>0.08</td>
      <td>0.11</td>
      <td>-0.15</td>
      <td>0.20</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.13</td>
      <td>0.07</td>
      <td>0.02</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>yr_built</th>
      <td>0.01</td>
      <td>0.15</td>
      <td>0.51</td>
      <td>0.32</td>
      <td>0.05</td>
      <td>0.48</td>
      <td>-0.02</td>
      <td>-0.05</td>
      <td>-0.36</td>
      <td>0.45</td>
      <td>0.42</td>
      <td>-0.13</td>
      <td>1.00</td>
      <td>-0.22</td>
      <td>-0.35</td>
      <td>-0.15</td>
      <td>0.41</td>
      <td>0.33</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>-1.00</td>
      <td>-0.22</td>
      <td>-0.20</td>
      <td>-0.17</td>
    </tr>
    <tr>
      <th>yr_renovated</th>
      <td>-0.02</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>-0.06</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.07</td>
      <td>-0.22</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>-0.06</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.02</td>
      <td>0.22</td>
      <td>1.00</td>
      <td>0.75</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>zipcode</th>
      <td>-0.01</td>
      <td>-0.15</td>
      <td>-0.20</td>
      <td>-0.20</td>
      <td>-0.13</td>
      <td>-0.06</td>
      <td>0.03</td>
      <td>0.09</td>
      <td>0.01</td>
      <td>-0.18</td>
      <td>-0.26</td>
      <td>0.08</td>
      <td>-0.35</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>0.26</td>
      <td>-0.56</td>
      <td>-0.27</td>
      <td>-0.15</td>
      <td>0.00</td>
      <td>0.35</td>
      <td>0.06</td>
      <td>0.05</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>lat</th>
      <td>-0.00</td>
      <td>-0.00</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>-0.08</td>
      <td>0.05</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.01</td>
      <td>0.12</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>-0.15</td>
      <td>0.03</td>
      <td>0.26</td>
      <td>1.00</td>
      <td>-0.13</td>
      <td>0.05</td>
      <td>-0.09</td>
      <td>-0.03</td>
      <td>0.14</td>
      <td>0.03</td>
      <td>0.02</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>long</th>
      <td>0.02</td>
      <td>0.12</td>
      <td>0.22</td>
      <td>0.23</td>
      <td>0.23</td>
      <td>0.13</td>
      <td>-0.04</td>
      <td>-0.08</td>
      <td>-0.11</td>
      <td>0.20</td>
      <td>0.34</td>
      <td>-0.15</td>
      <td>0.41</td>
      <td>-0.06</td>
      <td>-0.56</td>
      <td>-0.13</td>
      <td>1.00</td>
      <td>0.33</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>-0.41</td>
      <td>-0.06</td>
      <td>-0.05</td>
      <td>-0.24</td>
    </tr>
    <tr>
      <th>sqft_living15</th>
      <td>-0.01</td>
      <td>0.39</td>
      <td>0.57</td>
      <td>0.76</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.09</td>
      <td>0.28</td>
      <td>-0.09</td>
      <td>0.72</td>
      <td>0.73</td>
      <td>0.20</td>
      <td>0.33</td>
      <td>-0.01</td>
      <td>-0.27</td>
      <td>0.05</td>
      <td>0.33</td>
      <td>1.00</td>
      <td>0.18</td>
      <td>-0.03</td>
      <td>-0.33</td>
      <td>-0.01</td>
      <td>-0.02</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>sqft_lot15</th>
      <td>-0.14</td>
      <td>0.03</td>
      <td>0.09</td>
      <td>0.18</td>
      <td>0.70</td>
      <td>-0.01</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>-0.01</td>
      <td>0.12</td>
      <td>0.19</td>
      <td>0.01</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>-0.15</td>
      <td>-0.09</td>
      <td>0.25</td>
      <td>0.18</td>
      <td>1.00</td>
      <td>-0.01</td>
      <td>-0.08</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>-0.04</td>
    </tr>
    <tr>
      <th>year_sold</th>
      <td>0.01</td>
      <td>-0.01</td>
      <td>-0.03</td>
      <td>-0.03</td>
      <td>0.00</td>
      <td>-0.02</td>
      <td>-0.00</td>
      <td>0.00</td>
      <td>-0.04</td>
      <td>-0.04</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.02</td>
      <td>0.00</td>
      <td>-0.03</td>
      <td>0.00</td>
      <td>-0.03</td>
      <td>-0.01</td>
      <td>1.00</td>
      <td>0.01</td>
      <td>-0.02</td>
      <td>-0.01</td>
      <td>-0.00</td>
    </tr>
    <tr>
      <th>y_old_sold</th>
      <td>-0.01</td>
      <td>-0.15</td>
      <td>-0.51</td>
      <td>-0.32</td>
      <td>-0.05</td>
      <td>-0.48</td>
      <td>0.02</td>
      <td>0.05</td>
      <td>0.36</td>
      <td>-0.45</td>
      <td>-0.42</td>
      <td>0.13</td>
      <td>-1.00</td>
      <td>0.22</td>
      <td>0.35</td>
      <td>0.14</td>
      <td>-0.41</td>
      <td>-0.33</td>
      <td>-0.08</td>
      <td>0.01</td>
      <td>1.00</td>
      <td>0.22</td>
      <td>0.20</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>was_renovated</th>
      <td>-0.02</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>-0.06</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.07</td>
      <td>-0.22</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>0.03</td>
      <td>-0.06</td>
      <td>-0.01</td>
      <td>0.00</td>
      <td>-0.02</td>
      <td>0.22</td>
      <td>1.00</td>
      <td>0.75</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>yrs_since_reno</th>
      <td>-0.01</td>
      <td>-0.01</td>
      <td>-0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>-0.00</td>
      <td>0.10</td>
      <td>0.08</td>
      <td>-0.00</td>
      <td>-0.03</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>-0.20</td>
      <td>0.75</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>-0.05</td>
      <td>-0.02</td>
      <td>0.01</td>
      <td>-0.01</td>
      <td>0.20</td>
      <td>0.75</td>
      <td>1.00</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>has_bsmnt</th>
      <td>0.00</td>
      <td>0.16</td>
      <td>0.16</td>
      <td>0.20</td>
      <td>-0.04</td>
      <td>-0.26</td>
      <td>0.04</td>
      <td>0.19</td>
      <td>0.13</td>
      <td>0.05</td>
      <td>-0.21</td>
      <td>0.82</td>
      <td>-0.17</td>
      <td>0.05</td>
      <td>0.17</td>
      <td>0.13</td>
      <td>-0.24</td>
      <td>0.04</td>
      <td>-0.04</td>
      <td>-0.00</td>
      <td>0.17</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
corr_val(df.drop('price',axis=1), figsize=(20,20));
```


    
![png](output_138_0.png)
    



```python
## Correlation results ignoring duplicate values 
## Except for "was_renovated", which shows a perfect relationship to itself

df_corr_results = df.corr().unstack().sort_values(ascending=False).drop_duplicates()
```


```python
# Show strongest postive and negative correlations
display(df_corr_results[1:11], df_corr_results[-11:-1])
```


    yr_renovated    was_renovated     0.999968
    sqft_living     sqft_above        0.876696
    has_bsmnt       sqft_basement     0.818269
    sqft_living     grade             0.762929
    grade           sqft_above        0.758247
    sqft_living     bathrooms         0.755270
    sqft_living15   sqft_living       0.755066
    was_renovated   yrs_since_reno    0.754774
    yrs_since_reno  yr_renovated      0.749501
    sqft_living15   sqft_above        0.731077
    dtype: float64



    sqft_living  y_old_sold      -0.319912
    y_old_sold   sqft_living15   -0.332739
    zipcode      yr_built        -0.348366
    condition    yr_built        -0.358921
    long         y_old_sold      -0.407716
    sqft_above   y_old_sold      -0.424795
    y_old_sold   grade           -0.448558
    floors       y_old_sold      -0.484020
    y_old_sold   bathrooms       -0.508209
    zipcode      long            -0.563027
    dtype: float64



```python
# Dropping columns to address multicollinearity (over .8)

df.drop(['yr_renovated','sqft_basement','sqft_above'], axis=1, inplace=True)
```


```python
# Rerunning model

corr_val(df.drop('price',axis=1), figsize=(20,20));
```


    
![png](output_142_0.png)
    



```python
# Correlation results ignoring (most) duplicate values
df_corr_results = df.corr().unstack().sort_values(ascending=False).drop_duplicates()

# Show strongest postive and negative correlations
display(df_corr_results[1:11], df_corr_results[-11:-1])
```


    grade           sqft_living      0.762929
    sqft_living     bathrooms        0.755270
                    sqft_living15    0.755066
    yrs_since_reno  was_renovated    0.754774
    sqft_living15   grade            0.715745
    sqft_living     price            0.703540
    sqft_lot        sqft_lot15       0.701875
    price           grade            0.666215
    grade           bathrooms        0.664191
    price           sqft_living15    0.585567
    dtype: float64



    zipcode      sqft_living15   -0.273903
    sqft_living  y_old_sold      -0.319912
    y_old_sold   sqft_living15   -0.332739
    yr_built     zipcode         -0.348366
    condition    yr_built        -0.358921
    long         y_old_sold      -0.407716
    grade        y_old_sold      -0.448558
    y_old_sold   floors          -0.484020
                 bathrooms       -0.508209
    long         zipcode         -0.563027
    dtype: float64


### Interpretation of Correlations

***
**Top 10 Positive Relationships**
>* Nothing too surprising as most of the matches are intuitively related.
>  * E.g. "yr_renovated" and "was_renovated" have a nearly-perfect positive correlation as "was_renovated" is determined by "yr_renovated" in our feature engineering.
>
>
>* Two interesting relationships would be:
>  * The living space (ft^2) and grade
>    * Indicates that a larger house has a higher grade
> * The living space (ft^2) of the 15 nearest houses sold
>   * Indicates a larger area above ground (ft^2)
>    * Perhaps larger houses are more likely to be nearby each other?
***
**Top 10 Negative Relationships**
>* Older houses may have fewer bathrooms
>* Older houses may have fewer floors
>* Older houses have a lower grade
***

# **Statistical Testing**

## One-Way ANOVA

I would assume that the condition of a house would have an impact on the price of a house (who would pay more for a run-down house that would need a lot of repair work?).

I wanted to confirm that the different condition levels would be significant for modeling.

### *Testing `'condition'`*

* H0: The feature "condition" does not have an effect on price.

* Ha: The feature "condition" does  have an effect on price.



```python
## Defining variables for the prices of each value of conditions

condition_1 = df.loc[df['condition'] == 1, 'price']
condition_2 = df.loc[df['condition'] == 2, 'price']
condition_3 = df.loc[df['condition'] == 3, 'price']
condition_4 = df.loc[df['condition'] == 4, 'price']
condition_5 = df.loc[df['condition'] == 5, 'price']
```


```python
## Running ANOVA test to determine significance

## Define alpha
alpha = .05

## Run test
result = stats.f_oneway(condition_1, condition_2, condition_3, condition_4, condition_5)
f_stat, p_value = result

# Evaluate signficance
if p_value < .05:
    print(f'The condition of a home is statistically significant with a p-value of {p_value}.')
    
else:
     print(f'The condition of a home is not statistically significant with a p-value of {p_value}.')
```

    The condition of a home is statistically significant with a p-value of 6.813536869427384e-24.
    


```python
# Show visual of conclusion

sns.barplot(data=df, x= 'condition', y = 'price', ci=68)
plt.suptitle("Conditon's Affect on Price", size = (20))
plt.xlabel('Condition')
plt.ylabel('Price ($)');
```


    
![png](output_153_0.png)
    


#### Interpretation

> The t-test shows that the condition of a house is statistically significant due to the p-value below our alpha of .05.
>
> This means that the quality of a house will have a statistically significant impact on the sell price.

## Two-Sample T-Tests

Again, common sense tells us that waterfront homes would sell for higher prices, but I wanted to confirm that assumption.

### *Testing  `'waterfront'`*

---
**Hypotheses**

>* **H0:** The feature "waterfront" does not have an effect on price.
>
>* **HA:** The feature "waterfront" does  have an effect on price.
>
>* **Alpha** = .05
---


```python
# Set variables to represent whether or not a property is listed as 'waterfront.'

wf_yes = df.loc[df['waterfront'] == 1, 'price']
wf_no = df.loc[df['waterfront'] == 0, 'price']
```


```python
ttest_review(wf_yes, wf_no)
```

    The feature "waterfront" is statistically significant with a p-value of 2.465038317654299e-299.
    




    2.465038317654299e-299




```python
# Show visual of conclusion

sns.barplot(data=df, x= 'waterfront', y = 'price', ci=68);
```


    
![png](output_162_0.png)
    


#### Interpretation

> The t-test shows that waterfront is statistically significant due to the p-value below our alpha of .05.
>
> This means that having a house on the waterfront will have a significant impact on the sell price.

### *Testing `"was_renovated"`*

**Hypotheses**

---
>* **H0:** There is not a statistically significant difference in price in homes with a basement than those without.
>
>* **HA:** There is a statistically significant difference in price in homes with a basement than those without.
>
>* **Alpha** = 0.05
---


```python
reno_y = df.loc[df['was_renovated'] == 1, 'price']
reno_n = df.loc[df['was_renovated'] == 0, 'price']

```


```python
ttest_review(reno_y, reno_n)
```

    The feature "waterfront" is statistically significant with a p-value of 4.007705820364685e-54.
    




    4.007705820364685e-54




```python
# Show visual of conclusion

sns.barplot(data=df, x= 'was_renovated', y = 'price', ci=68);
```


    
![png](output_169_0.png)
    


#### Interpretation

> The t-test shows that having a basement is statistically significant due to the p-value below our alpha of .05.
>
> This means that having a house with a basement will have a significant impact on the sell price.

# **Inferential Modeling**

## Baseline Model with Raw Data

Creating a baseline model using the raw dataset.


```python
## Reviewing feature names and their datatypes
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 17290 entries, 0 to 17289
    Data columns (total 23 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   id              17290 non-null  int64  
     1   date            17290 non-null  object 
     2   price           17290 non-null  float64
     3   bedrooms        17290 non-null  int64  
     4   bathrooms       17290 non-null  float64
     5   sqft_living     17290 non-null  int64  
     6   sqft_lot        17290 non-null  int64  
     7   floors          17290 non-null  float64
     8   waterfront      17290 non-null  int64  
     9   view            17290 non-null  int64  
     10  condition       17290 non-null  int64  
     11  grade           17290 non-null  int64  
     12  yr_built        17290 non-null  int64  
     13  zipcode         17290 non-null  int64  
     14  lat             17290 non-null  float64
     15  long            17290 non-null  float64
     16  sqft_living15   17290 non-null  int64  
     17  sqft_lot15      17290 non-null  int64  
     18  year_sold       17290 non-null  int64  
     19  y_old_sold      17290 non-null  int64  
     20  was_renovated   17290 non-null  int32  
     21  yrs_since_reno  17290 non-null  float64
     22  has_bsmnt       17290 non-null  int32  
    dtypes: float64(6), int32(2), int64(14), object(1)
    memory usage: 3.7+ MB
    


```python
## Establishing the features for inferential modeling

categorical_features = ['was_renovated','has_bsmnt', 'waterfront']

continuous_features = ['y_old_sold','yrs_since_reno', 'bedrooms', 'bathrooms',
                       'condition','grade', 'floors']

model_baseline = create_model(df,continuous_features,categorical_features);
```

    price~+y_old_sold+yrs_since_reno+bedrooms+bathrooms+condition+grade+floors+C(was_renovated)+C(has_bsmnt)+C(waterfront)
    


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.599</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.598</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   2576.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Jun 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:10:07</td>     <th>  Log-Likelihood:    </th> <td>-2.3848e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 17290</td>      <th>  AIC:               </th>  <td>4.770e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 17279</td>      <th>  BIC:               </th>  <td>4.771e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>-1.531e+06</td> <td> 1.81e+04</td> <td>  -84.504</td> <td> 0.000</td> <td>-1.57e+06</td> <td> -1.5e+06</td>
</tr>
<tr>
  <th>C(was_renovated)[T.1]</th> <td> 5.842e+04</td> <td> 1.41e+04</td> <td>    4.130</td> <td> 0.000</td> <td> 3.07e+04</td> <td> 8.62e+04</td>
</tr>
<tr>
  <th>C(has_bsmnt)[T.1]</th>     <td> 3.158e+04</td> <td> 4156.712</td> <td>    7.597</td> <td> 0.000</td> <td> 2.34e+04</td> <td> 3.97e+04</td>
</tr>
<tr>
  <th>C(waterfront)[T.1]</th>    <td> 8.042e+05</td> <td> 2.05e+04</td> <td>   39.232</td> <td> 0.000</td> <td> 7.64e+05</td> <td> 8.44e+05</td>
</tr>
<tr>
  <th>y_old_sold</th>            <td> 4325.6941</td> <td>   85.480</td> <td>   50.605</td> <td> 0.000</td> <td> 4158.144</td> <td> 4493.244</td>
</tr>
<tr>
  <th>yrs_since_reno</th>        <td>-2511.5735</td> <td>  564.503</td> <td>   -4.449</td> <td> 0.000</td> <td>-3618.057</td> <td>-1405.090</td>
</tr>
<tr>
  <th>bedrooms</th>              <td>-2081.5430</td> <td> 2266.058</td> <td>   -0.919</td> <td> 0.358</td> <td>-6523.247</td> <td> 2360.161</td>
</tr>
<tr>
  <th>bathrooms</th>             <td>  1.25e+05</td> <td> 3930.644</td> <td>   31.793</td> <td> 0.000</td> <td> 1.17e+05</td> <td> 1.33e+05</td>
</tr>
<tr>
  <th>condition</th>             <td> 2.008e+04</td> <td> 3045.263</td> <td>    6.594</td> <td> 0.000</td> <td> 1.41e+04</td> <td>  2.6e+04</td>
</tr>
<tr>
  <th>grade</th>                 <td>  1.98e+05</td> <td> 2117.430</td> <td>   93.491</td> <td> 0.000</td> <td> 1.94e+05</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>floors</th>                <td> 1.586e+04</td> <td> 4400.252</td> <td>    3.604</td> <td> 0.000</td> <td> 7232.686</td> <td> 2.45e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>15571.873</td> <th>  Durbin-Watson:     </th>  <td>   1.991</td>  
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>1844101.619</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 3.873</td>   <th>  Prob(JB):          </th>  <td>    0.00</td>  
</tr>
<tr>
  <th>Kurtosis:</th>       <td>52.998</td>   <th>  Cond. No.          </th>  <td>    610.</td>  
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_176_2.png)
    


---
**Interpretation:**

>* R^2: ~.6, less than target of .75
>
>* Residual plots show heteroscedasticity
>
>* Q-Q Plot shows non-normal residuals
>
>* Changes: remove outliers and retest
---

## Model without Outliers in 'Price'

What would the model look like if we ignored the outliers?

### Removing Outliers from 'Price' Data


```python
## Remove outliers from price 
idx_outs = find_outliers_z(df['price'])
df_clean = df[~idx_outs].copy()
df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>year_sold</th>
      <th>y_old_sold</th>
      <th>was_renovated</th>
      <th>yrs_since_reno</th>
      <th>has_bsmnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2591820310</td>
      <td>20141006T000000</td>
      <td>365000.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>2070</td>
      <td>8893</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>1986</td>
      <td>98058</td>
      <td>47.4388</td>
      <td>-122.162</td>
      <td>2390</td>
      <td>7700</td>
      <td>2014</td>
      <td>28</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7974200820</td>
      <td>20140821T000000</td>
      <td>865000.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>2900</td>
      <td>6730</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>1977</td>
      <td>98115</td>
      <td>47.6784</td>
      <td>-122.285</td>
      <td>2370</td>
      <td>6283</td>
      <td>2014</td>
      <td>37</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7701450110</td>
      <td>20140815T000000</td>
      <td>1038000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3770</td>
      <td>10893</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>11</td>
      <td>1997</td>
      <td>98006</td>
      <td>47.5646</td>
      <td>-122.129</td>
      <td>3710</td>
      <td>9685</td>
      <td>2014</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9522300010</td>
      <td>20150331T000000</td>
      <td>1490000.0</td>
      <td>3</td>
      <td>3.50</td>
      <td>4560</td>
      <td>14608</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>12</td>
      <td>1990</td>
      <td>98034</td>
      <td>47.6995</td>
      <td>-122.228</td>
      <td>4050</td>
      <td>14226</td>
      <td>2015</td>
      <td>25</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9510861140</td>
      <td>20140714T000000</td>
      <td>711000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2550</td>
      <td>5376</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2004</td>
      <td>98052</td>
      <td>47.6647</td>
      <td>-122.083</td>
      <td>2250</td>
      <td>4050</td>
      <td>2014</td>
      <td>10</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17285</th>
      <td>627300195</td>
      <td>20150303T000000</td>
      <td>750000.0</td>
      <td>5</td>
      <td>2.50</td>
      <td>3240</td>
      <td>9960</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1958</td>
      <td>98008</td>
      <td>47.5858</td>
      <td>-122.112</td>
      <td>2730</td>
      <td>10400</td>
      <td>2015</td>
      <td>57</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17286</th>
      <td>8819900270</td>
      <td>20140520T000000</td>
      <td>440000.0</td>
      <td>2</td>
      <td>1.75</td>
      <td>1300</td>
      <td>4000</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1948</td>
      <td>98105</td>
      <td>47.6687</td>
      <td>-122.288</td>
      <td>1350</td>
      <td>4013</td>
      <td>2014</td>
      <td>66</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17287</th>
      <td>3816300095</td>
      <td>20140514T000000</td>
      <td>310000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1050</td>
      <td>9876</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1953</td>
      <td>98028</td>
      <td>47.7635</td>
      <td>-122.262</td>
      <td>1760</td>
      <td>9403</td>
      <td>2014</td>
      <td>61</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17288</th>
      <td>122069107</td>
      <td>20141204T000000</td>
      <td>427500.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1900</td>
      <td>43186</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>1971</td>
      <td>98038</td>
      <td>47.4199</td>
      <td>-121.990</td>
      <td>2080</td>
      <td>108028</td>
      <td>2014</td>
      <td>43</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17289</th>
      <td>6703100135</td>
      <td>20150116T000000</td>
      <td>348000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1330</td>
      <td>6768</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>1952</td>
      <td>98155</td>
      <td>47.7366</td>
      <td>-122.319</td>
      <td>1320</td>
      <td>6910</td>
      <td>2015</td>
      <td>63</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16977 rows × 23 columns</p>
</div>




```python
## Remove outliers from bedrooms
idx_outs = find_outliers_z(df_clean['bedrooms'])
df_clean = df_clean[~idx_outs].copy()

df_clean
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>year_sold</th>
      <th>y_old_sold</th>
      <th>was_renovated</th>
      <th>yrs_since_reno</th>
      <th>has_bsmnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2591820310</td>
      <td>20141006T000000</td>
      <td>365000.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>2070</td>
      <td>8893</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>8</td>
      <td>1986</td>
      <td>98058</td>
      <td>47.4388</td>
      <td>-122.162</td>
      <td>2390</td>
      <td>7700</td>
      <td>2014</td>
      <td>28</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7974200820</td>
      <td>20140821T000000</td>
      <td>865000.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>2900</td>
      <td>6730</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>1977</td>
      <td>98115</td>
      <td>47.6784</td>
      <td>-122.285</td>
      <td>2370</td>
      <td>6283</td>
      <td>2014</td>
      <td>37</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7701450110</td>
      <td>20140815T000000</td>
      <td>1038000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3770</td>
      <td>10893</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>11</td>
      <td>1997</td>
      <td>98006</td>
      <td>47.5646</td>
      <td>-122.129</td>
      <td>3710</td>
      <td>9685</td>
      <td>2014</td>
      <td>17</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9522300010</td>
      <td>20150331T000000</td>
      <td>1490000.0</td>
      <td>3</td>
      <td>3.50</td>
      <td>4560</td>
      <td>14608</td>
      <td>2.0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>12</td>
      <td>1990</td>
      <td>98034</td>
      <td>47.6995</td>
      <td>-122.228</td>
      <td>4050</td>
      <td>14226</td>
      <td>2015</td>
      <td>25</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9510861140</td>
      <td>20140714T000000</td>
      <td>711000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>2550</td>
      <td>5376</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>9</td>
      <td>2004</td>
      <td>98052</td>
      <td>47.6647</td>
      <td>-122.083</td>
      <td>2250</td>
      <td>4050</td>
      <td>2014</td>
      <td>10</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17285</th>
      <td>627300195</td>
      <td>20150303T000000</td>
      <td>750000.0</td>
      <td>5</td>
      <td>2.50</td>
      <td>3240</td>
      <td>9960</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1958</td>
      <td>98008</td>
      <td>47.5858</td>
      <td>-122.112</td>
      <td>2730</td>
      <td>10400</td>
      <td>2015</td>
      <td>57</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17286</th>
      <td>8819900270</td>
      <td>20140520T000000</td>
      <td>440000.0</td>
      <td>2</td>
      <td>1.75</td>
      <td>1300</td>
      <td>4000</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1948</td>
      <td>98105</td>
      <td>47.6687</td>
      <td>-122.288</td>
      <td>1350</td>
      <td>4013</td>
      <td>2014</td>
      <td>66</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17287</th>
      <td>3816300095</td>
      <td>20140514T000000</td>
      <td>310000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1050</td>
      <td>9876</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>7</td>
      <td>1953</td>
      <td>98028</td>
      <td>47.7635</td>
      <td>-122.262</td>
      <td>1760</td>
      <td>9403</td>
      <td>2014</td>
      <td>61</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17288</th>
      <td>122069107</td>
      <td>20141204T000000</td>
      <td>427500.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1900</td>
      <td>43186</td>
      <td>1.5</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>1971</td>
      <td>98038</td>
      <td>47.4199</td>
      <td>-121.990</td>
      <td>2080</td>
      <td>108028</td>
      <td>2014</td>
      <td>43</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17289</th>
      <td>6703100135</td>
      <td>20150116T000000</td>
      <td>348000.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>1330</td>
      <td>6768</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>7</td>
      <td>1952</td>
      <td>98155</td>
      <td>47.7366</td>
      <td>-122.319</td>
      <td>1320</td>
      <td>6910</td>
      <td>2015</td>
      <td>63</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>16923 rows × 23 columns</p>
</div>




```python
df_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>grade</th>
      <th>yr_built</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>year_sold</th>
      <th>y_old_sold</th>
      <th>was_renovated</th>
      <th>yrs_since_reno</th>
      <th>has_bsmnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.692300e+04</td>
      <td>1.692300e+04</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.00000</td>
      <td>1.692300e+04</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
      <td>16923.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.580302e+09</td>
      <td>5.068460e+05</td>
      <td>3.346038</td>
      <td>2.081856</td>
      <td>2030.81209</td>
      <td>1.502735e+04</td>
      <td>1.482302</td>
      <td>0.004668</td>
      <td>0.210010</td>
      <td>3.407375</td>
      <td>7.602494</td>
      <td>1970.751994</td>
      <td>98078.442416</td>
      <td>47.558885</td>
      <td>-122.213853</td>
      <td>1961.790995</td>
      <td>12762.476629</td>
      <td>2014.323170</td>
      <td>43.571175</td>
      <td>0.040064</td>
      <td>0.746381</td>
      <td>0.388170</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.875948e+09</td>
      <td>2.594154e+05</td>
      <td>0.871877</td>
      <td>0.731297</td>
      <td>831.18918</td>
      <td>4.172312e+04</td>
      <td>0.536899</td>
      <td>0.068167</td>
      <td>0.717421</td>
      <td>0.649637</td>
      <td>1.107905</td>
      <td>29.237036</td>
      <td>53.534877</td>
      <td>0.139363</td>
      <td>0.141553</td>
      <td>654.638580</td>
      <td>27137.875033</td>
      <td>0.467701</td>
      <td>29.239539</td>
      <td>0.196115</td>
      <td>4.855077</td>
      <td>0.487348</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000102e+06</td>
      <td>7.500000e+04</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>370.00000</td>
      <td>5.720000e+02</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1900.000000</td>
      <td>98001.000000</td>
      <td>47.155900</td>
      <td>-122.519000</td>
      <td>399.000000</td>
      <td>659.000000</td>
      <td>2014.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.123049e+09</td>
      <td>3.200000e+05</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>1420.00000</td>
      <td>5.040000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1951.000000</td>
      <td>98033.000000</td>
      <td>47.466900</td>
      <td>-122.330000</td>
      <td>1480.000000</td>
      <td>5100.000000</td>
      <td>2014.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.904940e+09</td>
      <td>4.464500e+05</td>
      <td>3.000000</td>
      <td>2.250000</td>
      <td>1900.00000</td>
      <td>7.587000e+03</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>7.000000</td>
      <td>1974.000000</td>
      <td>98065.000000</td>
      <td>47.569600</td>
      <td>-122.230000</td>
      <td>1830.000000</td>
      <td>7597.000000</td>
      <td>2014.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.309100e+09</td>
      <td>6.290000e+05</td>
      <td>4.000000</td>
      <td>2.500000</td>
      <td>2500.00000</td>
      <td>1.051150e+04</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>1996.000000</td>
      <td>98118.000000</td>
      <td>47.678750</td>
      <td>-122.124000</td>
      <td>2330.000000</td>
      <td>10004.500000</td>
      <td>2015.000000</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.900000e+09</td>
      <td>1.655000e+06</td>
      <td>6.000000</td>
      <td>6.000000</td>
      <td>7350.00000</td>
      <td>1.651359e+06</td>
      <td>3.500000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>2015.000000</td>
      <td>98199.000000</td>
      <td>47.777600</td>
      <td>-121.315000</td>
      <td>5790.000000</td>
      <td>858132.000000</td>
      <td>2015.000000</td>
      <td>115.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Redefining updated features and re-running model

categorical_features = ['was_renovated','has_bsmnt', 'waterfront']

continuous_features = ['y_old_sold','yrs_since_reno', 'bedrooms', 'bathrooms',
                       'condition','grade', 'floors']

model_clean = create_model(df_clean,cont=continuous_features,cat=categorical_features);
```

    price~+y_old_sold+yrs_since_reno+bedrooms+bathrooms+condition+grade+floors+C(was_renovated)+C(has_bsmnt)+C(waterfront)
    


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.589</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.589</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   2422.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Jun 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:10:10</td>     <th>  Log-Likelihood:    </th> <td>-2.2746e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16923</td>      <th>  AIC:               </th>  <td>4.549e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16912</td>      <th>  BIC:               </th>  <td>4.550e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    10</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>-1.162e+06</td> <td> 1.35e+04</td> <td>  -86.344</td> <td> 0.000</td> <td>-1.19e+06</td> <td>-1.14e+06</td>
</tr>
<tr>
  <th>C(was_renovated)[T.1]</th> <td>  5.14e+04</td> <td> 1.02e+04</td> <td>    5.021</td> <td> 0.000</td> <td> 3.13e+04</td> <td> 7.15e+04</td>
</tr>
<tr>
  <th>C(has_bsmnt)[T.1]</th>     <td> 4.068e+04</td> <td> 2964.749</td> <td>   13.721</td> <td> 0.000</td> <td> 3.49e+04</td> <td> 4.65e+04</td>
</tr>
<tr>
  <th>C(waterfront)[T.1]</th>    <td>  3.24e+05</td> <td> 1.89e+04</td> <td>   17.130</td> <td> 0.000</td> <td> 2.87e+05</td> <td> 3.61e+05</td>
</tr>
<tr>
  <th>y_old_sold</th>            <td> 3284.2377</td> <td>   62.183</td> <td>   52.816</td> <td> 0.000</td> <td> 3162.353</td> <td> 3406.122</td>
</tr>
<tr>
  <th>yrs_since_reno</th>        <td>-1913.9577</td> <td>  402.413</td> <td>   -4.756</td> <td> 0.000</td> <td>-2702.729</td> <td>-1125.187</td>
</tr>
<tr>
  <th>bedrooms</th>              <td> 5271.9827</td> <td> 1739.192</td> <td>    3.031</td> <td> 0.002</td> <td> 1862.984</td> <td> 8680.981</td>
</tr>
<tr>
  <th>bathrooms</th>             <td> 7.191e+04</td> <td> 2921.530</td> <td>   24.613</td> <td> 0.000</td> <td> 6.62e+04</td> <td> 7.76e+04</td>
</tr>
<tr>
  <th>condition</th>             <td> 2.121e+04</td> <td> 2167.706</td> <td>    9.783</td> <td> 0.000</td> <td>  1.7e+04</td> <td> 2.55e+04</td>
</tr>
<tr>
  <th>grade</th>                 <td> 1.593e+05</td> <td> 1562.805</td> <td>  101.927</td> <td> 0.000</td> <td> 1.56e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>floors</th>                <td> 3.852e+04</td> <td> 3166.597</td> <td>   12.164</td> <td> 0.000</td> <td> 3.23e+04</td> <td> 4.47e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2820.763</td> <th>  Durbin-Watson:     </th> <td>   1.995</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7575.714</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.908</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 5.728</td>  <th>  Cond. No.          </th> <td>    784.</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



    
![png](output_184_2.png)
    


Interpretation:

* R^2: ~.59, less than target of .75 (performed slightly worse)

* Residual plots show somewhat increased homoscedasticity

* Q-Q Plot shows more normal residuals (vs. earlier plot)

* Changes: add zipcode 

## Model (Clean) - w/ Zip Codes


```python
## Adding zip codes as a categorical variable

categorical_features = ['was_renovated','has_bsmnt', 'waterfront', 'zipcode']

continuous_features = ['y_old_sold','yrs_since_reno', 'bedrooms', 'bathrooms',
                       'condition','grade', 'floors']

model_w_zip = create_model(df_clean,continuous_features, categorical_features)
```

    price~+y_old_sold+yrs_since_reno+bedrooms+bathrooms+condition+grade+floors+C(was_renovated)+C(has_bsmnt)+C(waterfront)+C(zipcode)
    


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.772</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.771</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   723.0</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Jun 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:10:12</td>     <th>  Log-Likelihood:    </th> <td>-2.2246e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16923</td>      <th>  AIC:               </th>  <td>4.451e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16843</td>      <th>  BIC:               </th>  <td>4.457e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    79</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>-9.082e+05</td> <td> 1.29e+04</td> <td>  -70.617</td> <td> 0.000</td> <td>-9.33e+05</td> <td>-8.83e+05</td>
</tr>
<tr>
  <th>C(was_renovated)[T.1]</th> <td> 5.205e+04</td> <td> 7669.920</td> <td>    6.786</td> <td> 0.000</td> <td>  3.7e+04</td> <td> 6.71e+04</td>
</tr>
<tr>
  <th>C(has_bsmnt)[T.1]</th>     <td> 4194.0254</td> <td> 2410.118</td> <td>    1.740</td> <td> 0.082</td> <td> -530.059</td> <td> 8918.109</td>
</tr>
<tr>
  <th>C(waterfront)[T.1]</th>    <td> 4.146e+05</td> <td> 1.46e+04</td> <td>   28.492</td> <td> 0.000</td> <td> 3.86e+05</td> <td> 4.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>   <td> 7700.3206</td> <td> 1.24e+04</td> <td>    0.622</td> <td> 0.534</td> <td>-1.66e+04</td> <td>  3.2e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>   <td>-2.942e+04</td> <td> 1.11e+04</td> <td>   -2.651</td> <td> 0.008</td> <td>-5.12e+04</td> <td>-7666.780</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>   <td> 6.095e+05</td> <td> 1.21e+04</td> <td>   50.472</td> <td> 0.000</td> <td> 5.86e+05</td> <td> 6.33e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>   <td> 2.997e+05</td> <td>  1.3e+04</td> <td>   22.968</td> <td> 0.000</td> <td> 2.74e+05</td> <td> 3.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>   <td> 2.806e+05</td> <td> 9970.638</td> <td>   28.143</td> <td> 0.000</td> <td> 2.61e+05</td> <td>    3e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>   <td> 2.074e+05</td> <td> 1.35e+04</td> <td>   15.307</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>   <td> 2.307e+05</td> <td> 1.12e+04</td> <td>   20.617</td> <td> 0.000</td> <td> 2.09e+05</td> <td> 2.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>   <td> 1.156e+05</td> <td> 1.62e+04</td> <td>    7.135</td> <td> 0.000</td> <td> 8.39e+04</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>   <td> 1.349e+05</td> <td> 1.22e+04</td> <td>   11.047</td> <td> 0.000</td> <td> 1.11e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>   <td> 1.667e+05</td> <td> 1.44e+04</td> <td>   11.542</td> <td> 0.000</td> <td> 1.38e+05</td> <td> 1.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>   <td> 1.188e+05</td> <td> 1.27e+04</td> <td>    9.368</td> <td> 0.000</td> <td>  9.4e+04</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>   <td> 3.376e+04</td> <td> 1.17e+04</td> <td>    2.892</td> <td> 0.004</td> <td> 1.09e+04</td> <td> 5.66e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>   <td>-3.732e+04</td> <td> 9591.428</td> <td>   -3.891</td> <td> 0.000</td> <td>-5.61e+04</td> <td>-1.85e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>   <td> 2.266e+05</td> <td> 1.73e+04</td> <td>   13.095</td> <td> 0.000</td> <td> 1.93e+05</td> <td> 2.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>   <td> 1.868e+05</td> <td> 1.01e+04</td> <td>   18.439</td> <td> 0.000</td> <td> 1.67e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>   <td> 1.311e+05</td> <td> 1.12e+04</td> <td>   11.735</td> <td> 0.000</td> <td> 1.09e+05</td> <td> 1.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>   <td> 1.883e+05</td> <td> 1.09e+04</td> <td>   17.344</td> <td> 0.000</td> <td> 1.67e+05</td> <td>  2.1e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>   <td>-7414.8887</td> <td> 1.14e+04</td> <td>   -0.648</td> <td> 0.517</td> <td>-2.98e+04</td> <td>  1.5e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>   <td>-5749.4528</td> <td> 1.12e+04</td> <td>   -0.511</td> <td> 0.609</td> <td>-2.78e+04</td> <td> 1.63e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>   <td>-1.688e+04</td> <td> 1.43e+04</td> <td>   -1.181</td> <td> 0.238</td> <td>-4.49e+04</td> <td> 1.11e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>   <td> 3.474e+05</td> <td> 9992.388</td> <td>   34.769</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>   <td> 1.682e+05</td> <td> 9518.772</td> <td>   17.668</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 1.87e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>   <td>  5.18e+04</td> <td> 9424.092</td> <td>    5.497</td> <td> 0.000</td> <td> 3.33e+04</td> <td> 7.03e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>   <td> 8.197e+05</td> <td> 3.19e+04</td> <td>   25.677</td> <td> 0.000</td> <td> 7.57e+05</td> <td> 8.82e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>   <td> 4.667e+05</td> <td> 1.17e+04</td> <td>   39.877</td> <td> 0.000</td> <td> 4.44e+05</td> <td>  4.9e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>   <td> 1.474e+04</td> <td> 9540.112</td> <td>    1.545</td> <td> 0.122</td> <td>-3957.891</td> <td> 3.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>   <td> 1.216e+05</td> <td> 1.19e+04</td> <td>   10.209</td> <td> 0.000</td> <td> 9.83e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>   <td> 2.325e+05</td> <td> 9361.152</td> <td>   24.832</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 2.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>   <td> 2.761e+05</td> <td> 1.02e+04</td> <td>   27.076</td> <td> 0.000</td> <td> 2.56e+05</td> <td> 2.96e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>   <td> 3.111e+04</td> <td> 1.13e+04</td> <td>    2.747</td> <td> 0.006</td> <td> 8912.468</td> <td> 5.33e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>   <td> 1.057e+05</td> <td>    1e+04</td> <td>   10.521</td> <td> 0.000</td> <td>  8.6e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>   <td> 2.717e+04</td> <td> 9715.247</td> <td>    2.796</td> <td> 0.005</td> <td> 8122.425</td> <td> 4.62e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>   <td> 1.253e+05</td> <td> 9806.382</td> <td>   12.782</td> <td> 0.000</td> <td> 1.06e+05</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>   <td> 1.717e+05</td> <td> 1.09e+04</td> <td>   15.819</td> <td> 0.000</td> <td>  1.5e+05</td> <td> 1.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>   <td>  1.21e+05</td> <td> 1.48e+04</td> <td>    8.173</td> <td> 0.000</td> <td>  9.2e+04</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>   <td> 1.799e+05</td> <td> 1.13e+04</td> <td>   15.988</td> <td> 0.000</td> <td> 1.58e+05</td> <td> 2.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>   <td> 2.038e+05</td> <td> 9989.473</td> <td>   20.397</td> <td> 0.000</td> <td> 1.84e+05</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>   <td> 2.444e+05</td> <td> 1.06e+04</td> <td>   23.079</td> <td> 0.000</td> <td> 2.24e+05</td> <td> 2.65e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>   <td> 1.818e+05</td> <td> 1.24e+04</td> <td>   14.708</td> <td> 0.000</td> <td> 1.58e+05</td> <td> 2.06e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>   <td>-1.356e+04</td> <td> 1.04e+04</td> <td>   -1.301</td> <td> 0.193</td> <td> -3.4e+04</td> <td> 6869.682</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>   <td> 3.381e+05</td> <td> 1.67e+04</td> <td>   20.287</td> <td> 0.000</td> <td> 3.05e+05</td> <td> 3.71e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>   <td> 2.578e+05</td> <td> 9666.639</td> <td>   26.673</td> <td> 0.000</td> <td> 2.39e+05</td> <td> 2.77e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>   <td> 3.731e+05</td> <td> 1.26e+04</td> <td>   29.548</td> <td> 0.000</td> <td> 3.48e+05</td> <td> 3.98e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>   <td> 8.367e+04</td> <td> 1.07e+04</td> <td>    7.839</td> <td> 0.000</td> <td> 6.27e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>   <td> 2.401e+05</td> <td> 1.15e+04</td> <td>   20.845</td> <td> 0.000</td> <td> 2.18e+05</td> <td> 2.63e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>   <td> 7.647e+04</td> <td> 1.26e+04</td> <td>    6.077</td> <td> 0.000</td> <td> 5.18e+04</td> <td> 1.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>   <td> 3.639e+05</td> <td> 1.61e+04</td> <td>   22.662</td> <td> 0.000</td> <td> 3.32e+05</td> <td> 3.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>   <td> 4.291e+05</td> <td> 1.22e+04</td> <td>   35.188</td> <td> 0.000</td> <td> 4.05e+05</td> <td> 4.53e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>   <td> 2.738e+05</td> <td> 9561.468</td> <td>   28.632</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 2.93e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>   <td> 2.591e+05</td> <td> 1.08e+04</td> <td>   24.009</td> <td> 0.000</td> <td> 2.38e+05</td> <td>  2.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>   <td> 2.563e+05</td> <td> 9689.714</td> <td>   26.454</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 2.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>   <td> 1.385e+05</td> <td> 9747.827</td> <td>   14.207</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>   <td> 3.655e+05</td> <td> 1.31e+04</td> <td>   27.878</td> <td> 0.000</td> <td>  3.4e+05</td> <td> 3.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>   <td> 2.284e+05</td> <td> 1.13e+04</td> <td>   20.223</td> <td> 0.000</td> <td> 2.06e+05</td> <td> 2.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>   <td> 1.715e+05</td> <td> 1.02e+04</td> <td>   16.811</td> <td> 0.000</td> <td> 1.51e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>   <td>  1.54e+05</td> <td> 1.06e+04</td> <td>   14.512</td> <td> 0.000</td> <td> 1.33e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>   <td> 1.161e+05</td> <td> 9703.189</td> <td>   11.966</td> <td> 0.000</td> <td> 9.71e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>   <td> 2.234e+05</td> <td> 1.14e+04</td> <td>   19.560</td> <td> 0.000</td> <td> 2.01e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>   <td> 2.165e+05</td> <td> 1.09e+04</td> <td>   19.869</td> <td> 0.000</td> <td> 1.95e+05</td> <td> 2.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>   <td> 1.136e+05</td> <td> 1.09e+04</td> <td>   10.448</td> <td> 0.000</td> <td> 9.23e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>   <td> 2.775e+04</td> <td> 2.03e+04</td> <td>    1.365</td> <td> 0.172</td> <td>-1.21e+04</td> <td> 6.76e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>   <td> 1.177e+05</td> <td> 9977.705</td> <td>   11.800</td> <td> 0.000</td> <td> 9.82e+04</td> <td> 1.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>   <td> 9.723e+04</td> <td> 1.15e+04</td> <td>    8.485</td> <td> 0.000</td> <td> 7.48e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>   <td>  5.43e+04</td> <td> 1.13e+04</td> <td>    4.805</td> <td> 0.000</td> <td> 3.21e+04</td> <td> 7.64e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>   <td> 2.171e+05</td> <td> 1.14e+04</td> <td>   19.125</td> <td> 0.000</td> <td> 1.95e+05</td> <td> 2.39e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>   <td> 4.547e+04</td> <td> 1.13e+04</td> <td>    4.022</td> <td> 0.000</td> <td> 2.33e+04</td> <td> 6.76e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>   <td> 2.522e+04</td> <td>  1.4e+04</td> <td>    1.802</td> <td> 0.072</td> <td>-2217.644</td> <td> 5.27e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>   <td> 2.883e+04</td> <td> 1.12e+04</td> <td>    2.580</td> <td> 0.010</td> <td> 6923.167</td> <td> 5.07e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>   <td> 3.282e+05</td> <td> 1.09e+04</td> <td>   30.061</td> <td> 0.000</td> <td> 3.07e+05</td> <td>  3.5e+05</td>
</tr>
<tr>
  <th>y_old_sold</th>            <td> 1627.0931</td> <td>   55.079</td> <td>   29.541</td> <td> 0.000</td> <td> 1519.132</td> <td> 1735.054</td>
</tr>
<tr>
  <th>yrs_since_reno</th>        <td>-1367.4953</td> <td>  301.397</td> <td>   -4.537</td> <td> 0.000</td> <td>-1958.265</td> <td> -776.726</td>
</tr>
<tr>
  <th>bedrooms</th>              <td> 1.827e+04</td> <td> 1339.987</td> <td>   13.635</td> <td> 0.000</td> <td> 1.56e+04</td> <td> 2.09e+04</td>
</tr>
<tr>
  <th>bathrooms</th>             <td> 7.179e+04</td> <td> 2196.916</td> <td>   32.675</td> <td> 0.000</td> <td> 6.75e+04</td> <td> 7.61e+04</td>
</tr>
<tr>
  <th>condition</th>             <td> 2.066e+04</td> <td> 1670.976</td> <td>   12.366</td> <td> 0.000</td> <td> 1.74e+04</td> <td> 2.39e+04</td>
</tr>
<tr>
  <th>grade</th>                 <td> 1.186e+05</td> <td> 1269.882</td> <td>   93.416</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.21e+05</td>
</tr>
<tr>
  <th>floors</th>                <td>-4885.4475</td> <td> 2672.150</td> <td>   -1.828</td> <td> 0.068</td> <td>-1.01e+04</td> <td>  352.246</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3595.690</td> <th>  Durbin-Watson:     </th> <td>   2.016</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>13713.791</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.024</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.906</td>  <th>  Cond. No.          </th> <td>3.49e+03</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.49e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



    
![png](output_187_2.png)
    


Interpretation:

>* Model is significant with an R-squared value over the threshold of .75 (result: .77)
>
>* Residual plots show somewhat homoscedasticity
>
>* Q-Q Plot shows more normal residuals (vs. earlier plot)
>
>* Changes: remove features with p-values higher than .05 (statistically insignificant values)


```python
## Re-running model after removing 'has_bsmnt' and 'floors'" due to high p-value
categorical_features = ['was_renovated','waterfront', 'zipcode']

continuous_features = ['y_old_sold','yrs_since_reno', 'bedrooms', 'bathrooms',
                       'condition','grade']

model_clean = create_model(data=df_clean, cont=continuous_features, cat=categorical_features)
```

    price~+y_old_sold+yrs_since_reno+bedrooms+bathrooms+condition+grade+C(was_renovated)+C(waterfront)+C(zipcode)
    


<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.772</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.771</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   741.2</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 04 Jun 2021</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>12:10:14</td>     <th>  Log-Likelihood:    </th> <td>-2.2246e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 16923</td>      <th>  AIC:               </th>  <td>4.451e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 16845</td>      <th>  BIC:               </th>  <td>4.457e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    77</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>             <td>-9.145e+05</td> <td> 1.26e+04</td> <td>  -72.433</td> <td> 0.000</td> <td>-9.39e+05</td> <td> -8.9e+05</td>
</tr>
<tr>
  <th>C(was_renovated)[T.1]</th> <td> 5.184e+04</td> <td> 7671.429</td> <td>    6.757</td> <td> 0.000</td> <td> 3.68e+04</td> <td> 6.69e+04</td>
</tr>
<tr>
  <th>C(waterfront)[T.1]</th>    <td> 4.146e+05</td> <td> 1.45e+04</td> <td>   28.510</td> <td> 0.000</td> <td> 3.86e+05</td> <td> 4.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98002]</th>   <td> 6498.2936</td> <td> 1.24e+04</td> <td>    0.525</td> <td> 0.599</td> <td>-1.78e+04</td> <td> 3.08e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98003]</th>   <td> -2.86e+04</td> <td> 1.11e+04</td> <td>   -2.577</td> <td> 0.010</td> <td>-5.03e+04</td> <td>-6842.006</td>
</tr>
<tr>
  <th>C(zipcode)[T.98004]</th>   <td> 6.101e+05</td> <td> 1.21e+04</td> <td>   50.523</td> <td> 0.000</td> <td> 5.86e+05</td> <td> 6.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98005]</th>   <td> 3.014e+05</td> <td>  1.3e+04</td> <td>   23.107</td> <td> 0.000</td> <td> 2.76e+05</td> <td> 3.27e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98006]</th>   <td> 2.818e+05</td> <td> 9965.649</td> <td>   28.272</td> <td> 0.000</td> <td> 2.62e+05</td> <td> 3.01e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98007]</th>   <td> 2.073e+05</td> <td> 1.36e+04</td> <td>   15.295</td> <td> 0.000</td> <td> 1.81e+05</td> <td> 2.34e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98008]</th>   <td>  2.32e+05</td> <td> 1.12e+04</td> <td>   20.739</td> <td> 0.000</td> <td>  2.1e+05</td> <td> 2.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98010]</th>   <td> 1.148e+05</td> <td> 1.62e+04</td> <td>    7.083</td> <td> 0.000</td> <td>  8.3e+04</td> <td> 1.47e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98011]</th>   <td>  1.35e+05</td> <td> 1.22e+04</td> <td>   11.057</td> <td> 0.000</td> <td> 1.11e+05</td> <td> 1.59e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98014]</th>   <td> 1.658e+05</td> <td> 1.44e+04</td> <td>   11.478</td> <td> 0.000</td> <td> 1.37e+05</td> <td> 1.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98019]</th>   <td> 1.179e+05</td> <td> 1.27e+04</td> <td>    9.296</td> <td> 0.000</td> <td> 9.31e+04</td> <td> 1.43e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98022]</th>   <td> 3.311e+04</td> <td> 1.17e+04</td> <td>    2.836</td> <td> 0.005</td> <td> 1.02e+04</td> <td>  5.6e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98023]</th>   <td>-3.635e+04</td> <td> 9589.388</td> <td>   -3.791</td> <td> 0.000</td> <td>-5.51e+04</td> <td>-1.76e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98024]</th>   <td> 2.261e+05</td> <td> 1.73e+04</td> <td>   13.063</td> <td> 0.000</td> <td> 1.92e+05</td> <td>  2.6e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98027]</th>   <td>  1.87e+05</td> <td> 1.01e+04</td> <td>   18.496</td> <td> 0.000</td> <td> 1.67e+05</td> <td> 2.07e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98028]</th>   <td> 1.316e+05</td> <td> 1.12e+04</td> <td>   11.782</td> <td> 0.000</td> <td>  1.1e+05</td> <td> 1.54e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98029]</th>   <td> 1.866e+05</td> <td> 1.08e+04</td> <td>   17.204</td> <td> 0.000</td> <td> 1.65e+05</td> <td> 2.08e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98030]</th>   <td>-7704.2765</td> <td> 1.14e+04</td> <td>   -0.673</td> <td> 0.501</td> <td>-3.01e+04</td> <td> 1.47e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98031]</th>   <td>-5857.5752</td> <td> 1.12e+04</td> <td>   -0.521</td> <td> 0.602</td> <td>-2.79e+04</td> <td> 1.62e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98032]</th>   <td>-1.643e+04</td> <td> 1.43e+04</td> <td>   -1.149</td> <td> 0.250</td> <td>-4.44e+04</td> <td> 1.16e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98033]</th>   <td> 3.475e+05</td> <td> 9994.606</td> <td>   34.769</td> <td> 0.000</td> <td> 3.28e+05</td> <td> 3.67e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98034]</th>   <td> 1.693e+05</td> <td> 9515.521</td> <td>   17.787</td> <td> 0.000</td> <td> 1.51e+05</td> <td> 1.88e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98038]</th>   <td> 5.042e+04</td> <td> 9417.311</td> <td>    5.353</td> <td> 0.000</td> <td>  3.2e+04</td> <td> 6.89e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98039]</th>   <td> 8.205e+05</td> <td> 3.19e+04</td> <td>   25.699</td> <td> 0.000</td> <td> 7.58e+05</td> <td> 8.83e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98040]</th>   <td> 4.675e+05</td> <td> 1.17e+04</td> <td>   39.945</td> <td> 0.000</td> <td> 4.45e+05</td> <td>  4.9e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98042]</th>   <td> 1.417e+04</td> <td> 9540.489</td> <td>    1.486</td> <td> 0.137</td> <td>-4526.835</td> <td> 3.29e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98045]</th>   <td> 1.213e+05</td> <td> 1.19e+04</td> <td>   10.179</td> <td> 0.000</td> <td> 9.79e+04</td> <td> 1.45e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98052]</th>   <td> 2.326e+05</td> <td> 9362.476</td> <td>   24.843</td> <td> 0.000</td> <td> 2.14e+05</td> <td> 2.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98053]</th>   <td> 2.754e+05</td> <td> 1.02e+04</td> <td>   27.030</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 2.95e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98055]</th>   <td> 3.074e+04</td> <td> 1.13e+04</td> <td>    2.714</td> <td> 0.007</td> <td> 8542.101</td> <td> 5.29e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98056]</th>   <td> 1.052e+05</td> <td>    1e+04</td> <td>   10.472</td> <td> 0.000</td> <td> 8.55e+04</td> <td> 1.25e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98058]</th>   <td> 2.744e+04</td> <td> 9717.489</td> <td>    2.824</td> <td> 0.005</td> <td> 8392.419</td> <td> 4.65e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98059]</th>   <td> 1.243e+05</td> <td> 9803.305</td> <td>   12.683</td> <td> 0.000</td> <td> 1.05e+05</td> <td> 1.44e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98065]</th>   <td> 1.699e+05</td> <td> 1.08e+04</td> <td>   15.668</td> <td> 0.000</td> <td> 1.49e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98070]</th>   <td> 1.204e+05</td> <td> 1.48e+04</td> <td>    8.129</td> <td> 0.000</td> <td> 9.14e+04</td> <td> 1.49e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98072]</th>   <td> 1.806e+05</td> <td> 1.13e+04</td> <td>   16.055</td> <td> 0.000</td> <td> 1.59e+05</td> <td> 2.03e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98074]</th>   <td> 2.031e+05</td> <td> 9990.478</td> <td>   20.334</td> <td> 0.000</td> <td> 1.84e+05</td> <td> 2.23e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98075]</th>   <td> 2.433e+05</td> <td> 1.06e+04</td> <td>   22.985</td> <td> 0.000</td> <td> 2.23e+05</td> <td> 2.64e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98077]</th>   <td> 1.812e+05</td> <td> 1.24e+04</td> <td>   14.660</td> <td> 0.000</td> <td> 1.57e+05</td> <td> 2.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98092]</th>   <td>-1.425e+04</td> <td> 1.04e+04</td> <td>   -1.367</td> <td> 0.172</td> <td>-3.47e+04</td> <td> 6187.981</td>
</tr>
<tr>
  <th>C(zipcode)[T.98102]</th>   <td> 3.358e+05</td> <td> 1.65e+04</td> <td>   20.326</td> <td> 0.000</td> <td> 3.03e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98103]</th>   <td> 2.553e+05</td> <td> 9495.899</td> <td>   26.888</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 2.74e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98105]</th>   <td> 3.724e+05</td> <td> 1.25e+04</td> <td>   29.693</td> <td> 0.000</td> <td> 3.48e+05</td> <td> 3.97e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98106]</th>   <td>   8.4e+04</td> <td> 1.06e+04</td> <td>    7.894</td> <td> 0.000</td> <td> 6.31e+04</td> <td> 1.05e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98107]</th>   <td> 2.383e+05</td> <td> 1.14e+04</td> <td>   20.927</td> <td> 0.000</td> <td> 2.16e+05</td> <td> 2.61e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98108]</th>   <td> 7.693e+04</td> <td> 1.26e+04</td> <td>    6.129</td> <td> 0.000</td> <td> 5.23e+04</td> <td> 1.02e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98109]</th>   <td> 3.625e+05</td> <td> 1.59e+04</td> <td>   22.730</td> <td> 0.000</td> <td> 3.31e+05</td> <td> 3.94e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98112]</th>   <td> 4.277e+05</td> <td> 1.21e+04</td> <td>   35.458</td> <td> 0.000</td> <td> 4.04e+05</td> <td> 4.51e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98115]</th>   <td> 2.738e+05</td> <td> 9478.528</td> <td>   28.883</td> <td> 0.000</td> <td> 2.55e+05</td> <td> 2.92e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98116]</th>   <td> 2.591e+05</td> <td> 1.07e+04</td> <td>   24.197</td> <td> 0.000</td> <td> 2.38e+05</td> <td>  2.8e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98117]</th>   <td> 2.558e+05</td> <td> 9603.986</td> <td>   26.632</td> <td> 0.000</td> <td> 2.37e+05</td> <td> 2.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98118]</th>   <td> 1.385e+05</td> <td> 9698.476</td> <td>   14.281</td> <td> 0.000</td> <td> 1.19e+05</td> <td> 1.58e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98119]</th>   <td>  3.64e+05</td> <td>  1.3e+04</td> <td>   28.064</td> <td> 0.000</td> <td> 3.39e+05</td> <td> 3.89e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98122]</th>   <td> 2.265e+05</td> <td> 1.12e+04</td> <td>   20.305</td> <td> 0.000</td> <td> 2.05e+05</td> <td> 2.48e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98125]</th>   <td> 1.715e+05</td> <td> 1.02e+04</td> <td>   16.856</td> <td> 0.000</td> <td> 1.52e+05</td> <td> 1.91e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98126]</th>   <td>  1.54e+05</td> <td> 1.06e+04</td> <td>   14.582</td> <td> 0.000</td> <td> 1.33e+05</td> <td> 1.75e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98133]</th>   <td> 1.158e+05</td> <td> 9678.547</td> <td>   11.962</td> <td> 0.000</td> <td> 9.68e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98136]</th>   <td> 2.234e+05</td> <td> 1.14e+04</td> <td>   19.678</td> <td> 0.000</td> <td> 2.01e+05</td> <td> 2.46e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98144]</th>   <td>  2.16e+05</td> <td> 1.08e+04</td> <td>   20.053</td> <td> 0.000</td> <td> 1.95e+05</td> <td> 2.37e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98146]</th>   <td> 1.138e+05</td> <td> 1.09e+04</td> <td>   10.470</td> <td> 0.000</td> <td> 9.25e+04</td> <td> 1.35e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98148]</th>   <td> 2.791e+04</td> <td> 2.03e+04</td> <td>    1.373</td> <td> 0.170</td> <td>-1.19e+04</td> <td> 6.78e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98155]</th>   <td> 1.181e+05</td> <td> 9975.645</td> <td>   11.838</td> <td> 0.000</td> <td> 9.85e+04</td> <td> 1.38e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98166]</th>   <td> 9.794e+04</td> <td> 1.15e+04</td> <td>    8.553</td> <td> 0.000</td> <td> 7.55e+04</td> <td>  1.2e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98168]</th>   <td> 5.454e+04</td> <td> 1.13e+04</td> <td>    4.830</td> <td> 0.000</td> <td> 3.24e+04</td> <td> 7.67e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98177]</th>   <td> 2.185e+05</td> <td> 1.13e+04</td> <td>   19.270</td> <td> 0.000</td> <td> 1.96e+05</td> <td> 2.41e+05</td>
</tr>
<tr>
  <th>C(zipcode)[T.98178]</th>   <td> 4.614e+04</td> <td> 1.13e+04</td> <td>    4.085</td> <td> 0.000</td> <td>  2.4e+04</td> <td> 6.83e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98188]</th>   <td> 2.592e+04</td> <td>  1.4e+04</td> <td>    1.851</td> <td> 0.064</td> <td>-1525.345</td> <td> 5.34e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98198]</th>   <td> 2.934e+04</td> <td> 1.12e+04</td> <td>    2.626</td> <td> 0.009</td> <td> 7436.794</td> <td> 5.12e+04</td>
</tr>
<tr>
  <th>C(zipcode)[T.98199]</th>   <td> 3.294e+05</td> <td> 1.08e+04</td> <td>   30.411</td> <td> 0.000</td> <td> 3.08e+05</td> <td> 3.51e+05</td>
</tr>
<tr>
  <th>y_old_sold</th>            <td> 1672.8602</td> <td>   52.032</td> <td>   32.151</td> <td> 0.000</td> <td> 1570.873</td> <td> 1774.848</td>
</tr>
<tr>
  <th>yrs_since_reno</th>        <td>-1408.1210</td> <td>  301.100</td> <td>   -4.677</td> <td> 0.000</td> <td>-1998.308</td> <td> -817.934</td>
</tr>
<tr>
  <th>bedrooms</th>              <td>  1.85e+04</td> <td> 1334.327</td> <td>   13.861</td> <td> 0.000</td> <td> 1.59e+04</td> <td> 2.11e+04</td>
</tr>
<tr>
  <th>bathrooms</th>             <td> 7.185e+04</td> <td> 2064.623</td> <td>   34.800</td> <td> 0.000</td> <td> 6.78e+04</td> <td> 7.59e+04</td>
</tr>
<tr>
  <th>condition</th>             <td> 2.118e+04</td> <td> 1663.980</td> <td>   12.730</td> <td> 0.000</td> <td> 1.79e+04</td> <td> 2.44e+04</td>
</tr>
<tr>
  <th>grade</th>                 <td> 1.181e+05</td> <td> 1258.259</td> <td>   93.886</td> <td> 0.000</td> <td> 1.16e+05</td> <td> 1.21e+05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>3599.647</td> <th>  Durbin-Watson:     </th> <td>   2.017</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>13732.009</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.025</td>  <th>  Prob(JB):          </th> <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 6.908</td>  <th>  Cond. No.          </th> <td>3.48e+03</td> 
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.48e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



    
![png](output_189_2.png)
    



```python
coeff_clean.sort_values(ascending=False)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-77-c04d813cdc11> in <module>
    ----> 1 coeff_clean.sort_values(ascending=False)
    

    NameError: name 'coeff_clean' is not defined



```python
# Graphing mode paramnerters to show values
coeff_clean = model_clean.params.sort_values(ascending=False)
coeff_clean.plot(kind='barh');
```

# Recommendations based on Inferential Modeling

If a homeowner is interested in renovating their house, I would recommend the following improvements based on the high coefficients from my model.

* Add bathrooms
* Also consider adding bedrooms
* Use high-quality materials in renovations

While it is outside of the homeowner's ability to change, the zip code for a house's location also has a large impact on the price. A prospective buyer should keep the zip code of the area in mind to help determine the prices of the homes there.


```python
## Visualizing the impact of number of bathrooms on price
sns.regplot(data=df_clean, x="bathrooms", y='price')
plt.suptitle('Impact of Number of Bathrooms on Price')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Price ($)');
```


    
![png](output_194_0.png)
    



```python
## Visualizing impact of number of bedrooms on price
sns.regplot(data=df_clean, x="bedrooms", y='price')
plt.suptitle('Impact of Number of Bedrooms on Price')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Price ($)');
```


    
![png](output_195_0.png)
    



```python
## Visualizing impact of the grade of construction materials on price
sns.regplot(data=df_clean, x="grade", y='price')
plt.suptitle('Impact of Grade of Materials on Price')
plt.xlabel('Grade of Materials')
plt.ylabel('Price ($)');
```


    
![png](output_196_0.png)
    



```python
## Representing the diversity of prices based on zip codes

fg = sns.catplot(data=df_clean, x="zipcode", y='price', aspect=2.75, height=5)
fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=45, ha='right')
plt.suptitle('Impact of Zip Code on Price')
plt.xlabel('Zip Code')
plt.ylabel('Price ($)');
```


    
![png](output_197_0.png)
    


# **Model Testing for Predictions**

Now that I have a working model, I want to go ahead and test it to see how well it works. I will split the data in a 75/25 split into two groups; train a model on the 75%; and then compare it against the remaining 25%. 

**My goal is to have a Root Mean Squared Error (RMSE) less than 200,000.**

## ID Features, Target

First, I want to establish the features I want to use in my testing and my target variable (in this case, price).


```python
## Save the features that we used previously - I will use these features 
## to fit my model.

X = df_clean[['was_renovated','waterfront', 'zipcode', 'y_old_sold',
            'yrs_since_reno', 'bedrooms', 'bathrooms', 'condition','grade']].copy()
```


```python
## Creating the y values by setting them equal to the 'price' values from the dataframe

y = df_clean['price'].copy()
```


```python
## Verifying the two groups are of equal length

print(X.shape[0] == y.shape[0])
```

    True
    

## Create Train/Test Split

Now I will split the data into the train/test groups. Then, I will run the first linear regression on the "train" data, then another regression on the "test" data.


```python
## Establishing the train and test data before doing anything else

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=5)
```


```python
## Using OneHotEncoder to create columns for the categorical variables.

## Create list of categorical variables
cat_feats = ['condition','grade', 'zipcode']

## Create the OHE without "drop='first" as it would throw an error in this case
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

## Using OHE on our categorical variables for training (NOT testing)
train_ohe = ohe.fit_transform(X_train[cat_feats])
test_ohe = ohe.transform(X_test[cat_feats])
```


```python
## Getting feature names from our list of categories
feat_col_name = ohe.get_feature_names(cat_feats)
```


```python
## Creating dataframes from the results of our fit&transform and transform

train_ohe_df = pd.DataFrame(train_ohe, columns=feat_col_name, index=X_train.index)

test_ohe_df = pd.DataFrame(test_ohe, columns=feat_col_name, index=X_test.index)
```


```python
## Reviewing one of the resulting dataframes
test_ohe_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition_1</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>grade_3</th>
      <th>grade_4</th>
      <th>grade_5</th>
      <th>grade_6</th>
      <th>grade_7</th>
      <th>grade_8</th>
      <th>grade_9</th>
      <th>grade_10</th>
      <th>grade_11</th>
      <th>grade_12</th>
      <th>zipcode_98001</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4033</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6074</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6183</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9758</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4352</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15332</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>957</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4231 rows × 85 columns</p>
</div>




```python
## Reviewing the training dataset
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>was_renovated</th>
      <th>waterfront</th>
      <th>zipcode</th>
      <th>y_old_sold</th>
      <th>yrs_since_reno</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>condition</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6486</th>
      <td>0</td>
      <td>0</td>
      <td>98032</td>
      <td>36</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>6240</th>
      <td>0</td>
      <td>0</td>
      <td>98115</td>
      <td>89</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.75</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>940</th>
      <td>0</td>
      <td>0</td>
      <td>98058</td>
      <td>8</td>
      <td>0.0</td>
      <td>4</td>
      <td>5.50</td>
      <td>3</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3011</th>
      <td>0</td>
      <td>0</td>
      <td>98136</td>
      <td>59</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>10378</th>
      <td>0</td>
      <td>0</td>
      <td>98052</td>
      <td>31</td>
      <td>0.0</td>
      <td>3</td>
      <td>3.25</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>758</th>
      <td>0</td>
      <td>0</td>
      <td>98042</td>
      <td>23</td>
      <td>0.0</td>
      <td>5</td>
      <td>3.25</td>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1055</th>
      <td>0</td>
      <td>0</td>
      <td>98115</td>
      <td>87</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5648</th>
      <td>0</td>
      <td>0</td>
      <td>98014</td>
      <td>53</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3117</th>
      <td>0</td>
      <td>0</td>
      <td>98024</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2979</th>
      <td>0</td>
      <td>0</td>
      <td>98001</td>
      <td>66</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>12692 rows × 9 columns</p>
</div>




```python
## Concatenating the two training dataframes after OHE
X_train_ohe = pd.concat([X_train.drop(cat_feats, axis=1), train_ohe_df], axis=1)
```


```python
## Concatenating the two test dataframes after OHE
X_test_ohe = pd.concat([X_test.drop(cat_feats, axis=1), test_ohe_df], axis=1)
X_test_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>was_renovated</th>
      <th>waterfront</th>
      <th>y_old_sold</th>
      <th>yrs_since_reno</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>condition_1</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>grade_3</th>
      <th>grade_4</th>
      <th>grade_5</th>
      <th>grade_6</th>
      <th>grade_7</th>
      <th>grade_8</th>
      <th>grade_9</th>
      <th>grade_10</th>
      <th>grade_11</th>
      <th>grade_12</th>
      <th>zipcode_98001</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4033</th>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6074</th>
      <td>0</td>
      <td>0</td>
      <td>56</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6183</th>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3013</th>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9758</th>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>0.0</td>
      <td>5</td>
      <td>2.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4352</th>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7198</th>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15332</th>
      <td>0</td>
      <td>0</td>
      <td>108</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1793</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.0</td>
      <td>3</td>
      <td>3.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>957</th>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4231 rows × 91 columns</p>
</div>



## Fitting the Model

Now I will create the linear regression model.


```python
## Instantiating the model and fitting it
lr = LinearRegression()

lr.fit(X_train_ohe, y_train)
```




    LinearRegression()




```python
## Checking the R^2 for train and test

train_r = lr.score(X_train_ohe, y_train)
print(f'R-Square value for training data is {round(train_r,3)}.')

test_r = lr.score(X_test_ohe, y_test)
print(f'R-Square value for test data is {round(test_r,3)}.')

## Getting model coefficients
train_coef = pd.Series(lr.coef_, index=X_train_ohe.columns)
train_coef['intercept'] = lr.intercept_

## Displaying resulting features and coefficients
train_coef.sort_values(ascending=False)#.plot(kind='barh');
```

    R-Square value for training data is 0.791.
    R-Square value for test data is 0.789.
    




    condition_5    3.065945e+15
    condition_4    3.065945e+15
    condition_3    3.065945e+15
    condition_2    3.065945e+15
    condition_1    3.065945e+15
                       ...     
    grade_3       -3.367975e+15
    grade_7       -3.367975e+15
    grade_6       -3.367975e+15
    grade_4       -3.367975e+15
    grade_5       -3.367975e+15
    Length: 92, dtype: float64




```python
## Determining predictions for training data

y_train_pred = lr.predict(X_train_ohe)

## Determining predictions for test data

y_test_pred = lr.predict(X_test_ohe)
```

## Evaluate Performance

Let's see how it performed against the test dataset!


```python
## Calculating scores for training model

train_mae = metrics.mean_absolute_error(y_train, y_train_pred)
train_mse = metrics.mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))

print('Evaluating performance of training data.')
print()
print(f'Mean Absolute Error: {train_mae:,.2f}')
print(f'Mean Squared Error:  {train_mse:,.2f}')
print()
print(f'Root Mean Squared Error: {train_rmse:,.2f}')
train_r = lr.score(X_train_ohe, y_train)
print(f'R-Square value for training data is {round(train_r,2)}.')
```

    Evaluating performance of training data.
    
    Mean Absolute Error: 83,184.52
    Mean Squared Error:  13,848,532,695.00
    
    Root Mean Squared Error: 117,679.79
    R-Square value for training data is 0.79.
    


```python
## Calculating scores for test model

test_mae = metrics.mean_absolute_error(y_test, y_test_pred)
test_mse = metrics.mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print(f'Mean Absolute Error: {test_mae:,.2f}')
print(f'Mean Squared Error: {test_mse:,.2f}')
print()
print(f'Root Mean Squared Error: {test_rmse:,.2f}')
test_r = lr.score(X_test_ohe, y_test)
print(f'R-Square value for test data is {round(test_r,2)}.')
```

    Mean Absolute Error: 84,807.42
    Mean Squared Error: 14,948,378,800.01
    
    Root Mean Squared Error: 122,263.56
    R-Square value for test data is 0.79.
    

## Comparing Results


```python
## Calculating difference between RMSE(Training) and RMSE(Test)

print('Our RMSE values for each set:')
print(f'Train RMSE: {train_rmse:,.2f}')
print(f'Test RMSE: {test_rmse:,.2f}')

diff = train_rmse - test_rmse

print()
print(f'The difference in RMSE between our training and test sets is {diff:,.2f}.')
```

    Our RMSE values for each set:
    Train RMSE: 117,679.79
    Test RMSE: 122,263.56
    
    The difference in RMSE between our training and test sets is -4,583.77.
    

## Testing Assumptions


```python
## Determining residuals
residuals = (y_test - y_test_pred)
residuals
```




    4033      23041.96875
    6074       9783.62500
    6183      65636.15625
    3013     149509.12500
    9758      27055.59375
                 ...     
    4352     112064.03125
    7198     -70832.03125
    15332   -132080.28125
    1793     -45394.78125
    957       83897.09375
    Name: price, Length: 4231, dtype: float64




```python
## Plotting to test for normality
sns.histplot(data=residuals);
```


    
![png](output_228_0.png)
    



```python
## Checking the homoscedasticity of the new model
sns.residplot(x=y_test, y=residuals, lowess=True, color="g");
```


    
![png](output_229_0.png)
    


### Interpretation of Model Results

The model does a great job of explaining the variance between our selected features, with an r-squared value of .79/1.

The RMSE for both the training and test sets are below our target of 200,000! Additionally, the difference between the two is within an acceptable range.

The assumption tests confirm the assumptions that our residuals would be normally distributed and homoscedastic.

## Recursive Feature Elimination

I have a good model created above, but I'd like to see if changing the features would have an impact for me.

**I will use the Recursive Feature Elimination method with Cross Validation to determine another selection of features.**


```python
## Creating the settings for the selector
selector = RFECV(estimator=lr, step=2, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
```


```python
## Running it on our training data
selector.fit(X_train_ohe, y_train)
```

    Fitting estimator with 91 features.
    Fitting estimator with 89 features.
    Fitting estimator with 87 features.
    Fitting estimator with 85 features.
    Fitting estimator with 83 features.
    Fitting estimator with 81 features.
    Fitting estimator with 79 features.
    Fitting estimator with 77 features.
    Fitting estimator with 75 features.
    Fitting estimator with 73 features.
    Fitting estimator with 71 features.
    Fitting estimator with 69 features.
    Fitting estimator with 67 features.
    Fitting estimator with 65 features.
    Fitting estimator with 63 features.
    Fitting estimator with 61 features.
    Fitting estimator with 59 features.
    Fitting estimator with 57 features.
    Fitting estimator with 55 features.
    Fitting estimator with 53 features.
    Fitting estimator with 51 features.
    Fitting estimator with 49 features.
    Fitting estimator with 47 features.
    Fitting estimator with 45 features.
    Fitting estimator with 43 features.
    Fitting estimator with 41 features.
    Fitting estimator with 39 features.
    Fitting estimator with 37 features.
    Fitting estimator with 35 features.
    Fitting estimator with 33 features.
    Fitting estimator with 31 features.
    Fitting estimator with 29 features.
    Fitting estimator with 27 features.
    Fitting estimator with 25 features.
    Fitting estimator with 23 features.
    Fitting estimator with 21 features.
    Fitting estimator with 19 features.
    Fitting estimator with 17 features.
    Fitting estimator with 15 features.
    




    RFECV(cv=3, estimator=LinearRegression(), n_jobs=-1,
          scoring='neg_mean_squared_error', step=2, verbose=1)




```python
## Dividing the data into two separate variables for inspection
selected_columns = X_train_ohe.columns[selector.get_support()]
removed_columns = X_train_ohe.columns[~selector.get_support()]
```


```python
## Inspecting the columns with the greatest impact according to the RSECV method
list(selected_columns.values)
```




    ['condition_1',
     'condition_2',
     'condition_3',
     'condition_4',
     'condition_5',
     'grade_3',
     'grade_4',
     'grade_5',
     'grade_6',
     'grade_7',
     'grade_8',
     'grade_9',
     'grade_10']




```python
## Testing the new list of columns

## Instantiating a new regression model
lr_rfe = LinearRegression()

## Fitting the model to the new data
lr_rfe.fit(X_train_ohe[selected_columns], y_train)

# Using the model to create new predictions
y_rfe = lr_rfe.predict(X_train_ohe[selected_columns])

# Determining RMSE for new list of columns and TRAIN data
trainRFE_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_rfe))

print(f'Training Root Mean Squared Error: {trainRFE_rmse:,.2f}')

# Determining RMSE for new list of columns and TEST data
y_pred_rfe = lr_rfe.predict(X_test_ohe[selected_columns])

testRFE_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfe))

print(f'Testing Root Mean Squared Error: {testRFE_rmse:,.2f}')

diff2 = trainRFE_rmse - testRFE_rmse
print()
print(f'The difference in RMSE between our training and test sets is {diff2:,.2f}.')
```

    Training Root Mean Squared Error: 182,158.41
    Testing Root Mean Squared Error: 191,360.88
    
    The difference in RMSE between our training and test sets is -9,202.47.
    


```python
test_rsq = lr_rfe.score(X_test_ohe[selected_columns], y_test)
print(f'R-Square value for testing data after RFECV is {round(test_rsq,2)}.')
```

    R-Square value for testing data after RFECV is 0.48.
    

### Interpretation of Results of RFEVC

---
**Comparing RMSEs from original model to RFECV:**

* **Pre-RFECV model results:**
   * The R^2 value for both training and testing data was .79 (significant as it is over the .70 threshold).
   * *Our RMSE values for each set:*
        * Train RMSE: 117,679.79
        * Test RMSE: 122,263.56
    * The difference in RMSE between our training and test sets is **-4,583.77**.


 * **Post-RFECV model results:**
     * The R^2 value for the test data is .48 (.31 lower than other model, and far below threshold).
     * *Our RMSE values for each set:*
         * Training Root Mean Squared Error: 182,158.41
         * Testing Root Mean Squared Error: 191,360.88
     * The difference in RMSE between our training and test sets is **-9,202.47**.

---

**Final Analysis of RFE method:** The RFE results produced a large decrease in model perfomance as determined by the lower R^2 and lower RMSE values. I will continue using the first model for future predictions.

---

# **Model Testing - Original Data**

The model produced above was based on data from which we removed any outliers. I am re-running the tests to determine if there is any difference in the results. 

**My goal is to have a Root Mean Squared Error (RMSE) less than 200,000.**

## ID Features, Target


```python
## Saving the features that I used for inferential modeling - I will use these features 
## to fit my model.

X = df[['was_renovated','waterfront', 'zipcode', 'y_old_sold',
            'yrs_since_reno', 'bedrooms', 'bathrooms', 'condition','grade']].copy()
```


```python
## Creating the y values by setting them equal to the 'price' values from the dataframe

y = df['price'].copy()
```


```python
## Verifying the two groups are of equal length

print(X.shape[0] == y.shape[0])
```

    True
    

## Create Train/Test Split

Now I will split the data into the train/test groups. Then, I will run the first linear regression on the "train" data, then another regression on the "test" data.


```python
## Establishing the train and test data before doing anything else

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=5)
```


```python
## Using OneHotEncoder to create columns for the categorical variables.

## Create list of categorical variables
cat_feats = ['condition','grade', 'zipcode']

## Create the OHE without "drop='first" as it would throw an error in this case
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

## Using OHE on our categorical variables for training (NOT testing)
train_ohe = ohe.fit_transform(X_train[cat_feats])
test_ohe = ohe.transform(X_test[cat_feats])
```


```python
## Getting feature names from our list of categories
feat_col_name = ohe.get_feature_names(cat_feats)
```


```python
## Creating dataframes from the results of our fit&transform and transform

train_ohe_df = pd.DataFrame(train_ohe, columns=feat_col_name, index=X_train.index)

test_ohe_df = pd.DataFrame(test_ohe, columns=feat_col_name, index=X_test.index)
```


```python
## Reviewing one of the resulting dataframes
test_ohe_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition_1</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>grade_1</th>
      <th>grade_3</th>
      <th>grade_4</th>
      <th>grade_5</th>
      <th>grade_6</th>
      <th>grade_7</th>
      <th>grade_8</th>
      <th>grade_9</th>
      <th>grade_10</th>
      <th>grade_11</th>
      <th>grade_12</th>
      <th>grade_13</th>
      <th>zipcode_98001</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3701</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3580</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3903</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12786</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3604</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15993</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12954</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13200</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4323 rows × 87 columns</p>
</div>




```python
X_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>was_renovated</th>
      <th>waterfront</th>
      <th>zipcode</th>
      <th>y_old_sold</th>
      <th>yrs_since_reno</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>condition</th>
      <th>grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14038</th>
      <td>0</td>
      <td>0</td>
      <td>98058</td>
      <td>40</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2202</th>
      <td>0</td>
      <td>0</td>
      <td>98052</td>
      <td>46</td>
      <td>0.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>325</th>
      <td>0</td>
      <td>0</td>
      <td>98030</td>
      <td>2</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9061</th>
      <td>0</td>
      <td>0</td>
      <td>98168</td>
      <td>58</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>251</th>
      <td>0</td>
      <td>0</td>
      <td>98119</td>
      <td>65</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.50</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>740</th>
      <td>0</td>
      <td>0</td>
      <td>98133</td>
      <td>67</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1032</th>
      <td>0</td>
      <td>0</td>
      <td>98027</td>
      <td>45</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>5520</th>
      <td>0</td>
      <td>0</td>
      <td>98031</td>
      <td>32</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3046</th>
      <td>0</td>
      <td>0</td>
      <td>98030</td>
      <td>51</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2915</th>
      <td>0</td>
      <td>0</td>
      <td>98038</td>
      <td>24</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>3</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>12967 rows × 9 columns</p>
</div>




```python
## Combining the training dataframes after performing OHE
X_train_ohe = pd.concat([X_train.drop(cat_feats, axis=1), train_ohe_df], axis=1)
```


```python
## Combining the testing dataframes after performing OHE
X_test_ohe = pd.concat([X_test.drop(cat_feats, axis=1), test_ohe_df], axis=1)
X_test_ohe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>was_renovated</th>
      <th>waterfront</th>
      <th>y_old_sold</th>
      <th>yrs_since_reno</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>condition_1</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>grade_1</th>
      <th>grade_3</th>
      <th>grade_4</th>
      <th>grade_5</th>
      <th>grade_6</th>
      <th>grade_7</th>
      <th>grade_8</th>
      <th>grade_9</th>
      <th>grade_10</th>
      <th>grade_11</th>
      <th>grade_12</th>
      <th>grade_13</th>
      <th>zipcode_98001</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3701</th>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3580</th>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3903</th>
      <td>0</td>
      <td>1</td>
      <td>34</td>
      <td>0.0</td>
      <td>4</td>
      <td>4.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12786</th>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>0</td>
      <td>0</td>
      <td>68</td>
      <td>0.0</td>
      <td>4</td>
      <td>1.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3604</th>
      <td>0</td>
      <td>0</td>
      <td>14</td>
      <td>0.0</td>
      <td>3</td>
      <td>3.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15993</th>
      <td>0</td>
      <td>0</td>
      <td>23</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12954</th>
      <td>0</td>
      <td>0</td>
      <td>94</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13200</th>
      <td>0</td>
      <td>0</td>
      <td>54</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14995</th>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4323 rows × 93 columns</p>
</div>



## Fitting the Model


```python
## Instantiating the model and fitting it
lr = LinearRegression()

lr.fit(X_train_ohe, y_train)
```




    LinearRegression()




```python
## Checking the R^2 for train and test

train_r = lr.score(X_train_ohe, y_train)
print(f'R-Square value for training data is {round(train_r,3)}.')

test_r = lr.score(X_test_ohe, y_test)
print(f'R-Square value for test data is {round(test_r,3)}.')

## Getting model coefficients
train_coef = pd.Series(lr.coef_, index=X_train_ohe.columns)
train_coef['intercept'] = lr.intercept_

## Displaying resulting features and coefficients
train_coef.sort_values(ascending=False)[:11]#.plot(kind='barh');
```

    R-Square value for training data is 0.788.
    R-Square value for test data is 0.79.
    




    grade_13    1.404118e+14
    grade_12    1.404118e+14
    grade_11    1.404118e+14
    grade_10    1.404118e+14
    grade_9     1.404118e+14
    grade_8     1.404118e+14
    grade_3     1.404118e+14
    grade_7     1.404118e+14
    grade_1     1.404118e+14
    grade_4     1.404118e+14
    grade_6     1.404118e+14
    dtype: float64




```python
## Determining predictions for training data

y_train_pred = lr.predict(X_train_ohe)


## Determining predictions for test data

y_test_pred = lr.predict(X_test_ohe)

```

## Evaluate Performance


```python
## Calculating scores for training model

train_mae = metrics.mean_absolute_error(y_train, y_train_pred)
train_mse = metrics.mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))

print('Evaluating performance of training data.')
print()
print(f'Mean Absolute Error: {train_mae:,.2f}')
print(f'Mean Squared Error:  {train_mse:,.2f}')
print()
print(f'Root Mean Squared Error: {train_rmse:,.2f}')
train_r = lr.score(X_train_ohe, y_train)
print(f'R-Square value for training data is {round(train_r,2)}.')
```

    Evaluating performance of training data.
    
    Mean Absolute Error: 100,712.64
    Mean Squared Error:  28,849,162,106.62
    
    Root Mean Squared Error: 169,850.41
    R-Square value for training data is 0.79.
    


```python
## Calculating scores for test model

test_mae = metrics.mean_absolute_error(y_test, y_test_pred)
test_mse = metrics.mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

print(f'Mean Absolute Error: {test_mae:,.2f}')
print(f'Mean Squared Error: {test_mse:,.2f}')
print()
print(f'Root Mean Squared Error: {test_rmse:,.2f}')
test_r = lr.score(X_test_ohe, y_test)
print(f'R-Square value for test data is {round(test_r,2)}.')
```

    Mean Absolute Error: 101,914.46
    Mean Squared Error: 31,243,540,022.14
    
    Root Mean Squared Error: 176,758.42
    R-Square value for test data is 0.79.
    

## Comparing Results


```python
## Calculating difference between RMSE(Training) and RMSE(Test)

print('Our RMSE values for each set:')
print(f'Train RMSE: {train_rmse:,.2f}')
print(f'Test RMSE: {test_rmse:,.2f}')

diff = train_rmse - test_rmse

print()
print(f'The difference in RMSE between our training and test sets is {diff:,.2f}.')
```

    Our RMSE values for each set:
    Train RMSE: 169,850.41
    Test RMSE: 176,758.42
    
    The difference in RMSE between our training and test sets is -6,908.01.
    

## Testing Assumptions


```python
## Determining residuals
residuals = (y_test - y_test_pred)
residuals
```




    3701    -1.906941e+05
    3580     3.038108e+04
    3903     1.282617e+06
    12786   -6.472320e+03
    1694     9.917154e+04
                 ...     
    3604    -2.094837e+05
    15993   -4.126162e+04
    12954    5.035755e+04
    13200   -2.724516e+04
    14995   -5.624730e+04
    Name: price, Length: 4323, dtype: float64




```python
## Plotting to test for normality
sns.histplot(data=residuals);
```


    
![png](output_269_0.png)
    



```python
## Checking the homoscedasticity of the new model
sns.residplot(x=y_test, y=residuals, lowess=True, color="g");
```


    
![png](output_270_0.png)
    


### Interpretation of Model Results

The model continues to do a job of explaining the variance between our selected features, with an r-squared value of .79/1.

The RMSE for both the training and test sets are below our target of 200,000, but higher than our model without the outliers. The difference between the two RMSEs did increase, indicating area for improvement.

The assumption tests confirm the assumptions that our residuals would be normally distributed and homoscedastic.

## Recursive Feature Elimination

The model using the original data performed worse in terms of RMSE, but maintained the R^2 value of .79.

Let's see if the RFECV has any impact this time.


```python
## Creating the settings for the selector
selector = RFECV(estimator=lr, step=2, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
```


```python
## Running it on our training data
selector.fit(X_train_ohe, y_train)
```

    Fitting estimator with 93 features.
    Fitting estimator with 91 features.
    Fitting estimator with 89 features.
    Fitting estimator with 87 features.
    Fitting estimator with 85 features.
    Fitting estimator with 83 features.
    Fitting estimator with 81 features.
    Fitting estimator with 79 features.
    Fitting estimator with 77 features.
    Fitting estimator with 75 features.
    Fitting estimator with 73 features.
    Fitting estimator with 71 features.
    Fitting estimator with 69 features.
    Fitting estimator with 67 features.
    Fitting estimator with 65 features.
    Fitting estimator with 63 features.
    Fitting estimator with 61 features.
    Fitting estimator with 59 features.
    Fitting estimator with 57 features.
    Fitting estimator with 55 features.
    Fitting estimator with 53 features.
    Fitting estimator with 51 features.
    Fitting estimator with 49 features.
    Fitting estimator with 47 features.
    Fitting estimator with 45 features.
    Fitting estimator with 43 features.
    Fitting estimator with 41 features.
    Fitting estimator with 39 features.
    Fitting estimator with 37 features.
    Fitting estimator with 35 features.
    Fitting estimator with 33 features.
    Fitting estimator with 31 features.
    Fitting estimator with 29 features.
    Fitting estimator with 27 features.
    Fitting estimator with 25 features.
    Fitting estimator with 23 features.
    Fitting estimator with 21 features.
    Fitting estimator with 19 features.
    Fitting estimator with 17 features.
    Fitting estimator with 15 features.
    Fitting estimator with 13 features.
    Fitting estimator with 11 features.
    




    RFECV(cv=3, estimator=LinearRegression(), n_jobs=-1,
          scoring='neg_mean_squared_error', step=2, verbose=1)




```python
## Dividing the data into two separate variables for inspection
selected_columns = X_train_ohe.columns[selector.get_support()]
removed_columns = X_train_ohe.columns[~selector.get_support()]
```


```python
## Inspecting the columns with the greatest impact according to the RSECV method
list(selected_columns.values)
```




    ['grade_1',
     'grade_6',
     'grade_7',
     'grade_8',
     'grade_9',
     'grade_10',
     'grade_11',
     'grade_12',
     'grade_13']




```python
## Testing the new list of columns

## Instantiating a new regression model
lr_rfe = LinearRegression()

## Fitting the model to the new data
lr_rfe.fit(X_train_ohe[selected_columns], y_train)

# Using the model to create new predictions
y_rfe = lr_rfe.predict(X_train_ohe[selected_columns])

# Determining RMSE for new list of columns and TRAIN data
trainRFE_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_rfe))

print(f'Training Root Mean Squared Error: {trainRFE_rmse:,.2f}')

# Determining RMSE for new list of columns and TEST data
y_pred_rfe = lr_rfe.predict(X_test_ohe[selected_columns])

testRFE_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_rfe))

print(f'Testing Root Mean Squared Error: {testRFE_rmse:,.2f}')

diff2 = trainRFE_rmse - testRFE_rmse
print()
print(f'The difference in RMSE between our training and test sets is {diff2:,.2f}.')
```

    Training Root Mean Squared Error: 255,961.96
    Testing Root Mean Squared Error: 258,659.15
    
    The difference in RMSE between our training and test sets is -2,697.18.
    


```python
test_rsq = lr_rfe.score(X_test_ohe[selected_columns], y_test)
print(f'R-Square value for testing data after RFECV is {round(test_rsq,2)}.')
```

    R-Square value for testing data after RFECV is 0.55.
    

---
**Comparing RMSEs from original to RFECV:**

* **Pre-RFECV model results:**
   * The R^2 value for both training and testing data was .79 (significant as it is over the .70 threshold).
   * *Our RMSE values for each set:*
        * Train RMSE: 117,679.79
        * Test RMSE: 122,263.56
    * The difference in RMSE between our training and test sets is **-4,583.77**.


 * **Post-RFECV model results:**
     * The R^2 value for the test data is .48 (.31 lower than other model, and far below threshold).
     * *Our RMSE values for each set:*
         * Training Root Mean Squared Error: 182,158.41
         * Testing Root Mean Squared Error: 191,360.88
     * The difference in RMSE between our training and test sets is **-9,202.47**.

---

**Final Analysis of RFE method:** The RFE results produced a large decrease in model perfomance as determined by the lower R^2 and lower RMSE values. I will continue with the first model for future predictions.

---

# Pickling the Model for Reuse


```python
## Setting coefficients and target
X = df[['was_renovated','waterfront', 'zipcode', 'y_old_sold',
            'yrs_since_reno', 'bedrooms', 'bathrooms', 'condition','grade']].copy()
y = df['price'].copy()
```


```python
## Create list of categorical variables
cat_feats = ['condition','grade', 'zipcode']

## Create the OHE without "drop='first" as it would throw an error in this case
ohe = OneHotEncoder(handle_unknown = 'ignore', sparse=False)

## Using OHE on our categorical variables for training (NOT testing)
final_ohe = ohe.fit_transform(X[cat_feats])
```


```python
## Getting feature names from our list of categories
feat_col_name = ohe.get_feature_names(cat_feats)
```


```python
## Creating DF of final results
final_ohe_df1 = pd.DataFrame(final_ohe, columns=feat_col_name, index=X.index)
```


```python
final_ohe_df = pd.concat([X.drop(cat_feats, axis=1), final_ohe_df1], axis=1)
```


```python
lr_final = LinearRegression()

lr_final = lr_final.fit(final_ohe_df, y)
```


```python
final_pred = lr_final.predict(final_ohe_df)
```


```python
display(final_pred)
```


    array([ 392072.25,  805538.5 , 1301535.5 , ...,  304438.25,  259248.75,
            371298.75])



```python
import pickle


pickle_out = open("final_model.pickle","wb")
pickle.dump(lr_final, pickle_out)
pickle_out.close()
```


```python
final_ohe_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>was_renovated</th>
      <th>waterfront</th>
      <th>y_old_sold</th>
      <th>yrs_since_reno</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>condition_1</th>
      <th>condition_2</th>
      <th>condition_3</th>
      <th>condition_4</th>
      <th>condition_5</th>
      <th>grade_1</th>
      <th>grade_3</th>
      <th>grade_4</th>
      <th>grade_5</th>
      <th>grade_6</th>
      <th>grade_7</th>
      <th>grade_8</th>
      <th>grade_9</th>
      <th>grade_10</th>
      <th>grade_11</th>
      <th>grade_12</th>
      <th>grade_13</th>
      <th>zipcode_98001</th>
      <th>zipcode_98002</th>
      <th>zipcode_98003</th>
      <th>zipcode_98004</th>
      <th>zipcode_98005</th>
      <th>zipcode_98006</th>
      <th>zipcode_98007</th>
      <th>zipcode_98008</th>
      <th>zipcode_98010</th>
      <th>zipcode_98011</th>
      <th>zipcode_98014</th>
      <th>zipcode_98019</th>
      <th>zipcode_98022</th>
      <th>zipcode_98023</th>
      <th>zipcode_98024</th>
      <th>zipcode_98027</th>
      <th>zipcode_98028</th>
      <th>zipcode_98029</th>
      <th>zipcode_98030</th>
      <th>zipcode_98031</th>
      <th>zipcode_98032</th>
      <th>zipcode_98033</th>
      <th>zipcode_98034</th>
      <th>zipcode_98038</th>
      <th>zipcode_98039</th>
      <th>zipcode_98040</th>
      <th>zipcode_98042</th>
      <th>zipcode_98045</th>
      <th>zipcode_98052</th>
      <th>zipcode_98053</th>
      <th>zipcode_98055</th>
      <th>zipcode_98056</th>
      <th>zipcode_98058</th>
      <th>zipcode_98059</th>
      <th>zipcode_98065</th>
      <th>zipcode_98070</th>
      <th>zipcode_98072</th>
      <th>zipcode_98074</th>
      <th>zipcode_98075</th>
      <th>zipcode_98077</th>
      <th>zipcode_98092</th>
      <th>zipcode_98102</th>
      <th>zipcode_98103</th>
      <th>zipcode_98105</th>
      <th>zipcode_98106</th>
      <th>zipcode_98107</th>
      <th>zipcode_98108</th>
      <th>zipcode_98109</th>
      <th>zipcode_98112</th>
      <th>zipcode_98115</th>
      <th>zipcode_98116</th>
      <th>zipcode_98117</th>
      <th>zipcode_98118</th>
      <th>zipcode_98119</th>
      <th>zipcode_98122</th>
      <th>zipcode_98125</th>
      <th>zipcode_98126</th>
      <th>zipcode_98133</th>
      <th>zipcode_98136</th>
      <th>zipcode_98144</th>
      <th>zipcode_98146</th>
      <th>zipcode_98148</th>
      <th>zipcode_98155</th>
      <th>zipcode_98166</th>
      <th>zipcode_98168</th>
      <th>zipcode_98177</th>
      <th>zipcode_98178</th>
      <th>zipcode_98188</th>
      <th>zipcode_98198</th>
      <th>zipcode_98199</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>37</td>
      <td>0.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>0.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>0.0</td>
      <td>3</td>
      <td>3.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>17285</th>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0.0</td>
      <td>5</td>
      <td>2.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17286</th>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.75</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17287</th>
      <td>0</td>
      <td>0</td>
      <td>61</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17288</th>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17289</th>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>0.0</td>
      <td>3</td>
      <td>1.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>17290 rows × 93 columns</p>
</div>




```python

```
