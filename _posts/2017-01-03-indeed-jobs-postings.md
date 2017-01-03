---
title: "Web Scraping Indeed.com to Predict Salaries"
header:
  overlay_color: "#333"
date: 2017-01-03
tags: Web scraping, Indeed.com, Data science jobs
---


For this project, we used a set of characteristics to predict the salaries for data science jobs in the Boston area. We analyzed search results from Indeed.com, an online jobs board, and highlighted key phrases in each job posting. We also determined if each job's salary is above a threshold value. We used logistic regression to model this data. Machine learning and python experience were the strongest predictors of salary. Machine learning, python and scientist were the strongest predictors of salary. The model had poor precision --- approximately 63% precision and recall, where random guesses could have achieved 57% --- which may have been caused by variability in the source salary data.

Part I outlines the code used. [Part II](#Part-II) contains the writeup.

## Part I: Code

#### Import libraries


```python
# Webscraping
from urllib.request import urlopen
from bs4 import BeautifulSoup

#Data and analysis
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn import preprocessing, linear_model, metrics

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize': 20,
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
```

#### Define webscraping functions


```python
def count_results(query = None, location = "Boston", binary_level = 1):
    job_ids = pd.DataFrame()

    result_list = []

    # Find the number of results
    URL_for_count = "http://www.indeed.com/jobs?q=data+scientist+{}&jt=fulltime&l={}".format(query, location)
    soup_for_count = BeautifulSoup(urlopen(URL_for_count).read(), 'html.parser')

    results_number = soup_for_count.find("div", attrs = {"id": "searchCount"}).text
    number_of_results = int(results_number.split(sep = ' ')[-1].replace(',', ''))

    # Now loop through the pages. Viewing 100 results at a time means fewer page refreshes, which are a bottleneck.
    i = int(number_of_results/100)
    for page_number in range(i + 1):
        URL_for_results = "http://www.indeed.com/jobs?q=data+scientist+{}&jt=fulltime&l={}&limit=100&start={}".format(query, location, str(100 * page_number))
        soup_for_results = BeautifulSoup(urlopen(URL_for_results).read(), 'html.parser')
        results = soup_for_results.find_all('div', attrs={'data-tn-component': 'organicJob'})

        # Extract the ID for each job listing
        for x in results:
            job_id = x.find('h2', attrs={"class": "jobtitle"})['id']
            job_title = x.find('a', attrs={'data-tn-element': "jobTitle"}).text.strip().capitalize()
            job_link = "https://www.indeed.com" + x.find('h2', attrs={"class": "jobtitle"}).find('a')['href']
            result_list.append([job_id, job_title, job_link, binary_level])

        # Add the job ID numbers
        job_ids = job_ids.append(result_list)

    # Remove re-posted jobs
    job_ids.drop_duplicates(inplace = True)
    return job_ids
```


```python
# String format: "keyword, keyword combination, keyword, ..."
def count_results_by_keywords(query_string = None):

    # Ends the function if given invalid inputs
    if query_string == None:
        return(print("No keyword entered."))

    # Format the keyword string in to URL query
    query = "%20OR%20".join(query_string.split(", "))
    query = query.replace(' ', '+')

    # Perform the search
    job_ids = count_results("%28{}%29".format(query))

    # Rename job_ids's columns
    job_ids.columns = ['id', 'job title', 'job link', '{}'.format(" OR ".join(query_string.split(", ")))]

    return (job_ids)


def count_results_by_salary(salary_range_divider = None, salary_floor = 20000):

    if salary_range_divider <= salary_floor:
        return(print("Enter a number larger than ${}.".format(salary_floor)))

    job_ids = pd.DataFrame()
    # Set dividing salaries
    divider_strings = ["+${}-{}".format(salary_floor, salary_range_divider), "+${}".format(salary_range_divider)]

    # Perform two searches, starting with the low-salary jobs
    for level, salary_criterion in enumerate(divider_strings):
        job_ids = job_ids.append(count_results(salary_criterion, binary_level=level))

    # Rename job_ids's columns
    job_ids.columns = ['id', 'job title', 'job link', 'salary over {}'.format(salary_range_divider)]

    return(job_ids)


def count_results_by_years_experience(years_experience = None):

    # Ends the function if given invalid inputs
    if years_experience == None or type(years_experience) != int:
        return(print("Enter an integer value."))

    # Format the keyword string in to URL query
    query = "{}+years+or+{}%2B+years".format(str(years_experience), str(years_experience))

    # Perform the search
    job_ids = count_results("%28{}%29".format(query), binary_level = years_experience)

    # Rename job_ids's columns
    job_ids.columns = ['id', 'job title', 'job link', 'years experience']

    return (job_ids)
```

#### Collect data

Each dataframe is produced in a separate cell so that keywords can be adjusted or new searches added more quickly.


```python
phd_dataframe = count_results_by_keywords("PhD, ph.d")
```


```python
bachelors_dataframe = count_results_by_keywords("Bachelors, BS, BA") # Indeed.com's search includes "Bachelor's", "B.S.", etc.
```


```python
python_dataframe = count_results_by_keywords("Python")
```


```python
# Use %22 ... %22 to ensure at least one of these keywords are present
management_dataframe = count_results_by_keywords("%22Manager%22, %22director%22")

# Rename the column to a readable format
management_dataframe.columns = ['id', 'job title', 'job link', 'Manager OR director']
```


```python
startup_dataframe = count_results_by_keywords("Startup, start-up")
```


```python
scientist_dataframe = count_results_by_keywords("Scientist")
```


```python
machine_learning_dataframe = count_results_by_keywords("Machine learning")
```


```python
lab_dataframe = count_results_by_keywords("Lab, laboratory")
```


```python
# Use %22 ... %22 to search for the exact phrase "Software engineer"
software_engineer_dataframe = count_results_by_keywords("%22Software engineer%22")

# Rename the column to a readable format
software_engineer_dataframe.columns = ['id', 'job title', 'job link', 'Software engineer']
```


```python
experience_results_dataframe = pd.DataFrame()

# Scan search results from a range of year requirements
for years in range(1+7):
    experience_results_dataframe = experience_results_dataframe.append(count_results_by_years_experience(years))

experience_results_dataframe['years experience'].astype(int).astype('category', ordered = True, copy = False)
years_dummies_dataframe = pd.get_dummies(experience_results_dataframe['years experience'], prefix = 'years experience', prefix_sep = ' ')
experience_dataframe = pd.concat(objs = [experience_results_dataframe.drop('years experience', axis = 1), years_dummies_dataframe], axis = 1)
```


```python
salary_dataframe = count_results_by_salary(90000)
```

#### Combine data


```python
common_columns = ['id', 'job title', 'job link']
master_dataframe = phd_dataframe.merge(bachelors_dataframe, on = common_columns, how = 'outer')\
    .merge(python_dataframe, on = common_columns, how = 'outer')\
    .merge(startup_dataframe, on = common_columns, how = 'outer')\
    .merge(scientist_dataframe, on = common_columns, how = 'outer')\
    .merge(machine_learning_dataframe, on = common_columns, how = 'outer')\
    .merge(lab_dataframe, on = common_columns, how = 'outer')\
    .merge(software_engineer_dataframe, on = common_columns, how = 'outer')\
    .merge(management_dataframe, on = common_columns, how = 'outer')\
    .merge(experience_dataframe, on = common_columns, how = 'outer')\
    .merge(salary_dataframe, on = common_columns, how = 'outer')

# Convert non-id columns to integers and fill NaN values
data_conversion_mask = (master_dataframe.columns != 'id') & (master_dataframe.columns != 'job title') & (master_dataframe.columns != 'job link')
master_dataframe.ix[:, data_conversion_mask] = master_dataframe.ix[:, data_conversion_mask].fillna(value = 0).astype(int)
```


```python
master_dataframe.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>job title</th>
      <th>job link</th>
      <th>PhD OR ph.d</th>
      <th>Bachelors OR BS OR BA</th>
      <th>Python</th>
      <th>Startup OR start-up</th>
      <th>Scientist</th>
      <th>Machine learning</th>
      <th>Lab OR laboratory</th>
      <th>...</th>
      <th>Manager OR director</th>
      <th>years experience 0</th>
      <th>years experience 1</th>
      <th>years experience 2</th>
      <th>years experience 3</th>
      <th>years experience 4</th>
      <th>years experience 5</th>
      <th>years experience 6</th>
      <th>years experience 7</th>
      <th>salary over 90000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>jl_7896e61d3e45dfb4</td>
      <td>Scientist, research and modeling - cyber risk</td>
      <td>https://www.indeed.com/rc/clk?jk=7896e61d3e45d...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jl_a116f7a8d8db626e</td>
      <td>Research scientist, muscle</td>
      <td>https://www.indeed.com/rc/clk?jk=a116f7a8d8db6...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jl_89b3b83a5b0fd82c</td>
      <td>Cellular analytics scientist</td>
      <td>https://www.indeed.com/rc/clk?jk=89b3b83a5b0fd...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 21 columns</p>
</div>




```python
# Create a dataframe that keeps only the data features for analysis
data_dataframe_mask = (master_dataframe['Lab OR laboratory'] == 0) | (master_dataframe['Software engineer'] == 1)
data_dataframe = master_dataframe[data_dataframe_mask].drop(labels = common_columns, axis = 1).drop(labels = ['Lab OR laboratory', 'Software engineer'], axis = 1)
#data_dataframe.loc[:, 'years experience'] = sklearn.preprocessing.minmax_scale(data_dataframe['years experience'].astype(float), copy = False)
```

#### Model with statsmodel logistic regression


```python
sm_model_dataframe = data_dataframe.copy()
sm_model_dataframe.columns = sm_model_dataframe.columns.str.replace(' ', '_').str.replace('.', '_').str.replace('-', '_')
sm_model_dataframe.columns
```




    Index(['PhD_OR_ph_d', 'Bachelors_OR_BS_OR_BA', 'Python', 'Startup_OR_start_up',
           'Scientist', 'Machine_learning', 'Manager_OR_director',
           'years_experience_0', 'years_experience_1', 'years_experience_2',
           'years_experience_3', 'years_experience_4', 'years_experience_5',
           'years_experience_6', 'years_experience_7', 'salary_over_90000'],
          dtype='object')




```python
sm_model = sm.logit("salary_over_90000 ~ PhD_OR_ph_d + Bachelors_OR_BS_OR_BA + Python + Startup_OR_start_up + Scientist + Machine_learning + Manager_OR_director + years_experience_0 + years_experience_1 + years_experience_2 + years_experience_3 + years_experience_4 + years_experience_5 + years_experience_6 + years_experience_7", data=sm_model_dataframe).fit()
sm_model.summary()

```

    Optimization terminated successfully.
             Current function value: 0.625174
             Iterations 5





<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th> <td>salary_over_90000</td> <th>  No. Observations:  </th>  <td>  1128</td>  
</tr>
<tr>
  <th>Model:</th>               <td>Logit</td>       <th>  Df Residuals:      </th>  <td>  1112</td>  
</tr>
<tr>
  <th>Method:</th>               <td>MLE</td>        <th>  Df Model:          </th>  <td>    15</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Tue, 03 Jan 2017</td>  <th>  Pseudo R-squ.:     </th>  <td>0.08790</td>
</tr>
<tr>
  <th>Time:</th>              <td>10:06:47</td>      <th>  Log-Likelihood:    </th> <td> -705.20</td>
</tr>
<tr>
  <th>converged:</th>           <td>True</td>        <th>  LL-Null:           </th> <td> -773.16</td>
</tr>
<tr>
  <th> </th>                      <td> </td>         <th>  LLR p-value:       </th> <td>1.460e-21</td>
</tr>
</table>
<table class="simpletable">
<tr>
            <td></td>               <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th> <th>[95.0% Conf. Int.]</th>
</tr>
<tr>
  <th>Intercept</th>             <td>   -0.6329</td> <td>    0.132</td> <td>   -4.795</td> <td> 0.000</td> <td>   -0.892    -0.374</td>
</tr>
<tr>
  <th>PhD_OR_ph_d</th>           <td>   -0.0532</td> <td>    0.148</td> <td>   -0.360</td> <td> 0.718</td> <td>   -0.343     0.236</td>
</tr>
<tr>
  <th>Bachelors_OR_BS_OR_BA</th> <td>    0.0686</td> <td>    0.173</td> <td>    0.397</td> <td> 0.691</td> <td>   -0.270     0.407</td>
</tr>
<tr>
  <th>Python</th>                <td>    1.1100</td> <td>    0.181</td> <td>    6.117</td> <td> 0.000</td> <td>    0.754     1.466</td>
</tr>
<tr>
  <th>Startup_OR_start_up</th>   <td>    0.2948</td> <td>    0.216</td> <td>    1.366</td> <td> 0.172</td> <td>   -0.128     0.718</td>
</tr>
<tr>
  <th>Scientist</th>             <td>    0.5263</td> <td>    0.168</td> <td>    3.125</td> <td> 0.002</td> <td>    0.196     0.856</td>
</tr>
<tr>
  <th>Machine_learning</th>      <td>    0.4836</td> <td>    0.194</td> <td>    2.489</td> <td> 0.013</td> <td>    0.103     0.864</td>
</tr>
<tr>
  <th>Manager_OR_director</th>   <td>    0.4290</td> <td>    0.169</td> <td>    2.537</td> <td> 0.011</td> <td>    0.098     0.760</td>
</tr>
<tr>
  <th>years_experience_0</th>    <td>    0.7374</td> <td>    1.157</td> <td>    0.637</td> <td> 0.524</td> <td>   -1.531     3.006</td>
</tr>
<tr>
  <th>years_experience_1</th>    <td>   -0.2845</td> <td>    0.254</td> <td>   -1.119</td> <td> 0.263</td> <td>   -0.783     0.214</td>
</tr>
<tr>
  <th>years_experience_2</th>    <td>   -0.1667</td> <td>    0.219</td> <td>   -0.760</td> <td> 0.447</td> <td>   -0.597     0.263</td>
</tr>
<tr>
  <th>years_experience_3</th>    <td>   -0.0004</td> <td>    0.212</td> <td>   -0.002</td> <td> 0.998</td> <td>   -0.417     0.416</td>
</tr>
<tr>
  <th>years_experience_4</th>    <td>   -0.0320</td> <td>    0.304</td> <td>   -0.105</td> <td> 0.916</td> <td>   -0.628     0.564</td>
</tr>
<tr>
  <th>years_experience_5</th>    <td>    0.2982</td> <td>    0.206</td> <td>    1.450</td> <td> 0.147</td> <td>   -0.105     0.701</td>
</tr>
<tr>
  <th>years_experience_6</th>    <td>    0.2757</td> <td>    0.403</td> <td>    0.685</td> <td> 0.493</td> <td>   -0.513     1.065</td>
</tr>
<tr>
  <th>years_experience_7</th>    <td>    0.3683</td> <td>    0.360</td> <td>    1.024</td> <td> 0.306</td> <td>   -0.336     1.073</td>
</tr>
</table>



#### Model with sklearn's logistic regression


```python
import sklearn
X = data_dataframe.drop('salary over 90000', axis = 1)
y = data_dataframe['salary over 90000']
skl_logreg = sklearn.linear_model.LogisticRegressionCV(cv = 6)
skl_model = skl_logreg.fit(X, y)
predictions = skl_model.predict(X)

model_confusion_matrix = sklearn.metrics.confusion_matrix(y, predictions)
```


```python
print("Average cross validation scores: {}".format(sklearn.cross_validation.cross_val_score(skl_logreg, X, y, scoring='roc_auc').mean()))
print("Confusion matrix:\n", model_confusion_matrix)
print("Model performance: \n", sklearn.metrics.classification_report(y, predictions))
print("Area under ROC curve for model: {}".format(sklearn.metrics.roc_auc_score(y, predictions)))
```

    Average cross validation scores: 0.6704394926246255
    Confusion matrix:
     [[146 348]
     [ 79 555]]
    Model performance:
                  precision    recall  f1-score   support

              0       0.65      0.30      0.41       494
              1       0.61      0.88      0.72       634

    avg / total       0.63      0.62      0.58      1128

    Area under ROC curve for model: 0.5854704402355074



```python
logreg_probabilites_dataframe = pd.DataFrame(skl_logreg.predict_proba(X), columns = ["Probability over threshold", "Probability under threshold"])
model_vs_data_dataframe = pd.concat(objs = [logreg_probabilites_dataframe, master_dataframe[['salary over 90000']]],axis=1)
model_vs_data_dataframe.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Probability over threshold</th>
      <th>Probability under threshold</th>
      <th>salary over 90000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.499859</td>
      <td>0.500141</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.448430</td>
      <td>0.551570</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.435767</td>
      <td>0.564233</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.451176</td>
      <td>0.548824</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.346364</td>
      <td>0.653636</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
skl_logreg_model_coefficients = pd.DataFrame(data = skl_model.coef_, index = ['coefficient'], columns = data_dataframe.drop(labels = 'salary over 90000', axis = 1).columns).T
```

<br></br>
<br></br>
#<a name="Part-II"></a> Part II: Report

Since each company has different data science needs, we expected each job posting's requirements to vary by a large range. We therefore decided to focus our analysis on a few key pieces of information in each job posting, specifically:

* If the job required a PhD or a bachelor's degree
* If the job asked for experience with Python
* If the company was likely to be a start-up
* If the job specifically used "scientist", as a proxy for expectations of technical knowledge
* If the job description mentioned machine learning
* If the job function includes managing or directing responsibilities
* The mean number of years of experience required

Our goal is to predict which jobs will have a salary exceeding $90,000 based on these features. This value was chosen to represent the mean salary of our data, as is shown below. While we'd prefer to predict the salary more exactly, job salaries typically vary according to a number of factors (location, responsibilities, company policies, etc.) which means computing a salary range may be more useful.


```python
print("Ratio of jobs over salary threshold to under salary threshold: {}".format(round(data_dataframe['salary over 90000'].describe()['mean'],3)))
```

    Ratio of jobs over salary threshold to under salary threshold: 0.562



```python
sns.countplot(data = data_dataframe, x = 'salary over 90000')
plt.title("Number of Jobs Under and Over Salary Threshold")
sns.plt.show()
```


![png]({{ site.url }}/images/indeed-jobs-postings-images/output_36_0.png)


### Procedure: Scraping Job Listings From Indeed.com

By a rough estimation, only a fraction of jobs posts explicitly included more that one of these features. Rather than follow the hyperlink for each post and scanning each page, which would likely have formatting particular to each company, we reasoned that we could leverage Indeed.com's search to obtain information we could not readily access.

Each job posting on Indeed.com has a unique ID. (When capturing the data, we noticed approximately 20% had duplicated IDs. We determined by inspection that these were reposted jobs, and therefore reasonably removed these data points.) For this reason could we employ a strategy where we changed the search query and kept track of which jobs were among the results, perform multiple keyword searches, and merge the resulting data on the for each ID. For example, the following are the first few jobs that required a PhD and mentioned 3 years of experience:


```python
master_dataframe[(master_dataframe['PhD OR ph.d'] == 1) & (master_dataframe['years experience 3'] == 1)][['job title', 'PhD OR ph.d', 'years experience 3']].head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>job title</th>
      <th>PhD OR ph.d</th>
      <th>years experience 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Research scientist, muscle</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scientist, cancer cell dependencies, oncology ...</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Research scientist / 40 / day / bwh - neurology</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Phd research scientist- machine learning</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Research data scientist</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



This method of analysis is much simpler than the most obvious alternative strategy --- following each job posting on Indeed.com to its description on an external website and performing a keyword search --- but the method relied on Indeed.com's search algorithm, which introduced complications:

Indeed.com's support documentation stated that only about 20% of the job postings in a search result include salary information. The search results deliver this information if available; otherwise, the salaries are estimated from past job postings and user-provided data.

The webscraping provided a significant number of results that contained keywords like "data" and "science" but may be unrelated to data science. For example:


```python
master_dataframe[master_dataframe['Lab OR laboratory'] == 1]['job title'].head()
```




    2                         Cellular analytics scientist
    3                         Cellular analytics scientist
    4    Scientist, cancer cell dependencies, oncology ...
    5    Scientist, cancer cell dependencies, oncology ...
    6    Scientist, cancer cell dependencies, oncology ...
    Name: job title, dtype: object



An inspection of a sampling of the potentially unrelated jobs suggested that job postings that referred to a laboratory were more likely to focus on biology or related sciences. Similarly, job postings with titles that included "software engineer" were less likely to relate to data science. These keywords were used to filter these jobs from the data.

### Results and Analysis

<u>The results may not be current since they are drawn from live webpages.</u>

We analyzed postings for 1692 jobs. We removed 564 observations from the data, a 33.3% reduction, leaving 1128 observations. "Scientist" was by far the most common keyword, occurring in approximately 75% of observations, followed by PhD, "laboratory", and bachelor's degree (36%, 29% and 28%, respectively).


```python
print("Jobs analyzed: {}\nJob postings removed: {} ({}% change)\nRemaining postings: {}".format(master_dataframe.shape[0], master_dataframe.shape[0] - data_dataframe.shape[0], round(100 * (data_dataframe.shape[0] / master_dataframe.shape[0] - 1),1), data_dataframe.shape[0]))
print("\nKeywords\t\tFrequency\n{}".format(master_dataframe.drop('salary over 90000', axis = 1).describe().loc['mean'].sort_values(ascending = False)))
```

    Jobs analyzed: 1692
    Job postings removed: 564 (-33.3% change)
    Remaining postings: 1128

    Keywords		Frequency
    Scientist                0.791962
    PhD OR ph.d              0.372340
    Lab OR laboratory        0.336879
    Bachelors OR BS OR BA    0.217494
    Python                   0.211584
    Manager OR director      0.196809
    Machine learning         0.167849
    years experience 5       0.132388
    years experience 3       0.131797
    years experience 2       0.119385
    Startup OR start-up      0.091017
    years experience 1       0.085106
    years experience 4       0.059102
    Software engineer        0.048463
    years experience 7       0.043735
    years experience 6       0.033097
    years experience 0       0.007092
    Name: mean, dtype: float64


Of the keywords examined, Python, Scientist and Machine Learning were the strongest predictors of a job's salary.


```python
skl_logreg_model_coefficients.sort_values(by = 'coefficient', ascending = False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Python</th>
      <td>0.273112</td>
    </tr>
    <tr>
      <th>Scientist</th>
      <td>0.195354</td>
    </tr>
    <tr>
      <th>Machine learning</th>
      <td>0.188584</td>
    </tr>
    <tr>
      <th>Startup OR start-up</th>
      <td>0.068621</td>
    </tr>
    <tr>
      <th>Manager OR director</th>
      <td>0.062443</td>
    </tr>
    <tr>
      <th>PhD OR ph.d</th>
      <td>0.055231</td>
    </tr>
    <tr>
      <th>years experience 5</th>
      <td>0.053895</td>
    </tr>
    <tr>
      <th>years experience 7</th>
      <td>0.026132</td>
    </tr>
    <tr>
      <th>Bachelors OR BS OR BA</th>
      <td>0.018832</td>
    </tr>
    <tr>
      <th>years experience 3</th>
      <td>0.011098</td>
    </tr>
    <tr>
      <th>years experience 6</th>
      <td>0.006221</td>
    </tr>
    <tr>
      <th>years experience 0</th>
      <td>0.006133</td>
    </tr>
    <tr>
      <th>years experience 4</th>
      <td>0.000689</td>
    </tr>
    <tr>
      <th>years experience 2</th>
      <td>-0.013593</td>
    </tr>
    <tr>
      <th>years experience 1</th>
      <td>-0.022556</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10,6))
sns.barplot(data = skl_logreg_model_coefficients.sort_values(by = 'coefficient', ascending = False).T, orient='h')
plt.title("Influence of Features on Predicting Job Salary")
sns.plt.show()
```


![png]({{ site.url }}/images/indeed-jobs-postings-images/output_48_0.png)


## Conclusion

Python, Scientist, and Machine learning were the strongest predictors of salary. However this may be because these keywords are better correlated with data science roles; therefore, these keywords could be predicting a certain class of job from others which tend to have different corresponding salaries. (E.g., a data scientist is typically paid more than a data analyst.)

The model's precision was likely impinged from the variability in Indeed.com's salary estimates. If given more time, one could measure this variability by comparing job postings where the salary is explicitly stated to those with estimated salaries.
