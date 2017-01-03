---
title: "Web Scraping practice: retrieving job IDs and keywords from Indeed.com"
header:
  overlay_color: "#333"
date: 2016-12-22
tags: Web scraping, Indeed.com
---

Here's a bit of code from a class project.

In about two months, everyone here will be searching through job boards to find the next steps in our careers. For this project we want to find a way to predict data science salaries in Boston.

We scanned Indeed.com and scraped information related to features such as job title, years of experience, company, and keywords like Python. Only a few listings had salary information or descriptions we could search through. Therefore we found most of our data by manipulating the search query, and trusting Indeed's backend processing to deliver relevant results.

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
```


```python
# String format: "keyword keyword"
def count_results(query_string = None):
    if query_string == None:
        return(print("No keyword entered."))

    query = "%20OR%20".join(query_string.split(" "))

    job_ids = pd.DataFrame()

    result_list = []

    # Find the number of results
    URL_for_count = "http://www.indeed.com/jobs?q=data+scientist+%28{}%29&l=Boston".format(query)
    soup_for_count = BeautifulSoup(urlopen(URL_for_count).read(), 'html.parser')

    results_number = soup_for_count.find("div", attrs = {"id": "searchCount"}).text
    number_of_results = int(results_number.split(sep = ' ')[-1].replace(',', ''))

    # Now loop through the pages. Viewing 100 results at a time means fewer page refreshes.
    i = int(number_of_results/100)
    for page_number in range(i + 1):
        URL_for_results = "http://www.indeed.com/jobs?q=data+scientist+%28{}%29&l=Boston&limit=100&start={}".format(query, str(100 * page_number))
        soup_for_results = BeautifulSoup(urlopen(URL_for_results).read(), 'html.parser')
        results = soup_for_results.find_all('div', attrs={'data-tn-component': 'organicJob'})

        # Extract the ID for each job listing
        for x in results:
            result_list.append([x.find('h2', attrs={"class": "jobtitle"})['id'], 1])

        # Add the job ID numbers
        job_ids = job_ids.append(result_list)

    # Rename job_ids's columns
    job_ids.columns = ['id', "{}".format(" OR ".join(query_string.split(" ")))]

    # Remove re-posted jobs
    job_ids.drop_duplicates(inplace = True)
    return (job_ids)
    #job_ids.to_csv(path_or_buf="id_and_{}.csv".format(query))
```


```python
count_results()
```

    No keyword entered.



```python
count_results("PHD ph.d")
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>PHD OR ph.d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>jl_033222419605ed3f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>jl_d5e17d142783f070</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>jl_32062be6a68c7531</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>jl_5cacf17ffc563847</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>jl_81117a845679ae80</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>jl_174d886f23f3d05e</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>jl_0bb820bd8ae6e87b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>jl_7896e61d3e45dfb4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>jl_62948e9c407ca034</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>jl_5f30454eae8a42bd</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>jl_65d62034685dd0ed</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>jl_04a882314da504d4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>jl_6c9d0349b46d0aae</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>jl_10a1355277be089a</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>jl_ef56b76c21a4dd91</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>jl_6059a43bec56d8f4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>jl_98bed206eb115e04</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>jl_44da2bd2b0b7e145</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>jl_9315b3a96ad20ac0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>jl_e50079f12b84b9b6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>jl_2370348e80d8c420</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>jl_765d5d6d1e3c30af</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>jl_770ee63428ff8a22</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>jl_0ea00cbc973dc761</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>jl_c888c568c29a71b7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>jl_3b58a2212af3ba86</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>jl_216284a90c10e500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>jl_151c68314992c211</td>
      <td>1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>jl_117c5543fad4be7e</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>jl_5dcae32cb98bfb9b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>jl_ee99d2f1d59e54e5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>489</th>
      <td>jl_c8f54d2003b4e465</td>
      <td>1</td>
    </tr>
    <tr>
      <th>490</th>
      <td>jl_689430243e6a7bea</td>
      <td>1</td>
    </tr>
    <tr>
      <th>491</th>
      <td>jl_5002db1bbda7ff3b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>492</th>
      <td>jl_100daa4903ccb872</td>
      <td>1</td>
    </tr>
    <tr>
      <th>493</th>
      <td>jl_ec512dddb179aae9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>494</th>
      <td>jl_8a3e91781bf16ddb</td>
      <td>1</td>
    </tr>
    <tr>
      <th>495</th>
      <td>jl_2471538c70b44aaf</td>
      <td>1</td>
    </tr>
    <tr>
      <th>496</th>
      <td>jl_506ead9021c7fd81</td>
      <td>1</td>
    </tr>
    <tr>
      <th>497</th>
      <td>jl_d4a605cf31c4d6fa</td>
      <td>1</td>
    </tr>
    <tr>
      <th>498</th>
      <td>jl_f91843d934a9d80f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>499</th>
      <td>jl_7ca7ec20e8a87bef</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>jl_9b476549089a6ff7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>jl_9ffa35a19d0fab79</td>
      <td>1</td>
    </tr>
    <tr>
      <th>505</th>
      <td>jl_bddee21cf5be8648</td>
      <td>1</td>
    </tr>
    <tr>
      <th>507</th>
      <td>jl_29c0202cfc89d948</td>
      <td>1</td>
    </tr>
    <tr>
      <th>509</th>
      <td>jl_56a526f9757fc61b</td>
      <td>1</td>
    </tr>
    <tr>
      <th>510</th>
      <td>jl_d6baff3e82b1db8f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>511</th>
      <td>jl_fc1f842786c79ede</td>
      <td>1</td>
    </tr>
    <tr>
      <th>512</th>
      <td>jl_6f1969f9bde9a93f</td>
      <td>1</td>
    </tr>
    <tr>
      <th>514</th>
      <td>jl_7ff9a220edc5f335</td>
      <td>1</td>
    </tr>
    <tr>
      <th>515</th>
      <td>jl_1960eac165fffe54</td>
      <td>1</td>
    </tr>
    <tr>
      <th>516</th>
      <td>jl_e82d2fa409c92914</td>
      <td>1</td>
    </tr>
    <tr>
      <th>517</th>
      <td>jl_849a440bd44fd715</td>
      <td>1</td>
    </tr>
    <tr>
      <th>518</th>
      <td>jl_1130793f98655196</td>
      <td>1</td>
    </tr>
    <tr>
      <th>519</th>
      <td>jl_a742c28826776c08</td>
      <td>1</td>
    </tr>
    <tr>
      <th>520</th>
      <td>jl_7baab1e1b0e8e234</td>
      <td>1</td>
    </tr>
    <tr>
      <th>521</th>
      <td>jl_7e2dba1124b29dcf</td>
      <td>1</td>
    </tr>
    <tr>
      <th>522</th>
      <td>jl_57d36d880dc90113</td>
      <td>1</td>
    </tr>
    <tr>
      <th>523</th>
      <td>jl_85017d6b3a30bdd3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>501 rows × 2 columns</p>
</div>




```python

```
