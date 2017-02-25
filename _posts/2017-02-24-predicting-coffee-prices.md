---
title: "Predicting Coffee Prices Using Random Forest Regression"
header:
  overlay_color: "#333"
date: 2017-02-24
---

Temperature and production data were combined and used to model a benchmark indicator for global coffee prices. A random forest regressor was optimized and used to model the data with a 6-month lag between data and predictions. The model is strongly correlated with results, although it appears to have a strong bias problem. Predictions were made for early 2017, which have thus far proved directionally accurate.

### Introduction

Coffee prices are volatile and difficult to predict. (This is especially true when compared to coffee futures prices, which can be strongly affected by irrational investor speculation.) The USDA's Foreign Agricultural Service issues semi-annual reports on coffee production; among the influences in its reports are previous production volumes, droughts, abnormally warm temperatures, frosts, pest incidences, diseases that affect coffee plants, and labor shortages. However these influences are quantifiable and, to a limited degree, predictable.

Other analyses have focused on production volume to predict global prices. In 1955, Henry Hopp and Richard J. Foote found that global prices between 1882 and 1949 were principally associated with coffee held in storehouses and Brazilian exports. (Hopp, H., and Foote, R. <a href="http://econpapers.repec.org/RePEc:oup:ajagec:v:37:y:1955:i:3:p:429-438.">A Statistical Analysis of Factors That Affect Prices of Coffee</a>, American Journal of Agricultural Economics, 1955.) It is likely this relationship still holds, however the market has evolved considerably since then. This analysis will consider past production trends, ending stocks (the amount of coffee held in storehouses before exporting), and frost incidences, for the main coffee producing countries.


### Data

The International Coffee Organization is an intergovernmental body established by the UN in 1963 to improve economic conditions for farmers, strengthen global coffee production, and enable sustainable sector growth. As of February 2017, its 73 member countries represent 98% of global coffee production.

The ICO composite indicator price is a benchmark for global green coffee prices. (See figure 1.) It is a composite of prices paid for specific categories of coffee at US and EU ports. (The US and EU are the primary coffee importers.) Beginning in 2001, each category's prices are weighted to represent its relative share of international coffee trade.  While these weights are adjusted biennially, these values are relatively stable, and the effects of the adjustments could be reasonably ignored. (See figure 2.)

The nature of the coffee market changed in 1989, which created a serious complication for analysis. Under the agreements established in 1963, coffee producing countries were regulated by quotas (renewed every five years) which stabilized prices. However the ICO failed agree to new quotas in 1989, causing the quota system to break down, and the pricing system to convert into a free market. This effectively limited the available target data to the period from 1990 to present, and made this analysis especially susceptible to overfitting.

A large and continual complication came from the ICO grouping countries into traditional categories according to the main coffee species produced, and the primary method of processing. (See the list below.) In all the analysis, care was needed to ensure each country's data was placed in the appropriate grouping.



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_39_0.png)


    Fig. 1: The ICO composite indicator price over time. Notable events include a devastating frost in Brazil in 1975, 1984 and 1994; the collapse of the quota system in 1989 and the subsequent drop in price; demand exceeding supply in 2011, paired with poor production caused by high rainfall in 2010.



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_39_2.png)


    Fig. 2: The ICO composite indicator prices modeled by the producing categories versus actual indicator prices. The strongly linear relationship means that the category weights used to calculate the composite have predictable effects, and can be ignored for this analysis.


    Brazilian Naturals
    ['Brazil', 'Ethiopia', 'Paraguay', 'Philippines', 'Thailand', 'Timor-Leste', 'Vietnam', 'Yemen']

    Colombian Milds
    ['Colombia', 'Kenya']

    Other Milds
    ['Bolivia', 'Burundi', 'Cameroon', 'Costa Rica', 'Cuba', 'Dem. Rep. of Congo', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Haiti', 'Honduras', 'India', 'Indonesia', 'Jamaica', 'Madagascar', 'Malawi', 'Mexico', 'Nepal', 'Nicaragua', 'Panama', 'Papua New Guinea', 'Peru', 'Rwanda', 'Uganda', 'Venezuela', 'Zambia', 'Zimbabwe']

    Robustas
    ['Angola', 'Benin', 'Brazil', 'Burundi', 'Cameroon', 'Central African Republic', 'Congo Rep.', 'Côte d’Ivoire', 'Dem. Rep. of Congo', 'Dominican Republic', 'Ecuador', 'Equatorial Guinea', 'Gabon', 'Ghana', 'Guatemala', 'Guinea', 'Guyana', 'India', 'Indonesia', 'Lao', 'Liberia', 'Madagascar', 'Mexico', 'Nigeria', 'Papua New Guinea', 'People’s Dem. Rep.', 'Philippines', 'Sierra Leone', 'Sri Lanka', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Uganda', 'Vietnam']



Composite indicator prices were retrieved from the ICO using `pandas` and standard data importing practices.

Production data was obtained from the USDA's FAS, which includes annual country-specific arabica and robusta productions as well as ending stocks. These can be seen in figure 3 below.

Temperature data was obtained from Berkeley Earth, which is an independent organization that combines and cross-validates global temperature data from 16 datasets. Coffee grows in certain preferred conditions of temperature, rainfall, and elevation, which generally occur in particular ranges of latitude and elevation. This information was used to select data from weather stations located in ideal coffee-growing regions. (See figure 4 below and the accompanying map.) There is no apparent method for quantifying frost risk; for this analysis it was approximated by $\frac{1}{T^2}$, where $T$ is the minimum monthly temperature in degrees Celsius.




![png]({{ site.url }}/images/predicting-coffee-prices-images/output_42_0.png)


    Fig. 3: Annual production is aggregated into ICO categories. Coffee plant produce berries every other year; this biennial effect is most pronounced in Brazil, and therefore in the Brazilian Naturals category.



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_43_0.png)


    Fig. 4: Weather stations in the Berkeley Earth dataset that are located in ideal coffee-growing conditions. Compare to the following map from Climate.gov for general comparison.





![jpeg]({{ site.url }}/images/predicting-coffee-prices-images/output_43_2.jpeg)



### Procedure

All data was cleaned using `pandas` and standard practices. There were few notable conversions or exclusions of data:
* Temperature data was filtered, as described above.
* The Berkeley Earth temperature data was indexed by decimal fractions. For example: $$\text{January } 25, 2005 = 2005 + (25 - 0.5)/365 =2005.067$$ These values were converted to `pandas`-recognizable datetime formats.
* The FAS production data was backfilled to convert its annual data into monthly data.

The resulting dataset had 324 observations and approximately 87 features. The number of features was reduced 15 after accounting for the fact that of the 73 member countries in the ICO, a small group accounts for the majority of global production, as is shown in figure 5.

Principal component analysis was attempted to further reduce the number of features, and decrease the chances of overfitting to the data, however the standardized variations were not easily separable. (See figure 6 below.)


![png]({{ site.url }}/images/predicting-coffee-prices-images/output_46_0.png)


    Fig. 5: Cumulative shares of coffee production by country. As can be seen, less than a dozen countries account for more than 90% of global production.



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_46_2.png)


    Fig. 6: The cumulative sums of explained variances. The gradual increase of eigenvalues means that no eigenvectors were particularly good at separating the data.



The final dataset included information related to Brazil, Colombia, Cote d'Ivoire, Ethiopia, Guatemala, Mexico, Indonesia, Uganda, and Vietnam. It was used to model the ICO composite indicator using a random forest regressor which was optimized with a grid search and cross validation.

### Results and Discussion

The results of modeling are shown in figure 7 below. The model was used to to forecast ICO composite indicator prices for early 2017. (See figure 8.)


![png]({{ site.url }}/images/predicting-coffee-prices-images/output_50_0.png)

    Random forest model score: 0.9569884965546481



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_50_2.png)

    Random forest model score: 0.9569884965546481
    Fig. 7: Model (red) compared to actual indicator prices (blue). The score is the cross validation score.



![png]({{ site.url }}/images/predicting-coffee-prices-images/output_50_4.png)


![png]({{ site.url }}/images/predicting-coffee-prices-images/output_50_5.png)

    Fig. 8: Predicted indicator prices for early 2017. The information for the ICO composite indicator prices in the bottom plot was manually retrieved from the ICO's website.


The model appears to perform exceptionally well, although this is not likely the case. The bias-variance tradeoff is a major concern when there are relatively few observations per feature, especially in this analysis, where roughly half the data was constructed using backfilled annual data. While measures were taken to mitigate this concern, such as reducing the number of features, using cross validation, and creating forests with large number of trees, the results appear to indicate it is still a major problem.

Another possible interpretation is that this model simply reflects supply and demand economics. Since local consumption (and the percentage diverted to ending stocks) is relatively small, each country's total production correlates very strongly to its supply to the global market.

### Next Steps

More data is needed to relieve bias-variance concerns. Future analysis could use more sophisticated statistical techniques that specialize in working with small data samples, however a better approach may be to increase the number of observations. The ICO provides subscription-based access to internal data, such as monthly production of its member countries, however it was too expensive for this analysis. (This data may not be completely reliable, since the data are neither validated or balanced.)

With further research, principal component analysis may be successfully applied to the data used here to address dimensionality concerns.

After increasing the number of observations, there are additional features that would be of interest:
* Precipitation, to indicate drought or flooding.
* Spread of diseases like coffee rust, or pests like the coffee cherry borer.
* The financial conditions for the farmers, as crops may be unharvested or abandoned if it is not an economically feasible enterprise for the growers. This was the case in Colombia in 2008.

Lastly, before any business applications of this (or similar) analysis is realized, prediction intervals must be applied to the forecasts. Methods for calculating these intervals using python are not easily found; indeed, this may be analysis better suited to other programming languages like `R`. A computationally-intensive python approach is possible, however it involves calculating the model error obtained when fitting models to data increasingly in the past of a specific point of interest.

If given more time, I would strive to make these improvements to this analysis, in this order.
