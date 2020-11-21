# Summary

## Data
A large sum of data is made available by Citibike open-data project. including rides history starting from 2013.
Details such as ride's starting location, end location, trip duration, as well as basic information of the rider 
(e.g. member/non-member), etc., are readily available.

## Background
A couple of ideas jumped to mind when I first started looking at the data, including looking at how riding patterns 
change since Covid lock-down. An easy way to tackle this is by analyzing the volume of rides with different grouping. 
For example, one can look at how the number of rides on weekdays changes through time, 
whether the ratio of member/non-member shifts, whether the time/area that most rides occur is different. 
However, to challenge myself, I decided to pick one particular period (month of August in 2020) 
and look for patterns within that period. As business re-opened partially during the time, 
would there be any clustering of rides taken place? Can we classify the rides into different categories? 
And if that is the case, from a business perspective, can business targets the riders differently 
based on the purpose of the rides?

## EDA
Looking at the distribution of rides from different angles, such as by day/hour of rides, by area, by user type, 
one can see that some of the pre-Covid patterns still hold. 
Rides occur more often during commute hours on weekdays, with demand coming from residential 
areas in the morning and from office areas in the afternoon. 
Central park areas appear to be in heavier demand in the evening times.

## Feature selection
The feature I included are:
* Location: Normalized latitude/longitude of the start & end station. While Citibike covers both NYC and Jersey City, 
I decided to exclude Jersey City data to use location as a feature in a more meaningful way.
 
* Time: The day of the week and the hour of the day when the ride occurred. As both are cyclical features, 
I represented each of them using two features through sin/cos conversion.

* Supply/Demand of the station & area: Number of net bikes' check-out of the station/end station as well as 
in the start/end area. I included area breakdown to group stations with close proximity together, 
as in real life, they are used interchangeably in many cases. The area information (i.e. neighborhood) is not provided 
but can be derived by joining the station geography information with an area map.

* distance

## Dimension reduction visualization
To see if there is any obvious clustering of the data, I started with visualizing the data using dimension reduction. 
Unfortunately, there does not seem to be any clear clustering patterns when projecting my data on a 2D surface 
using PCA, T-SNE and Umap. 

## Clustering
As the clustering pattern is not evident, I decided to go with 'trial and error'. There are at least two grids 
that matter, specifically, the number of cluster, and the clustering technique.
For the first one, I tried clustering my data into 2/3/4 groups. Anything beyond that would be hard to interpret, 
in my opinion. To figure out the optimal clustering number, I relied on both scores 
(Silhouette Score, Calinski-Harabasz Score) and clustering visualization. 
As for the techniques, I started with k-means and tested GMM and Hierachy clustering using ward linkage. 
Out of curiosity, I also explored clustering with u-map features.

In summary, 3-cluster clustering seems to generate the most insights. 
Both K-means clustering and hierarchy clustering put emphasis on location and divide the rides into Brooklyn rides, 
midtown/downtown Manhattan rides, and uptown Manhattan/Queens rides.
GMM, in contrast, focuses more on the time and does a decent job separating commuting rides from 
'leisure' rides which occurred mostly in the late morning/early afternoon and in the evening. 
The story is not super clear-cut though, and more work can be done to understand the drivers of the clustering better.
On the other hand, supply/demand of the area does not seem to be a contributing feature: 
all clusters comprise mainly of rides that start from areas where bikes are in demand (net checkout>0),
to areas where bikes are docked (net checkout<0)


## Classification
Using the labels derived from the clustering, I was also able to predict the group of the ride fairly well 
(with 90% accuracy). The classification results show promises of a potentially useful way to look at this data-set.
   
## Deployment
For practice, I deployed some of the key charts onto a website using flask and host it via Heroku. 

More detail can be found at (https://github.com/cczhao9151/Citibike-flask).

Website can be found at (http://127.0.0.1:5001/)

# Conclusion and further work
The work shows promises of decoupling Citibike rides to smaller groups for better understanding of the rides and the riders.
More work can be done to explore/confirm the patterns, especially by testing it on other periods.
August 2020 is a somewhat unique period as NYC was still in partial lock down and human behaviors are 
still mixed, making the intention hard to interpret. It would be interesting to expand the analysis to
earlier periods prior to Covid, as well as to earlier this year when lock-down was in its full scale.
  