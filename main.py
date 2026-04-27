""" Module 3: Clustering
Spring 2026
Patrick Johnson"""

import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist

#Load data
df = pd.read_csv("warranty_claims_2019.csv")

df.head()


#Initial analysis of various features
print("Highest cost return: ", max(df["cost"]))
print("Lowest cost return: ", min(df["cost"]))
print("Percent of items repaired: ", df[df['replace'] == 1].count()/df['repair'].count())
print("Number of returns total: ",df['warehouse_id'].count())
print("Number of returns in warehouse 1: ",df[df['warehouse_id']==1].count())
print("Number of returns in warehouse 2: ",df[df['warehouse_id']==2].count())
print("Number of returns in warehouse 3: ",df[df['warehouse_id']==3].count())

print("Earliest return: ",min(df["claim_date"]))
print("Latest return: ",max(df["claim_date"]))

df = df[['claim_date','comp_id','cost','warehouse_id','repair','comp_cat']]

#Transform the chosen values to the appropriate scale (0-1) and manage categorical data
# using pandas get_dummies function

df_cat = df[['comp_cat','warehouse_id','repair']]
df_cat = pd.get_dummies(df_cat,drop_first=True)

#Convert the date variable to a continuous variable representing number of days from the first value
df["claim_date"] = pd.to_datetime(df["claim_date"])
df["claim_date"] = (df["claim_date"] - min(df["claim_date"])).dt.days

df_cat['claim_date'] = df['claim_date']
df_cat['cost'] = df['cost']

#Normalize all the variables to the correct scale
for column in df_cat.columns:
    df_cat[column] = pd.to_numeric(df_cat[column])

scaler = MinMaxScaler(copy=False)
df_cat = scaler.fit_transform(df_cat)

distance_matrix = pdist(df_cat,metric='hamming')
Z = linkage(distance_matrix,method='complete')

plt.figure(figsize=(8,4))
dendrogram(Z, labels=df.index.to_list())
plt.title("Heirarchical clustering dendrogram")
plt.xlabel("Data point index")
plt.ylabel("Hamming distance")
plt.show()

df_cat = pd.DataFrame(df_cat)

clusters = fcluster(Z,t=0.45,criterion="distance")
print("cluster assignments: ",clusters)
score = silhouette_score(df_cat,clusters,metric="hamming")
print("Silhouette score: ",round(score,3))

df = df.assign(cluster=clusters)

clus_1 = df[df['cluster'] == 1]
clus_2 = df[df['cluster'] == 2]
clus_3 = df[df['cluster'] == 3]
clus_4 = df[df['cluster'] == 4]
clus_5 = df[df['cluster'] == 5]
clus_6 = df[df['cluster'] == 6]
clus_7 = df[df['cluster'] == 7]
clus_8 = df[df['cluster'] == 8]

plt.scatter(clus_1.claim_date,clus_1.cost,color="Red")
plt.scatter(clus_2.claim_date,clus_2.cost,color="Blue")
plt.scatter(clus_3.claim_date,clus_3.cost,color="Green")
plt.scatter(clus_4.claim_date,clus_4.cost,color="Black")
plt.scatter(clus_5.claim_date,clus_5.cost,color="Orange")
plt.scatter(clus_6.claim_date,clus_6.cost,color="Purple")
plt.scatter(clus_7.claim_date,clus_7.cost,color="Brown")
plt.scatter(clus_7.claim_date,clus_7.cost,color="Olive")


plt.xlabel('Claim date')

print(df.groupby('cluster').size())
print(clus_1.groupby('repair').size())
print(clus_2.groupby('repair').size())
print(clus_3.groupby('repair').size())
print(clus_4.groupby('repair').size())
print(clus_5.groupby('repair').size())
print(clus_6.groupby('repair').size())
print(clus_7.groupby('repair').size())

print(df.groupby('cluster').size())
print(clus_1.groupby('warehouse_id').size())
print(clus_2.groupby('warehouse_id').size())
print(clus_3.groupby('warehouse_id').size())
print(clus_4.groupby('warehouse_id').size())
print(clus_5.groupby('warehouse_id').size())
print(clus_6.groupby('warehouse_id').size())
print(clus_7.groupby('warehouse_id').size())
print(clus_8.groupby('warehouse_id').size())

print(clus_1.groupby('comp_id').size())
print(clus_2.groupby('comp_id').size())
print(clus_3.groupby('comp_id').size())
print(clus_4.groupby('comp_id').size())
print(clus_5.groupby('comp_id').size())
print(clus_6.groupby('comp_id').size())
print(clus_7.groupby('comp_id').size())
print(clus_8.groupby('comp_id').size())


"""I chose to use heirarchical K-Means clustering due to the number of non-numerical values such as the component category and the repair classification - a standard K-Means would return a distance of 1 between an element which was repaired and one which was replaced, which would be equivalent in distance to the gap between the highest and lowest price. This is unusable, so I turned to heirarchical clustering.
The results of this clustering was eight primary clusters with a silhouette score of 0.651, which is quite good. Examining the clusters,cost clearly factors into its assessment: the highest-value returns make up at least one cluster. There is another cluster centered around $600 returns, which stays constant across the year. Overall, claim date does not appear to significantly classify elements: every cluster spans across the claim date range.
The wheels and forks appear to be the highest cost items, and the algorithm clearly sorts by those categories. The specific warehouse is not used for classification - there are some variations in how many items are associated with which warehouse, such as how cluster 2 (brakes and cassettes) has 46 warranty claims from warehouse 3 compared to 22 from warehouse 1 and 38 from warehouse 2.
When I grouped by the component ID, I found that each category's returns were dominated by one or two specific items. When I referenced these items, I found that four frequently-returned items came from the same supplier, fittingly named 'Spam'. In particular, item number 207 was returned 88 times compared to a max of 30 for the next most-returned item."""

"""To summarize my findings, the returns have been spaced across rather evenly across the year and price points.
There are 18 problematic items that make up all the returns, 4 of which are from the same supplier - "Spam". Shima and Fox have two problematic items each, but the rest come from disparate suppliers."""
