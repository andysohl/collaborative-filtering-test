# Databricks notebook source
# MAGIC %md
# MAGIC # Collaborative Filtering Practice
# MAGIC 
# MAGIC This notebook is to try using WCL data to build a recommender system.
# MAGIC 
# MAGIC There can be two ways to do this:
# MAGIC 
# MAGIC 1. Item-based: Takes similarities between the item's consumption histories
# MAGIC 2. User-based: Considers similarities between user consumption histories and item similarities
# MAGIC 
# MAGIC Since we don't have user ratings for each product/custom group/category, we have to come up with alternatives.
# MAGIC 
# MAGIC **Alternatives:**
# MAGIC 1. Transaction Count: How many times this product has been purchased?
# MAGIC 2. Sales Proportion: The proportion users have spent on each product/category should be a reasonable proxy to how they rate the products, we can multiply the proportion by 5 to get a score from 0 to 5, similar to actual ratings

# COMMAND ----------

import pandas as pd
from scipy.spatial.distance import cosine

# COMMAND ----------

# Try with custom groups first
data = pd.read_csv('/dbfs/mnt/terence/Wellcome/Mass/Behavioral/training_data/KPI_data/Transaction_Count.csv', index_col=None)
data

# COMMAND ----------

data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Item-based CF

# COMMAND ----------

# Start with a small sample first, n=1000
data_ib = data.copy()
data_ib = data_ib.sample(n=1000, random_state=123)
# Drop member id column, as this is item-based
data_ib = data_ib.reset_index()
data_ib = data_ib.drop(columns=['index', 'member_id'])
data_ib

# COMMAND ----------

# Create a placeholder dataframe listing item vs item
data_ibs = pd.DataFrame(index=data_ib.columns, columns=data_ib.columns)
data_ibs

# COMMAND ----------

# MAGIC %md
# MAGIC ### Similarity Measure
# MAGIC Using cosine similarity (-1<n<1, -1 means exact opposite, 1 means exact same, 0 means orthogonality)

# COMMAND ----------

for i in range(0,len(data_ibs.columns)) :
    # Loop through the columns for each column
    for j in range(0,len(data_ibs.columns)) :
      # Fill in placeholder with cosine similarities
      data_ibs.iloc[i,j] = 1-cosine(data_ib.iloc[:,i],data_ib.iloc[:,j])
      
data_ibs

# COMMAND ----------

# Look for each items' "neighbor" by sorting each column in descending order, and grabbing the top 3 / 10

top_n = 10
data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,top_n+1))
 
# Loop through our similarity dataframe and fill in neighbouring item names, skipping the first one (itself)
for i in range(0,len(data_ibs.columns)):
    data_neighbours.iloc[i,:top_n] = data_ibs.iloc[0:,i].sort_values(ascending=False)[1:top_n+1].index
    
data_neighbours

# COMMAND ----------

# MAGIC %md
# MAGIC ### User-based CF
# MAGIC The process for creating a User Based recommendation system is as follows:
# MAGIC 
# MAGIC 1. Have Item-Based similarity matrix
# MAGIC 2. Check which items the user has consumed
# MAGIC 3. For each item the user has consumed, get the top X neighbours
# MAGIC 4. Get the consumption record of the user for each neighbour.
# MAGIC 5. Compute similarity score
# MAGIC 6. Recommend the items with the highest score

# COMMAND ----------

# Helper function to get similarity scores between users
# Uses the sum of the product of 2 vectors (purchase history and item similarity), then divide by the sum of similarities in the respective vector

def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)

# COMMAND ----------

# Get a sample of 1000 users for user-based CF, this time we are leaving the member id in the table
data_sims1 = data.copy()
data_sims1 = data_sims1.sample(n=1000, random_state=123)
data_sims1 = data_sims1.reset_index(drop=True)
data_sims1

# COMMAND ----------

# Create a place holder matrix for similarities, and fill in the user name column
data_sims = pd.DataFrame(index=data_sims1.index,columns=data_sims1.columns)
data_sims.iloc[:,:1] = data_sims1.iloc[:,:1]
data_sims

# COMMAND ----------

# Loop through all rows, skip the user column, and fill with similarity scores
for i in range(0,len(data_sims.index)):
    for j in range(1,len(data_sims.columns)):
        user = data_sims1.index[i]
        product = data_sims1.columns[j]

        if data_sims1.iloc[i, j] >= 1:
            data_sims.iloc[i, j] = 0
        else:
            product_top_names = data_neighbours.loc[product][1:10]
            product_top_sims = data_ibs.loc[product].sort_values(ascending=False)[1:10]
            user_purchases = data_ib.loc[user,product_top_names]
            data_sims.iloc[i,j] = getScore(user_purchases.to_numpy(),product_top_sims.to_numpy())

# COMMAND ----------

# Similarity Score Matrix
data_sims

# COMMAND ----------

# Get the top products placeholder dataframe
data_recommend = pd.DataFrame(index=data_sims.index, columns=['Person','1','2','3','4','5','6'])
data_recommend.iloc[0:,0] = data_sims.iloc[:,0]
# data_recommend

# COMMAND ----------

# Instead of top product scores, we want to see names
for i in range(0,len(data_sims.index)):
    data_recommend.iloc[i,1:] = data_sims.iloc[i,:].sort_values(ascending=False).iloc[1:7,].index.transpose()

# COMMAND ----------

# See the top 3 recommendations for 20 users (Recommending products that user has never purchased)
data_recommend.iloc[:20, :4]

# COMMAND ----------

# MAGIC %md
# MAGIC # Next Steps
# MAGIC Use sales proportion as "rating" and we can then do model-based recommenders, i.e. predicting the "rating" sales proportion and output the top k recommendations based on the predicted "rating" (sales proportion) of each product

# COMMAND ----------

