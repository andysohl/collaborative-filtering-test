# Databricks notebook source
pip install mlxtend

# COMMAND ----------

# MAGIC %run /Users/trncetou@dairy-farm.com.hk/helper_cls

# COMMAND ----------

# MAGIC %run /Users/trncetou@dairy-farm.com.hk/helper_p_cls

# COMMAND ----------

import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
#association rules package
from mlxtend.frequent_patterns import apriori, association_rules

# COMMAND ----------

dbutils.fs.unmount("/mnt/terence/Mannings")

# COMMAND ----------

#unmount blob storage with same name
#dbutils.fs.unmount("/mnt/")
#hku
dbutils.fs.mount(
 source = "wasbs://04-publish@gpdl01pdseadfdl02.blob.core.windows.net/terence/Mannings",
 mount_point = "/mnt/terence/Mannings",
 extra_configs = {"fs.azure.account.key.gpdl01pdseadfdl02.blob.core.windows.net":dbutils.secrets.get(scope = "DATALAKE-PD", key = "azblob-pd02-key")}
)

# COMMAND ----------

# connect with the database
password=dbutils.secrets.get(scope = "DATALAKE-PD", key = "synapse-pd01-dfsg-datafactory-pwd")
url = "jdbc:sqlserver://sql-gpdl01-pd-sea-df-pd-sql01.database.windows.net:1433;database=sqldwh-gpdl01-pd-sea-dl01-sqldwh01;user=dfsg_datafactory@sql-gpdl01-pd-sea-df-pd-sql01;password="+password+";encrypt=true;trustServerCertificate=true;"

# COMMAND ----------

sql = """
select
	sales.transaction_external_reference,
	subcategory_name,
    case when count(*) > 0 then 1 else 0 end as total_units
from [PUBLISH].[ANFIELD_LOY_SALES_METRICS] sales
INNER JOIN [ANALYSE].[ANFIELD_TX_TRANSACTION_PRODUCT_INFO] PRODUCT
ON SALES.TRANSACTION_EXTERNAL_REFERENCE = PRODUCT.APP_TRANSACTION_ID
INNER JOIN (SELECT 'MNHK-' + [product_id] as product_id, subcategory_name from [ANALYSE].[HKMN_MD_ITEM] where category_name != 'GIFTS-GEN/REDEEM') MN_ITEM
ON PRODUCT.PRODUCT_ITEM_EXTERNAL_REFERENCE = MN_ITEM.PRODUCT_ID
WHERE [TRANSACTION_RETAIL_VALUE] <> 0
	AND TIME_PERIOD >= '2021-08-01' AND TIME_PERIOD <= '2022-01-31'
	AND [SPONSOR_BUSINESS_UNIT] = 'MNHK'
group by sales.transaction_external_reference,category_name
"""

# COMMAND ----------

df_spark = (spark.read
              .format("jdbc")
              .option("url", url)
              .option("query", sql)
              .load()
                )
df = df_spark.toPandas()

# COMMAND ----------

df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

df['transaction_external_reference'] = df['transaction_external_reference'].astype('str')

# COMMAND ----------

basket = (df
          .groupby(['transaction_external_reference', 'category_name'])['total_units']
          .sum().unstack().reset_index().fillna(0)
          .set_index('transaction_external_reference'))

# COMMAND ----------

basket.head()

# COMMAND ----------

basket['num_items'] = basket.sum(axis=1)
basket

# COMMAND ----------

big_basket = basket[basket['num_items'] >= 2]
big_basket

# COMMAND ----------

#set minimum support score
big_basket = big_basket.drop(columns=['num_items'])
frequent_itemsets = apriori(big_basket, min_support=0.01, use_colnames=True)
frequent_itemsets

# COMMAND ----------

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# COMMAND ----------

frequent_itemsets.sort_values(by="support", ascending=False)

# COMMAND ----------

rules

# COMMAND ----------

frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets[frequent_itemsets['length'] > 1].sort_values(by="support", ascending=False)

# COMMAND ----------

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules

# COMMAND ----------

