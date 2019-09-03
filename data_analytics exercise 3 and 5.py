
# coding: utf-8

# In[84]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
os.chdir("C:/Users/16z245/Downloads/")


# In[85]:


def geo_mean(iterable):
    print(iterable)
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))


# In[86]:


data = pd.read_csv("C:/Users/16z245/Downloads/books.csv")


# In[87]:


data.drop(["isbn","isbn13"],axis=1,inplace=True)
data=data.iloc[:1000]


# In[88]:


nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns')


# In[89]:


data.head()


# In[90]:


fig=plt.figure(figsize=(10,15))
ax=fig.gca()
data.hist(ax=ax,bins=20)
plt.show()


# In[91]:


bx_plt_data=data.head(10)
bx_plt_data.boxplot(column='ratings_count')
plt.show()


# In[92]:


data.mean()


# In[93]:


import math

def geomean(xs):
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))

geomean(data.num_pages)


# In[94]:


data.plot()


# In[95]:


data.iloc[:10].plot.box()
plt.show()


# In[96]:


data.iloc[:10].plot.bar()
plt.show()


# In[97]:


freq = {} 
for item in data.language_code: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1
key=[x for x in freq.keys()]
key
value=[freq[x] for x in freq.keys()]
value
plt.pie(value,labels=key,autopct='%.1f%%',startangle=90)
plt.show()
plt.bar(key,value)
plt.show()


# In[98]:
plt.scatter(value, key)
plt.title('Average Rating Distribution')
plt.xlabel('Count')
plt.ylabel('Rating')
freq = {} 
#df['average_rating']=df['average_rating'].astype('float64')
for item in df['average_rating']: 
        if (item in freq): 
            freq[item] += 1
        else: 
            freq[item] = 1plt.show()


data.columns


# In[99]:


for i in data.text_reviews_count:
    freq={}
    for item in data[i]: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
    key=[x for x in freq.keys()]
    value=[freq[x] for x in freq.keys()]
    plt.pie(value,labels=key,autopct='%.1f%%',startangle=90)
    plt.plot()
    plt.show()
    


# In[100]:


gmean=stats.gmean(data.num_pages)
print("gmean is ",gmean)
hmean=stats.hmean(data.num_pages)
print("hmean is ",hmean)

