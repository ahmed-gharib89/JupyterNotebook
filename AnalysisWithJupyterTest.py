#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


sales = pd.read_csv(r'C:\Users\Ahmed Gharib\Desktop\jupyter\sales.csv')
sales.head()


# In[5]:


sales.shape


# In[6]:


sales.sample(10)


# In[7]:


sales['Brand'].unique()


# In[8]:


sales.Brand.value_counts()


# In[9]:


plt.figure(figsize= (12, 8))
sales['Brand'].value_counts().plot(kind='bar')
plt.title('Net Sales per item by quantity', fontsize=15)

plt.xlabel('Item', fontsize=12)
plt.ylabel('Quantity', fontsize=12)
plt.show()


# In[11]:


plt.figure(figsize= (12, 8))
sales[['Quantity']].boxplot()


# In[12]:


sales.boxplot(by = 'Brand', column=['Quantity'], grid=False, figsize=(12, 8))
plt.show()


# In[13]:


plt.figure(figsize=(12, 8))

sns.swarmplot(x='Brand', y='Quantity', data=sales)

plt.title('Sales per item by quantity', fontsize=15)

plt.xlabel('Item', fontsize=12)
plt.ylabel('Quantity Sold', fontsize=12)
plt.show()


# In[14]:


from sklearn.preprocessing import LabelEncoder


# In[15]:


label_encoder = LabelEncoder()


# In[45]:


sales['Brand']


# In[46]:


sales.head()


# In[19]:


dummy_sales = pd.get_dummies(sales)
dummy_sales.head()


# In[21]:


plt.figure(figsize=(12, 8))

sns.swarmplot(x='Brand', y='Amount', data=sales)

plt.title('Sales', fontsize=15)

plt.xlabel('Item', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.show()


# In[22]:


dummy_sales.shape


# In[23]:


get_ipython().system('pip install ipywidgets')


# In[24]:


get_ipython().system('jupyter nbextension enable --py widgetsnbextension --sys-prefix')


# In[25]:


from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


# In[26]:


w = widgets.IntSlider(value=10,
                     min=5,
                     max=20,
                     step=1,
                     description='Range Slider',
                     containues_update=False,
                     orientation='horizontal')


# In[27]:


w


# In[29]:


w.value


# In[30]:


type(w)


# In[43]:


r = widgets.IntRangeSlider(value=[10, 1000],
                     min=0,
                     max=10000,
                     step=1,
                     description='Amount Range',
                     orientation='horizontal')


# In[44]:


r


# In[48]:


r.value


# In[50]:


p = widgets.IntProgress(value=70,
                     min=0,
                     max=100,
                     step=1,
                     description='Loading',
                     bar_style='success',
                     orientation='horizontal')


# In[51]:


p


# In[54]:


import time

for i in range(0, 110, 10):
    p.value = i
    
    time.sleep(1)


# In[55]:


t = widgets.BoundedIntText(value=5,
                     min=0,
                     max=100,
                     step=1,
                     description='Enter a Number',
                     disabled=False)
t


# In[56]:


t.value


# In[57]:


widgets.Checkbox(value=False,
                description='Check Me')


# In[58]:


dd = widgets.Dropdown(options=['None', '0', '1', '2', '3', '4'],
                     value='None',
                     description='Number',
                     disabled=False)


# In[59]:


dd


# In[61]:


rb = widgets.RadioButtons(options=['Samsuns', 'Mi', 'Tecno'],
                         description='Brand')


# In[62]:


rb


# In[63]:


button = widgets.Button(description='Click Here',
                       button_style='success',
                       tooltip='Click ya prince',
                       icon='Check')


# In[64]:


def button_click(button):
    print('You clicked 3ash yasta 3ash')
    print(button.description)


# In[65]:


button.on_click(button_click)


# In[66]:


button


# In[70]:


play = widgets.Play(value=50,
                   min=0,
                   max=100,
                   step=1,
                   description='Press Play')

slider = widgets.IntSlider()

p = widgets.IntProgress()

widgets.jslink((play, 'value'), (p, 'value'))
widgets.jslink((play, 'value'), (slider, 'value'))

widgets.HBox([play, slider, p])


# In[73]:


def f(m, b):
    plt.figure()
    
    x = np.linspace(-10, 10, num=1000)
    
    plt.plot(x, m * x + b)
    
    plt.ylim(-5, 5)
    plt.show()
    
interactive_plot = interactive(f, m=(-2.0, 2.0), b=(-3, 3, 0.5))
    


# In[74]:


interactive_plot


# In[75]:


output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot


# In[76]:


from IPython.display import display

text = widgets.Text()
display(text)

def make_upper_case(input_text):
    text.value = input_text.value.upper()
    
    print (text.value)
    
text.on_submit(make_upper_case)


# In[77]:


sales = pd.read_csv('sales.csv')


# In[78]:


sales.head()


# In[103]:


@interact
def show_brand_data(column = 'Quantity', x=1000, column2 = 'Brand', name=['Mi', 'Samsung', 'Tecno']):
    return sales.loc[(sales[column2] == name) & (sales[column] > x)]


# In[104]:


from IPython.display import Image
import os

fdir = '256/'

@interact
def show_images(file= os.listdir(fdir)):
    display(Image(fdir + file))
    


# In[ ]:




