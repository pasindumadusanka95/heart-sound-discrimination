#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[5]:


app = Flask(__name__)


# In[6]:


@app.route('/home')
def running():
    return 'Falsk running'


# In[ ]:




