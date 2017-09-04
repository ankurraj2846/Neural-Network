
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import tensorflow as tf
## Importing the data_sample file and assigning each sheet to a new dataframe

xls = pd.ExcelFile('/Users/AR/Desktop/PeopleStrong/dataset_sample.xls')
df1 = xls.parse('Job_Pool')
df2 = xls.parse('CV_Pool')
df3 = xls.parse('Job_CV_Mapping')

## This will generate a 12x16 matrix mapping relevant CVs to each Job        
## Below I formed a vector with of unique skills required for a job and unique skills a candidate possess.
## This will help us to find out the matching skills parameter 

rskill = df1['Required Skills']
cskill = df2['Skills']

result_cst=[]
result_rst=[]

for i in range(len(cskill)):
    cskill_t = cskill[i].strip().split(",")
    result_cst.append(cskill_t) 
    
for i in range(len(rskill)):
    rskill_t = rskill[i].strip().split(",")
    result_rst.append(rskill_t) 

## Here we are forming a matrix with all the 36 unique skills as their columns for both Job and CV (M1 and M2) 

skills_j = set(x for l in result_rst for x in l)
skills_j = list(skills_j)
skills_c = set(x for l in result_cst for x in l)
skills_c = list(skills_c)
m1 = pd.DataFrame()
m2 = pd.DataFrame()
for i in range(len(result_rst)):
    for j in range(len(result_rst[i])):
        for k in range(len(skills_j)):
            if result_rst[i][j] == skills_j[k]:
                  m1.loc[i,k] = 1
            else: m1.loc[i,k] = 0

for i in range(len(result_cst)):
    for j in range(len(result_cst[i])):
        for k in range(len(skills_c)):
            if result_cst[i][j] == skills_c[k]:
                  m2.loc[i,k] = 1
            else: m2.loc[i,k] = 0

## Below is the neural network code that I have built so far

def build_model(nn_hdim=1, num_passes=5000):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    
    ## Xavier Initialization; We can go with np.random.randn(nn_input_dim, nn_hdim) alone but this function below helps
    ## us to maintain variance of 1. If number of inputs are higher then it assign less weight across all input neurons
    
    W1 = np.random.randn(nn_input_dim,1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1,1))
    
    # This is what we return at the end
    model = {}
     
    # Gradient descent. 
    for i in xrange(num_passes):        
 
        # Forward propagation
        
        z1 = m1.dot(W1) + b1   
        a1 = np.tanh(z1)       ## We can use other activation functions like ReLU or Sigmoid  
        z2 = m2.dot(W1) + b1   ## I didn't play around with it much because of my blockage in backpropagation
        a1 = np.tanh(z2)                       
           
 
        # Backpropagation ( Incomplete )
 
        # Gradient descent parameter update
        W1 += -e * dW1   ## e is the learning rate, commonly used value is 0.01
        b1 += -e * db1   ## dW1 and db1 are calculated using backpropagation 
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1}
     
    return model


# In[111]:

#df4 = pd.DataFrame(index = range(len(df1)), columns=range(len(df2)))
#for i in range(len(df3)):
#    k = df3['Relevent_CV_Ids'][i].strip().split(",")
 #   k = temp
  #  for j in range(len(temp)):
   #     c = int(temp[j])
    #    df4[i][c] = 1 


# In[3]:

from IPython.display import display
skills_c = set(x for l in result_cst for x in l)
skills_c = list(skills_c)
m2 = pd.DataFrame(index = range(len(df2)), columns=range(len(skills_c)))

for i in range(len(result_cst)):
    for j in range(len(result_cst[i])):
        for k in range(len(skills_c)):
            if result_cst[i][j] == skills_c[k]:
                  m2.loc[i,k] = 1
            


# In[4]:

m2.columns = skills_c
#m2.fillna(0)
#m2.to_csv('/Users/AR/Desktop/PeopleStrong/m2.csv')


# In[5]:


m2['Designation']= df2['Designation']
m2['Industry']= df2['Industry']
m2['Institute']= df2['Institute']
m2['Degree']= df2['Degree']
m2 = m2.fillna(0)
m = m2.drop(['Designation','Industry','Institute','Degree'],axis=1)


# In[6]:

m


# In[2]:

import tensorflow as tf
# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 12X29 Matrix with 12 CV_ID and 29 Columns of unique skills with one hot encoding
x = tf.placeholder(tf.float32, [None, 29])
# now declare the output data placeholder - 16 Job_ID
y = tf.placeholder(tf.float32, [None, 12])


# In[3]:

W1 = tf.Variable(tf.random_normal([29, 20]), name='W1')
b1 = tf.Variable(tf.random_normal([20]), name='b1')
W2 = tf.Variable(tf.random_normal([20, 12]), name='W2')
b2 = tf.Variable(tf.random_normal([16]), name='b2')


# In[4]:

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.tanh(hidden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()



# In[7]:

df3


# In[ ]:



