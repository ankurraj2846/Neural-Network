

```python
import numpy as np
import pandas as pd

## Here I am defining the activation function for the hidden layer, We can also use tanh(x) or ReLU (max(0,x)).

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Softmax Function
def softmax(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))

## Importing the data_sample file and assigning each sheet to a new dataframe
xls = pd.ExcelFile('/Users/AR/Desktop/PeopleStrong/dataset_sample.xls')
df1 = xls.parse('Job_Pool')
df2 = xls.parse('CV_Pool')
df3 = xls.parse('Job_CV_Mapping')

df4 = pd.DataFrame(index = range(len(df1)), columns=range(len(df2)))

## This will generate a 12x16 matrix mapping relevant CVs to each Job
for i in range(len(df3)):
    k = df3['Relevent_CV_Ids'][i].strip().split(",")
    temp = k
    for j in range(len(temp)):
        c = int(temp[j])
        df4[i][c] = 1 
        
## Below I formed a matrix with 
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
 
match= pd.DataFrame()
for i in range(len(result_cst)):
        for j in range(len(result_rst)):
            c=set(result_cst[i]).intersection(result_rst[j])
            match.loc[i,j]= len(c)/len(result_rst[j])

skills = set(x for l in result_rst for x in l)
skills = list(skills)
type(skills[1])

data = pd.DataFrame()

result_cst

for i in range(len(result_cst)):
    for j in range(len(result_cst[i])):
        for k in range(len(skills)):
            if result_cst[i][j] == skills[k]:
                  data.loc[i,k] = 1
            else: data.loc[i,k] = 0

np.random.randn(16, 3)

num_examples = len(data) # training set size
nn_input_dim = data.shape[1] # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (I chose commonly used values)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


## Function to calculate Loss Function
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Forward propagation to calculate our predictions
    z1 = data.dot(W1) + b1   ## data is 16x37 and W is 37x1 so z1 is 16x1. b is 16x1
    a1 = np.tanh(z1)         ## 
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
##
## axis=1 means 'along the column', (y in the above image)
## axis=2 means 'along the depth', (z in the above image)
## axis=0 means 'along the row', (x in the above image)

    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    
    return (1./num_examples) * data_loss

def build_model(nn_hdim=1, num_passes=5000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    
    ## Xavier Initialization; We can go with np.random.randn(nn_input_dim, nn_hdim) alone but this function below helps
    ## us to maintain variance of 1. If inputs are higher then it assign less weight across all input neurons
    
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
 
    # This is what we return at the end
    model = {}
     
    # Gradient descent. 
    for i in xrange(num_passes):        ##range() will create a list of values from start to end (0 .. 20 in your example).
                                        ##This will become an expensive operation on very large ranges.
                                        ##xrange() on the other hand is much more optimised. it will only compute the next value when needed 
                                        ##(via an xrange sequence object) and does not create a list of all values like range() does.
 
        # Forward propagation
        z1 = data.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
 
        # We can also add regularization terms for weights(bias don't have regularization terms)
        # dW2 += reg_lambda * W2
        # dW1 += reg_lambda * W1
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        #if print_loss and i % 1000 == 0:
        #print "Loss after iteration %i: %f" %(i, calculate_loss(model))
     
    return model

def predict(model, data):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = data.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)       ## Returns the highest value comparing all the probabilities
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-2-4b09f432849f> in <module>()
         26 ## This will generate a 12x16 matrix mapping relevant CVs to each Job
         27 for i in range(len(df3)):
    ---> 28     k = df3['Relevent_CV_Ids'][i].strip().split(",")
         29     temp = k
         30     for j in range(len(temp)):


    AttributeError: 'int' object has no attribute 'strip'



```python
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Job_Id</th>
      <th>Designation</th>
      <th>Required Skills</th>
      <th>Min Exp (in months)</th>
      <th>Max Exp (in months)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>SDE Testing</td>
      <td>Java, J2EE, Sql, Javascript</td>
      <td>12.0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>SDE II Testing</td>
      <td>J2EE, Multithreading, Javascript, Regression T...</td>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>AVP</td>
      <td>Team Leading, MVC, Design, Feasability Analysi...</td>
      <td>60.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>SDE</td>
      <td>C#, Sql Server, Javascript</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>SDE II</td>
      <td>C#, Asp.Net, WPF, Framework 4</td>
      <td>36.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Team Lead</td>
      <td>ASP.NET, Team Leading, Blueprint Creation, Res...</td>
      <td>84.0</td>
      <td>No Limit</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Program Manager</td>
      <td>Presentation Skills, Leadership qualities, Tea...</td>
      <td>120.0</td>
      <td>No Limit</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>CTO</td>
      <td>Leadership Qualities, Entrepreneurial Spirit</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Data Scientist</td>
      <td>Machine Learning, Java, Python</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Lead Data Scientist</td>
      <td>Machine Learning, Deep Learning, NLP, RNN, LST...</td>
      <td>36.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Principal Data Scientist</td>
      <td>Team Leading, Machine Learning, AI, Deep Learn...</td>
      <td>84.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>SDE II</td>
      <td>J2EE, Multithreading, Javascript</td>
      <td>24.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
result_cst
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-195c870a11eb> in <module>()
    ----> 1 result_cst
    

    NameError: name 'result_cst' is not defined



```python

```
