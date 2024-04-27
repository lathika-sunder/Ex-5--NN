
<H3>EX. NO.5</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

<H3>DEVELOPED BY:Lathika Sunder</H3>
<H3>REGISTER NUMBER: 212221230054</H3>

```py
def predict_matrix(point, weights):
  gaussian_rbf_0 = gaussian_rbf(np.array (point), mu1)
  gaussian_rbf_1 = gaussian_rbf(np.array (point), mu2)
  A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
  return np.round(A.dot(weights))

  x1 = np.array([0, 0, 1, 1])
  x2 = np.array([0, 1, 0, 1])
  ys = np.array ([0, 1, 1, 0])
  mu1 = np.array([0, 1])
  mu2 = np.array([1, 0])
  w = end_to_end(x1, x2, ys, mu1, mu2)
  print(f"Input:{np.array ([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}") 
  print(f"Input: {np.array ([0, 1])}, Predicted: {predict_matrix(np.array ([0, 1]), w)}")
  print(f"Input: {np.array ([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
  print(f"Input: {np.array ([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")

def predict_matrix(point, weights):
  gaussian_rbf_0 = gaussian_rbf(np.array (point), mu1)
  gaussian_rbf_1 = gaussian_rbf(np.array (point), mu2)
  A = np.array([gaussian_rbf_0, gaussian_rbf_1, 1])
  return np.round(A.dot(weights))

  x1 = np.array([0, 0, 1, 1])
  x2 = np.array([0, 1, 0, 1])
  ys = np.array ([0, 1, 1, 0])
  mu1 = np.array([0, 1])
  mu2 = np.array([1, 0])
  w = end_to_end(x1, x2, ys, mu1, mu2)
  print(f"Input:{np.array ([0, 0])}, Predicted: {predict_matrix(np.array([0, 0]), w)}") 
  print(f"Input: {np.array ([0, 1])}, Predicted: {predict_matrix(np.array ([0, 1]), w)}")
  print(f"Input: {np.array ([1, 0])}, Predicted: {predict_matrix(np.array([1, 0]), w)}")
  print(f"Input: {np.array ([1, 1])}, Predicted: {predict_matrix(np.array([1, 1]), w)}")
```
<H3>OUTPUT:</H3>

![image](https://github.com/MeethaPrabhu/Ex-5--NN/assets/119401038/70fac2c1-2658-44a0-976d-2594e2ba0c7e)

<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.







