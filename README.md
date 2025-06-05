# tsetlin-binary-classification-python
Demo of Tsetlin binary classification (on a two-class subset of the Iris Dataset) using Python

The code here refactors the more-or-less official Tsetlin binary classification demo at github.com/cair/TsetlinMachine/tree/master from Pyrex (a kind of hybrid of C and Python) to pure Python with NumPy.

## Demo Output

The ouput of the demo code is:<br/>

Begin Tsetlin Machine binary classification<br/>
<br/>
Loading binary-features two-class Iris data<br/>
<br/>
First three X train items:<br/>
[[0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0]<br/>
 [0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0]<br/>
 [0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0]]<br/>
<br/>
First three y target (0-1) values:<br/>
0<br/>
0<br/>
0<br/>
<br/>
Setting n_clauses = 20<br/>
Setting n_states = 50<br/>
Setting s (random update inverse frequency) = 3.0<br/>
Setting threshold (voting max/min) = 10<br/>
Creating Tsetlin Machine binary classifier<br/>
Done<br/>
<br/>
Setting max_epochs = 100<br/>
Starting training<br/>
. . . . . . . . . .<br/>
Done<br/>
<br/>
Accuracy (train) = 0.8875<br/>
Accuracy (test) = 0.9500<br/>
<br/>
Prediciting class for train_X[0]<br/>
Predicted y = 0<br/>
<br/>
End demo<br/>

## The Data
For my demo data, I used the Iris Dataset from the github repository -- sort of. The 150-item Iris Dataset has three classes to predict (setosa, versicolor, virginica). So I used only the 50 setosa items (class 0) and 50 versicolor items (class 1). I used the first 40 of each class to make an 80-item training set, and the remaining 10 of each class to make a 20-item test set.

The raw Iris Dataset looks like:

5.1,3.5,1.4,0.2,Iris-setosa<br/>
4.9,3.0,1.4,0.2,Iris-setosa<br/>
4.7,3.2,1.3,0.2,Iris-setosa<br/>
. . .<br/>
7.0,3.2,4.7,1.4,Iris-versicolor<br/>
6.4,3.2,4.5,1.5,Iris-versicolor<br/>
. . .<br/>

To use Tsetlin classification the data must be converted to binary. Here is how the github data is encoded:

0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0<br/>
0, 0, 1, 1,  0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0<br/>
0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0<br/>
. . .<br/>
0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  1<br/>
0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  1<br/>
. . .<br/>

The raw Iris Dataset has four predictors: sepal length, sepal width, petal length, petal width. Each predictor value is mapped to four binary values, making 16 predictors per line, followed by the target class. The github page doesn't explain how the data is encoded, but after looking at it for a while, I deciphered that:

[0.0, 1.5] = 0000<br/>
[1.6, 3.1] = 0001<br/>
[3.2, 4.7] = 0010<br/>
[4.8, 6.3] = 0011<br/>
[6.4, 7.9] = 0100<br/>

This encoding is unusual. Notice that the first encoded value for each predictor is always 0, making it irrelevant. A more obvious encoding scheme would be to bucket the numeric predictor values into four buckets, and then use one-hot encoding. But I used the github encoding.
