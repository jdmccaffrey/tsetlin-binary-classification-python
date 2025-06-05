# tsetlin-binary-classification-python
Demo of Tsetlin binary classification on a two-class subset of the Iris Dataset, using Python

The code here (in file tsetlin_machine_binary.py) refactors the more-or-less official Tsetlin binary classification demo at github.com/cair/TsetlinMachine/tree/master from Pyrex (a kind of hybrid of C and Python) to pure Python with NumPy.

## Demo Output

The ouput of the demo code is:

    Begin Tsetlin Machine binary classification

    Loading binary-features two-class Iris data

    First three X train items:
    [[0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0]
     [0 0 1 1 0 0 0 1 0 0 0 0 0 0 0 0]
     [0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0]]

    First three y target (0-1) values:
    0
    0
    0

    Setting n_clauses = 20
    Setting n_states = 50
    Setting s (random update inverse frequency) = 3.0
    Setting threshold (voting max/min) = 10
    Creating Tsetlin Machine binary classifier
    Done

    Setting max_epochs = 100
    Starting training
    . . . . . . . . . .
    Done

    Accuracy (train) = 0.8875
    Accuracy (test) = 0.9500

    Prediciting class for train_X[0]
    Predicted y = 0

    End demo

## The Data
For my demo data, I used the Iris Dataset from the github repository -- sort of. The 150-item Iris Dataset has three classes to predict (setosa, versicolor, virginica). So I used only the 50 setosa items (class 0) and 50 versicolor items (class 1). I used the first 40 of each class to make an 80-item training set, and the remaining 10 of each class to make a 20-item test set.

The raw Iris Dataset looks like:

    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa
    4.7,3.2,1.3,0.2,Iris-setosa
    . . .
    7.0,3.2,4.7,1.4,Iris-versicolor
    6.4,3.2,4.5,1.5,Iris-versicolor
    . . .

To use Tsetlin classification the data must be converted to binary. Here is how the github data is encoded:

    0, 0, 1, 1,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0
    0, 0, 1, 1,  0, 0, 0, 1,  0, 0, 0, 0,  0, 0, 0, 0,  0
    0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, 0, 0,  0
    . . .
    0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  1
    0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 1, 0,  0, 0, 0, 0,  1
    . . .

The raw Iris Dataset has four predictors: sepal length, sepal width, petal length, petal width. Each predictor value is mapped to four binary values, making 16 predictors per line, followed by the target class. The github page doesn't explain how the data is encoded, but after looking at it for a while, I deciphered that:

    [0.0, 1.5] = 0000
    [1.6, 3.1] = 0001
    [3.2, 4.7] = 0010
    [4.8, 6.3] = 0011
    [6.4, 7.9] = 0100

This encoding is unusual. Notice that the first encoded value for each predictor is always 0, making it irrelevant. A more obvious encoding scheme would be to bucket the numeric predictor values into four buckets, and then use one-hot encoding. But I used the github encoding.

## Notes
The Tsetlin classifier requires four parameter values that must be determined by trial and error:

    Setting n_clauses = 20
    Setting n_states = 50
    Setting s (random update inverse frequency) = 3.0
    Setting threshold (voting max/min) = 10

Loosely, the number of clauses is essentially the number of rules, but the details are fairly complicated. The number of states refers to the automata component. The s parameter adds some randomness during training, to avoid model overfitting. The threshold value acts to clip output so that some clauses don't dominate a prediction.

From a practical point of view, I think that Tsetlin binary classification isn't used much because having to encode numeric predictors to binary is somewhat time-consuming. On the other hand, for problem scenarios where the predictors are all categorical (e.g., color is red, blue, or green), the binary encoding isn't a major issue.

The main source research paper is "The Tsetlin Machine - A Game Theoretic Bandit Driven Approach to Optimal Pattern Recognition with Propositional Logic" by Ole-Christoffer Granmo, 2021. It's available at arxiv.org/abs/1804.01508. The technique is named after M.L. Tsetlin, a Soviet Union mathematician from the 1960s.

Granmo has extended Tsetlin binary classification technique to multi-class classification, and to regression. I haven't looked at these techniques yet.
