# Example of Classification Algorithm
# A Classification Algorithm predicts a Class (0 or 1)
#
# In this example, the classifier predicts fish (0) or cat (1) based on 
# the features (legs, weight). 
#
# Practical example: 
# Given data [4 legs, 5kg], the algorithm predicts cat (1)
#
# Watch the video to see further explanation.

# Load Algorithm
from sklearn import tree

### 1: Create Training Data (= features and labels)

# Features are measurements (legs, weight)
features = [[0,50],[0,150000],[4,5],[4,6],[0,0.05]]

# Labels are class outcomes (0 or 1)
labels = [0,0,1,1,0]

### 2: Initialize and Train algorithm

# Initialize Classification Algorithm named DecisionTree
algorithm = tree.DecisionTreeClassifier()       

# Train algorithm with training data (features,labels)
algorithm = algorithm.fit(features, labels)

### 3: Predict

# New data/measurements: [4 legs, 5kg]
newData = [[2,150000]]

# Predict fish (0) or cat (1)?
print(algorithm.predict(newData))
