# Binary Classification of Mushroom

This project is focused on the binary classification in eadible and poisonous, throught:

1. Cap diameter
2. Cap shape
3. Gill attachment
4. Gill color
5. Stem height
6. Stem width
7. Stem color
8. Season

Create a __deep neural network__ with 7 layers and study its the general properties of convergence.  Then I studied the optimization of the choice of: _nodes numbers in the first layer_, _larning rate_.

Onece found the optimal parameters, I created a _DNN_ and evaluated its performace through accuracy, ROC curve and the _Area Under the Curve_.  Then I analyzed the murshroom which were wrongly claffisfied by the model.

Finaly I analyzed (with [SHAP](https://shap.readthedocs.io/en/latest/)) the behavoir of the _DNN_ to find which parameters influance mostly the result to find possible coorlations between those features and the poisouness of the mushroom.
