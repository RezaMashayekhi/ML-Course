# Machine Learning-Course-Project
Detecting real disaster tweets from the fake ones.

The final project of the ML course I had at UofA in 2021.


Although people may appreciate accessing news on almost every platform, each technology has its downsides. Explicitly speaking, we have seen many internet users who tend to release "Fake Disaster News" in their personal or public web pages (e.g., once in a while, you can see a non-logical trend on Twitter that doesn't make sense in any way). Disasters are considered emergencies and demand high-priority help from the government and people. Consequently, it would be valuable to have a classifier that can easily differentiate between fake vs. real disaster news. In this regard, we've tried to implement several machine learning models to solve the problem.

This project was a group project. I implemented the Perceptron model for this project in two ways. One from scratch and without using a pre-existing model in libraries, and the other one by using libraries like Scikit-learn.

The model was trained with different learning rates from 0.05 to 3, and was tested on the validation set. The learning rate 1.8 resulted in a higher accuracy on the validation set. The following figure is showing accuracies for different learning rates.

<p align="center">
<img width="500" alt="Perceptron_Learning_Rate" src="https://user-images.githubusercontent.com/29575804/177226207-b701d8b2-07fc-413d-9cfb-91796da30f63.png">
</p>
<p align="center">
Perceptron’s accuracy with respect to the different learning rates
</p>

Perceptron obtained an accuracy of 71.5 % on the test data.

<p align="center">
<img align="center" width="500" src="https://user-images.githubusercontent.com/29575804/177226219-039c0d8e-a986-463c-b971-c522cafff217.jpeg">
</p>
<p align="center">
Loss plot of Perceptron model
</p>

According to this result, it seems like the Perceptron model is not a good model for our data because the data is probably not linearly separable. I also tried to improve the result by randomly initialize the weights of the model, which improved the accuracy by 1.5% on average. Here is also the loss plot for the training and validation sets. Because of these up and downs in the validation set, choosing a good epoch doesn’t seem to be meaningful.
