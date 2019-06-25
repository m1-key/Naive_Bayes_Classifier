## Bayes Theorem
Let say there are two events A and B, then 
``` P(A/B) = (P(B/A) * P(A))/P(B) ```
where
* P(A/B) = Posterior Probability i.e Condtional Probability of A given B
* P(B/A) = Likelihood i.e Condtional Probability of B given A
* P(A) = Prior Probability
* P(B) = Normalisation Probability
---
#### Example
```
 Let say we have an email and now we have to check whether that email is spam or not. 
 If it is spam then it belongs to class 0 else class 1.
 Now we will calculate the posterior probability for spam and not spam. Let Y denotes class and X denotes email.
* P(Y=1/X) = (P(X/Y=1) * P(Y=1))/P(X)
* P(Y=0/X) = (P(X/Y=0) * P(Y=0))/P(X)
Now we will decide the class of mail X based on the maximum of these two posterior probabilities.
Since we are taking the maximum, then we can ignore or discard P(X) as it is common in both.
```
---
## Bayes Theorem for n features
Here we have a naive bayes assumption and that is that probability of a feature belonging to a class is not affected by the other features.
i.e Let say if a mushroom is blue then it won't give any other information about mushroom.
* Let X have n features and there are total m examples and k classes then ,
* ```P(Y=C/X) = (P(X/Y=C) * P(Y=C))/P(X) = (P(x1.x2.x3.x4...xn / Y=C) * P(Y=C))/P(X)```
* Acc. to bayes theorem ```(P(x1.x2.x3.x4...xn / Y=C) = P(x1/Y=C) * P(x2/Y=C) * P(x3/Y=C) * .....* P(xn/Y=C) ```
* ``` P(X) = P(X/Y=0) * P(Y=0) + P(X/Y=1) * P(Y=1) + .... + P(X/Y=K) * P(Y=K) ```
* So final formula is :-
* ```P(Y=C/X)=(P(x1/Y=C)*P(x2/Y=C)*P(x3/Y=C)* ...*P(xn/Y=C))*P(Y=C) / (P(X/Y=0)*P(Y=0)+P(X/Y=1)*P(Y=1)+...+ P(X/Y=K) * P(Y=K)) ```
* Here P(X) can discarded as we have to find maxm value of posterior prob and P(X) is common in every posterior prob so we can discard it.
