# Uncertainty Quantification
## This is a project I did as part of a PhD course at NTNU (DT8122:Probabilistic Artificial Intelligence )
### See [Project_report](Project_report.pdf) for full documentation.

Deep learning models have been widely used in many real-world applications and quantifying their uncertainties are of crucial importance. However, most deep learning models operate in a ”deterministic” fashion by having only point estimates of parameters and predictions, thus lacking credibility for decision making in real-life settings. In comparison, probabilist models place distributions over parameters and predictions, which provides more information about model uncertainty.

In this project, I explore Bayesian methods and their approximate inference as techniques to capture model uncertainty in prediction tasks. I implemented two deep generative modeling techniques: `Dropout as a Bayesian Approximation` to estimate uncertainty as well as `Bayes by Backprop` as a technique to approximate Bayesian neural networks using stochastic variational inference.  Then demonstrate the inference methods and perform empirical evaluations on various benchmark regression dataset. Finally, I discuss the shortcomings of these two techniques, then suggest and implement possible improvements along with empirical evaluation and results comparison.

