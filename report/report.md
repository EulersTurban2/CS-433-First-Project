# Important notes for project and the report



1. Implementation of ML methods  
  
- For SGD, you must use the standard mini-batch-size 1 (sample just one datapoint).  
- The mean squared error formula has a factor 0.5 to be consistent with the lecture notes  
- For least squares, you are allowed to use everything from numpy.linalg but not numpy.linalg.lstsq  
- All vectors should be implemented as 1D arrays  
with shape (X,) instead of (X, 1).  
  
- Return type: Note that all functions should return: (w, loss), which is the last weight vector of the  
method, and the corresponding loss value (cost function). Note that while in previous labs you might  
have kept track of all encountered w for iterative methods, here we only want the last one. Moreover, the  
loss returned by the regularized methods (ridge regression and reg logistic regression) should  
not include the penalty term.  
  
• File names: Please provide all function implementations in a single python file, called [implementations.py](http://implementations.py).
    


