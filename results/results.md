# Results

Stratified 5-fold means from `02_models_and_ablations.ipynb`. F1 uses a cutoff chosen on that fold's validation scores. AUROC does not.

Training labels: 28,975 positive of 328,135 (8.83%). Always -1: F1 0.000, accuracy 0.912. Always +1: F1 0.162, accuracy 0.088.

## Models

| model | F1 | std | precision | recall | accuracy | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| least squares | 0.422 | 0.002 | 0.346 | 0.539 | 0.869 | 0.854 |
| ridge 0.0001 | 0.422 | 0.001 | 0.346 | 0.539 | 0.869 | 0.854 |
| ridge 0.001 | 0.422 | 0.002 | 0.346 | 0.539 | 0.869 | 0.854 |
| ridge 0.01 | 0.421 | 0.002 | 0.342 | 0.548 | 0.867 | 0.854 |
| ridge 0.1 | 0.418 | 0.003 | 0.340 | 0.544 | 0.866 | 0.854 |
| mse gd | 0.421 | 0.003 | 0.339 | 0.557 | 0.865 | 0.854 |
| logistic | 0.430 | 0.004 | 0.352 | 0.556 | 0.870 | 0.860 |
| reg logistic 0.0001 | 0.430 | 0.004 | 0.357 | 0.541 | 0.873 | 0.860 |
| reg logistic 0.001 | 0.430 | 0.004 | 0.354 | 0.549 | 0.871 | 0.861 |
| reg logistic 0.01 | 0.428 | 0.002 | 0.347 | 0.561 | 0.867 | 0.859 |
| sgd | nan | nan | nan | nan | nan | nan |

## Ablations

Reference: ridge 0.001. `delta` is the mean of the fold-by-fold F1 differences.

| change | delta | std | F1 | against |
| --- | ---: | ---: | ---: | --- |
| default cutoff | -0.320 | 0.005 | 0.102 | ridge 0.001 |
| undersample 1:1 | +0.004 | 0.005 | 0.426 | ridge 0.001 |
| undersample 1:3 | +0.006 | 0.002 | 0.428 | ridge 0.001 |
| unpenalized intercept | +0.000 | 0.000 | 0.422 | ridge 0.001 |
| class weight | -0.001 | 0.001 | 0.429 | unweighted logistic |
| clinical products | +0.007 | 0.006 | 0.429 | ridge 0.001 |

## Figures

Least squares and ridge. The dashed line is always predicting +1.

![Validation F1 for least squares and ridge](figures/least_squares_ridge_f1.png)

Training loss on the first fold.

![Training loss](figures/training_loss.png)


## Predictions

Refit `logistic` on all training rows. Validation F1 0.430 +/- 0.004. Cutoff 0.196. Predicted +1 on 15,364 of 109,379 test rows (14.05%).

The submission file is [predictions.csv](predictions.csv), with columns `Id` and `Prediction`.
