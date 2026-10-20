# freight rate prediction

predicts the posted rate of a freight load from its route, distance, equipment, weight and date. built for the spotter ml engineer assessment.

## how to run

put the assessment files in `data/` then:

```
pip install -r requirements.txt
python train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

train.py prints the test scores, then writes `validation_predictions.csv` and fills `data/december_chart_inputs.csv`. score.py checks both files and makes the december chart.

## files

* `train.py`: the full pipeline, cleaning to final predictions
* `exploration_clean.ipynb`: the data exploration with every finding and decision
* `validation_predictions.csv`: predictions for the 12,000 validation loads
* `report.pdf`: split approach, results and the december chart
* `scorer_results/candidate_december.png`: the december chart

## what i found in the data

* rates are seasonal (11% swing, peak in june) and depend a little on the weekday (2.6%, wednesday and thursday highest)
* about 670 loads (1.4%) have rates far above or below normal and nothing explains them (not distance, equipment, weight, date or route). removed from training only and kept in the test
* 292 negative weights in training and 145 in validation. they closely mirror the positive weights so it is a sign error, fixed with abs()
* 300 missing weights in training and 165 in validation, filled with the training median
* weights are clipped at 5,000 and 47,500
* quote_signal is target leakage: its scatter plot against rate per mile is an X shape (two exact lines plus noise), so it was dropped
* market_index shows no signal and is missing from the december file, so it was dropped

## split

time based: train on january to august, test on september and october, because the real task is predicting two unseen future months. a random split would let the model see the future. the final model is retrained on all of january to october.

## results (test on september and october)

| model | MAE |
|---|---|
| baseline (median rate per mile x distance) | 256.94 |
| random forest | 106.45 |
| random forest + lane feature | 105.67 |
| lightgbm + lane feature | 107.84 |
| average of random forest and lightgbm | 102.45 |

the lane feature is the typical rate per mile of each pickup and delivery pair, computed with an out of fold target encoder so a load never sees its own price.

## december chart

![december chart](scorer_results/candidate_december.png)

## limitation

the training data ends in october so the model never saw december or any holiday. the december chart shows a weekly pattern only, christmas looks like a normal thursday. data from previous years would fix that.
