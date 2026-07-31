# Freight Rate Prediction

Predicts the posted rate of a freight load from its route, distance, equipment type, weight and date.

## Setup

Put the data files in a `data/` folder:

```
data/
├── train-test.csv
├── validation.csv
└── december-chart-inputs.csv
```

Then install what it needs:

```bash
python -m pip install -r requirements.txt
```

## Running it

```bash
python train.py
```

That writes `validation_predictions.csv` with all 12,000 rows and fills in the
`predicted_rate` column of `december-chart-inputs.csv`.

To check the files and get the chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

## How I split the data

The training file covers January to October 2025 and the validation file covers
November and December, so the actual job is predicting forward in time. I split
by date instead of randomly: January to August for training, September and
October for testing. A random split would let the model learn from dates sitting
on both sides of every test row, which would make the score look better than the
model really is.

## What I found in the data

Distance drives almost everything (0.91 correlation with the rate), but not in a
straight line. Short runs cost more per mile than long ones, around $2.56 under
500 miles down to $1.95 past 2,000. That's why I went with tree models instead of
a linear one.

Equipment matters too. Reefers average $2.38 a mile, flatbeds $2.29, dry vans
$2.12. Pickup and delivery cities also shift the price by roughly 20% between the
cheapest and most expensive, and it looks regional rather than directional. The
same cities sit at the same end whether they're the origin or the destination.

There's a weekly cycle as well. Rates dip on Sunday and Monday and peak midweek.
It's small on the rate itself, about 4% on loads near 360 miles, but it's
consistent and it's the only time signal that carries over into December.

## Missing values

`weight` is missing in 300 rows and `market_index` in 374. The missing weights are
spread evenly across dates so nothing systematic is going on, and median weight
barely moves between equipment types (31,373 to 31,524 lb), so I just used the
overall median. `market_index` swings with the day of week, so filling it with a
single global median would put a Thursday number into a Sunday row. I used the
median for each day of week instead.

Both medians come from the training data and get applied to validation unchanged.
Computing them from validation would be leakage, and in a real deployment you
wouldn't have that data sitting there to compute from anyway.

## The expensive loads

317 loads are priced above $5 a mile when the median is $2.15, and the worst one
hits $14. Before deciding what to do with them I checked whether anything in the
data marks them out. Nothing does. Their equipment mix matches the overall data
almost exactly, they're spread across all seven days, and their distances cover
the full range. So there's no way for any model to know in advance which load is
going to price like that.

Leaving them in doesn't help, it just drags predictions upward for every normal
load that looks similar. I cut the top and bottom 1% of rate per mile from the
training data only. Test MAE went from $150.88 to $116.88. I left the test set
alone, because trimming that too would make the score look good without the model
being any better.

## Features

Every city maps to exactly one lat/lon pair, so I used the coordinates rather than
one-hot encoding 64 city names. Equipment is one-hot encoded, three columns.

For the date I only pulled out day of week. Month would be a problem: the model
never sees months 11 or 12 in training, and trees can't extrapolate past the range
they've seen, so the December chart would come out flat or wrong. The weekly cycle
repeats the same in December as it does in January, so that part transfers fine.

I dropped `market_index` and `quote_signal`. They aren't in the December file at
all, so I had to test a model without them anyway, and that model turned out
better: MAE went from $116.88 down to $106.92. Both features correlate near zero
with the rate (0.03 and -0.04), so they seem to have been feeding the trees noise
to split on. Since the reduced model was better everywhere, I used it for both the
validation predictions and the chart instead of keeping two models around.

## Model

Random Forest, 500 trees.

I also tried XGBoost, since gradient boosting usually does well on tabular data.
It came out worse, MAE $138.86 against $116.88. I switched its objective to
absolute error in case that was the issue and it barely moved, $138.86 to $138.86.
My read is that boosting builds trees one after another to fix the previous
errors, so it keeps chasing those unpredictable expensive loads, while a forest
averages independent trees and stays calmer about them.

Tuning didn't do much either. `min_samples_leaf` from 1 to 20 moved MAE by less
than a dollar and got worse past 5, and going from 100 to 500 trees changed it by
11 cents. That fits the rest of what I found: the error that's left isn't the
model underfitting, it's noise in the data that isn't there to be learned.

## Results

Tested on September and October, with outliers left in:

R² 0.8282, MAE $106.92, on rates averaging around $2,370.

## Files

- `train.py` runs the whole thing, preprocessing through to the output files
- `notebook.ipynb` is the exploration I did before writing that
- `requirements.txt`
