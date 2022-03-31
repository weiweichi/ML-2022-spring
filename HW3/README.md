## Brief description
I implement **TA's CNN model** and **two Residual models** (TA's and Pytorch model), but something different is that I use **5-fold cross validation** and **esemble** methods.

---
## Run code
* Run `k-fold_cnn.py`, `k-fold_myRes.py` and `k-fold_res18.py` respectively. Each of them will generate `5` models and `1` prediction csv file.

* Run `ensemble_all.py`. It will ensemble whole `15` models by fusion and the final predictions are saved in ensemble-all.csv

* Run `predict_by_voting.py`. It will ensemble 3 submission files by voting and the final predictions are saved in predict_by_voting.csv

|   model   |public score|private score|
|:---------:|:----------:|:-----------:|
|   myRes    |0.89243     |0.87153|
|    cnn     |0.90936     |0.89201|
|   res18    |0.85159     |0.85159|
|cnn + myRes |0.90737     |0.88817|
|ensemble all|0.90737     |0.88988|
|   voting   |0.90139     |0.88390|