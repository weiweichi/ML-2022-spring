
# Before run, make sure that torchaudio version is 11.0 or above
* If not, please command `pip3 install --upgrade torchaudio`
---
# Run code
* Just run `hw04_k-fold.py`. It will generate 5 models and ensemble them by fusion.
  
Reference by TA recommendation: [self-attaion pooling](https://arxiv.org/pdf/2008.01077v1.pdf) & [Additive Margin Softmax](https://arxiv.org/pdf/1801.05599.pdf)
|ranking|public score|private score|
|:-----:|------------|-------------|
|67/517 |0.85825|0.85625|