from scipy.stats import norm
import numpy as np

# Compute the z-score, and tail probability outlier score.
#     zscore(x) = |x - EX| / σ(X)
#     outlier score (x) = -log P(|X - EX| >= |x - EX|)
class ZOutlierScore:
    def __init__(self, X):
        self.loc = np.mean(X) 
        self.scale = np.std(X) 

    def score(self, X):
        return -norm.logcdf(np.abs((X - self.loc) / self.scale))

def test_z_scorer():
    X = np.random.rand(10)
    scorer = ZOutlierScore(X)
    scorer.loc
    scorer.scale
    scorer.score(X)
