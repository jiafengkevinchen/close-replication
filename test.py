# Tests REBayes, Mosek, RPy2 installation
import numpy as np
import rpy2
from rpy2.robjects import numpy2ri
from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
import rpy2.situation

numpy2ri.activate()
rebayes = importr("REBayes")
nprobust = importr("nprobust")

np.random.seed(42)

x = np.random.randn(1000)
y = np.random.randn(1000)
reg = nprobust.lprobust(x=ro.FloatVector(x), y=ro.FloatVector(y), deriv=0)
print(reg)

estimates = np.random.randn(1000)
standard_errors = np.ones(1000)
result = rebayes.GLmix(estimates, sigma=standard_errors)

plt.plot(result.rx["x"][0], result.rx["y"][0])
print(result)
plt.savefig("test.png")
