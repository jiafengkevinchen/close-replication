library(REBayes)
library(nprobust)
library(Rmosek)

set.seed(42)
x <- rnorm(100)
y <- rnorm(100)

reg <- lprobust(y, x)
npeb <- GLmix(x)

print(reg)
print(npeb)
