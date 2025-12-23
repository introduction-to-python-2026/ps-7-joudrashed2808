import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)


print(df.head())
print(df.describe())


x = "sepal length (cm)"
y = "petal length (cm)"


df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df[x], df[y])
plt.xlabel(x)
plt.ylabel(y)
plt.title("Correlation between Sepal Length and Petal Length")


plt.savefig("correlation.png")
plt.show()


correlation = df[x].corr(df[y])
print(f"Correlation between {x} and {y}: {correlation:.2f}")

