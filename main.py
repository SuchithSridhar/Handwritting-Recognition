from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt


# Support Vector Classification
svc = svm.SVC(gamma=0.001, C=100.)


# 8x8 pixel images
digits = datasets.load_digits()

# print(digits.DESCR)

# Image 0 = white, 15 = black
# digits.images[0] --> Gives an array of one of the images


# Display the figure using matplotlib
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

# What its supposed to be  :
# digits.target

# We are supposed to consider the first 1791 as the training set and
# The remaining 6 as a validation set

train_count = 1790
test_count = 6

# Train the SVC estimator :
svc.fit(digits.data[1:train_count], digits.target[1:train_count])


final = 1796
# After the svc has done its thing, its time to test it
pre = (svc.predict(digits.data[final-test_count:final]))

# Then compare :
ans = (digits.target[final-test_count:final])

errors = 0
total = len(pre)

for i in range(len(pre)):
    if pre[i] != ans[i]:
        print(pre[i], "---/---", ans[i])
        errors += 1

print(pre, "\n", ans)
print("trains = ", train_count, "Mistakes :", errors, ", tot = ", total)
