import matplotlib.pyplot as plt
# use the same function f(x, a) as before
def f(x, a, b):
    import math
    #return -1 + (x / a) ** 2
    if a < x <= b:
        c = (a + b) / 2
        # calculate the coefficient of the quadratic term
        k = -4 / ((b - a) ** 2)
        # return the value of the parabola
        return k * (x - c) ** 2 + 1
    if x <= a:
        return -1 + (x / a) ** 2
    else:
        k = 1/((b-1)**2)
        return k * (x-1)**2 - 1
  # 返回f(x)的值
a = 0.3
b = 0.5
# create a list of x values from 0 to a, for example, with 100 points
x = [i * 1 / 100 for i in range(101)]
# create a list of y values using f(x, a)
y = [f(i, a , b) for i in x]
# plot the curve using matplotlib
plt.plot(x, y)
# add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('A monotonically increasing curve')
# save the figure as an image file, for example, as png format
plt.savefig('curve.png')

