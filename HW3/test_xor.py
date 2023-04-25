import numpy as np
import matplotlib.pyplot as plt

# implement the forward pass for approximating the XOR function
relu = lambda x: np.maximum(np.zeros(x.shape), x)
xor = lambda u, W, c, x: u.T @ relu(W.T @ x + c)

niter = 1

W = np.random.rand(2, 2)
u, c = np.random.rand(2), np.random.rand(2)


x = np.array([0, 0])
for ii in range(niter):
	# we can potentially also introduce another bias b
	# fx = u.T @ relu(W.T @ x + c)
	fx = xor(u, W, c, x)
	# x = x - eps * grad

# to make the above algorithm complete, we need to compute 
# the derivative of fx wrt x

# Instead, we take the (analytic) solution from the book
W = np.array([[1, 1], [1, 1]])
c = np.array([0, -1])
u = np.array([1, -2])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

print('X @ W : {}'.format(X @ W))
print('X @ W + c: {}'.format(X @ W + c))
print('relu(X @ W + c): {}'.format(relu(X @ W + c)))
print('relu(X @ W + c) @ u: {}'.format(relu(X @ W + c) @ u))

H = relu(X @ W + c)
plt.scatter(H[:, 0], H[:, 1])
plt.title('Hidden Representation')
plt.xlabel('h1')
plt.ylabel('h2')
plt.show()

