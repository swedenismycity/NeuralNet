import numpy as np 
from numpy.typing import NDArray


class NeuralNet:
    def __init__(self, IL: int, ML: int, OL: int):
        self.middle_W = np.ones((IL, ML)) #Middle weights matrix 
        self.output_W = np.ones((ML, OL)) #Output weights matrix 

        self.middle_B = np.ones((ML, ))
        self.output_B = np.ones((OL, ))

        self.learning_rate = 0.8

    def predict(self, I: NDArray[np.float64]):
        self.middle_raw = np.matmul(I, self.middle_W) + self.middle_B
        self.middle_guess = self.sigmoid(self.middle_raw)
        self.output_guess = self.sigmoid(np.matmul(self.middle_guess, self.output_W) + self.output_B)

    def backprop(self, inp: NDArray[np.float64], awnser: NDArray[np.float64]) -> NDArray[np.float64]:
        
        self.predict(inp)
        
        output_err = (self.output_guess - awnser) * self.sigmoid_prime(self.output_guess)
        middle_err = np.matmul(np.transpose(self.output_W),output_err) * self.sigmoid_prime(self.middle_guess)

        output_W_delta = np.matmul(self.middle_guess.reshape(-1, 1), output_err.reshape(1, -1))
        middle_W_delta = np.matmul(inp.reshape(-1, 1), middle_err.reshape(1, -1))
        #print(self.middle_W)

        self.output_W -= output_W_delta * self.learning_rate
        self.output_B -= output_err * self.learning_rate

        self.middle_W -= middle_W_delta * self.learning_rate
        self.middle_B -= middle_err * self.learning_rate

        #print(np.mean(np.pow(self.output_guess - awnser,2)))
        return self.output_guess
        """
        print("Truth:",awnser)
        print("Guess:",self.middle_guess)
        print("-----------------------")
        """



    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    def sigmoid_prime(self, z):
        return z * (1-z)


"""
nn = NeuralNet(4,2,2)

while True:
    nn.backprop(np.array([1,1,1,1]), [0.2, 0.5])
"""
