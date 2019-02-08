import numpy as np


class Test:

    def __init__(self):
        self.number = 1

    def function_1(self):
        return self.number + 1

    def __call__(self, number):
        return number


if __name__ == "__main__":

    myClass = Test()

    myClass(5)
