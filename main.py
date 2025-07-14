#imports

import pandas


def print_hi(name):
    print(f'Hi, {name}')


print_hi("siema")


def load_file(filename):
    file = pandas.read_csv(filename, header=None, names=["t", "x", "y", "z"])
    print(file)
    return(
        file["t"].values[:10000],
        file["x"].values[:10000],
        file["y"].values[:10000],
        file["z"].values[:10000]
    )

t,x,y,z = load_file("res/normalne11.csv")

print(t)
print(x)
print(y)
print(z)

