from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pythreejs as three
from IPython.display import display
import ipywidgets as widgets
import math

imagen = Image.open('azul.jpg')  

pixeles = list(imagen.getdata())
pixeles=pixeles[:400000]
r_c = [pixel[0] for pixel in pixeles]
g_c = [pixel[1] for pixel in pixeles]
b_c = [pixel[2] for pixel in pixeles]

def promedio(array):
    promedio = sum(array) / len(array)
    return promedio

def cov(X,Y):
    mean_X = promedio(X)
    mean_Y = promedio(Y)
    n = len(X)
    covariance = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n)) / (n - 1)
    return covariance

vector_prom = [promedio(r_c), promedio(g_c), promedio(b_c)]
r,g,b=vector_prom

matriz_covarianza=[
    [cov(r_c,r_c), cov(r_c,g_c), cov(r_c,b_c) ],
    [cov(r_c,g_c), cov(g_c,g_c), cov(b_c,g_c) ],
    [cov(r_c,b_c), cov(b_c,g_c), cov(b_c,b_c)]
]

eigenvalores, eigenvectores = np.linalg.eig(matriz_covarianza)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
origen = np.zeros(3)
for eigenvector, color in zip(eigenvectores, ['#F6A21E', '#E55B13', '#7A871E']):
    ax.quiver(*origen, *eigenvector*100, color=color, label='Eigenvector')

ax.scatter(r_c, g_c, b_c, c='green', marker='x', alpha=0.01, s=1, label='Datos originales')

ax.plot([0, r * 2.5], [0, g * 2.5], [0, b * 2.5], c='blue', linestyle='-', linewidth=1, label='Vector promedio')

# Configuración del gráfico
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')
ax.legend()
plt.show()