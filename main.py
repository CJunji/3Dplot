import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

#df = pd.read_csv('material.dat', delimiter='   ', names=['Index', 'X', 'Y', 'Energia'], header=None)
df = pd.read_csv('originalMod.dat', delimiter='  ', names=['X', 'Y', 'Energia'], header=None)

#print(df.head())

#X_unique = np.unique(df['X'])
#Y_unique = np.unique(df['Y'])
#X, Y = np.meshgrid(X_unique, Y_unique)
#Z = df.pivot(index='Y', columns='X', values='Energia').fillna(0).values


# Creating figure
#fig = plt.figure()
#ax = fig.add_subplot(111,   projection ='3d')
 
# Creating plot
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')
#ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, cmap='coolwarm')

#ax.set_xlabel('Eixo X')
#ax.set_ylabel('Eixo Y')
#ax.set_zlabel('Energia')
 
# show plot
#plt.show()

#x, y, z, xticks, yticks = plottable_3d_info(df)

#fig = plt.figure()
#ax = fig.add_subplot(projection = '3d')
#ax.plot_surface(x, y, z)
#ax.contour3D(x, y, z)
#ax.contour3D(X, Y, Z, 50, cmap='binary')

#ax.set_zlabel('Energia')
#plt.xticks(**xticks)
#plt.yticks(**yticks)
#plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.colors as mcolors


# Leitura dos dados a partir de um arquivo CSV
#df = pd.read_csv('seu_arquivo.csv')

# Função para diminuir a saturação de uma colormap
def desaturate_cmap(cmap, factor=0.5):
    new_cmap = cmap(np.arange(cmap.N))
    new_cmap[:, :3] = new_cmap[:, :3] * factor + (1 - factor)
    return mcolors.ListedColormap(new_cmap)

# Remover quaisquer linhas com valores NaN em X, Y ou Z
df = df.dropna(subset=['X', 'Y', 'Energia'])

# Transformar os dados em arrays
X = df['X'].values
Y = df['Y'].values
Z = df['Energia'].values

# Criação de um grid para interpolação
grid_x, grid_y = np.meshgrid(np.linspace(X.min(), X.max(), 100), np.linspace(Y.min(), Y.max(), 100))

# Interpolação dos valores de Z no grid
grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')

# Preenchendo possíveis valores NaN resultantes da interpolação com zero ou interpolação 'nearest'
grid_z = np.nan_to_num(grid_z, nan=0.0)

# Criação do gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotagem da superfície interpolada
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.75)

# Adicionando barra de cores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Energia (eV)')

# Adicionando rótulos aos eixos
ax.set_xlabel('Eixo X')
ax.set_ylabel('Eixo Y')
ax.set_zlabel('Energia (eV)')

img = mpimg.imread('MapaMaterial.jpg')
img = img/255.0

# Adicionar a imagem no plano Z=0
x_img, y_img = np.meshgrid(np.linspace(-5, 11, img.shape[1]), np.linspace(0, 7, img.shape[0]))
ax.plot_surface(x_img, y_img, np.zeros_like(x_img), rstride=5, cstride=5, facecolors=img, shade=False)

# Mostrando o gráfico
plt.show()
