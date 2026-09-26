import numpy as np
import matplotlib.pyplot as plt


tamano = 256
imagen_ideal = np.ones((tamano, tamano)) * 0.15


# cuadrado
inicio_square = 64
fin_square = 192

imagen_ideal[inicio_square:fin_square,
             inicio_square:fin_square] = 0.45


# circulo
centro_x = tamano // 2
centro_y = tamano // 2
radio_circle = 32

y_pos, x_pos = np.ogrid[:tamano, :tamano]

distancia = (x_pos - centro_x)**2 + (y_pos - centro_y)**2
mask_circle = distancia <= radio_circle**2

imagen_ideal[mask_circle] = 0.80


# mask cuadrado sin circulo
mask_square = np.zeros((tamano, tamano), dtype=bool)
mask_square[inicio_square:fin_square,
            inicio_square:fin_square] = True

mask_square[mask_circle] = False


# mask fondo
mask_background = np.ones((tamano, tamano), dtype=bool)
mask_background[inicio_square:fin_square,
                inicio_square:fin_square] = False


# check masks
count_background = np.sum(mask_background)
count_square = np.sum(mask_square)
count_circle = np.sum(mask_circle)

total_pixeles = count_background + count_square + count_circle

check = total_pixeles == tamano * tamano

if check:
    print("Masks correctas")
else:
    print("Revisar masks")




# agregar ruido poisson
N = 40
seed_numero = 12

np.random.seed(seed_numero)

imagen_ruido = np.random.poisson(N * imagen_ideal) / N



# revisar media y varianza
media_background = np.mean(imagen_ruido[mask_background])
media_square = np.mean(imagen_ruido[mask_square])
media_circle = np.mean(imagen_ruido[mask_circle])

var_background = np.var(imagen_ruido[mask_background])
var_square = np.var(imagen_ruido[mask_square])
var_circle = np.var(imagen_ruido[mask_circle])

print("\nRuido Poisson")
print("Media background:", media_background)
print("Media square:", media_square)
print("Media circle:", media_circle)

print("Var background:", var_background)
print("Var square:", var_square)
print("Var circle:", var_circle)





print("Background:", count_background)
print("Square:", count_square)
print("Circle:", count_circle)


plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(imagen_ideal, cmap="gray", vmin=0, vmax=1)
plt.title("Imagen ideal")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(mask_background, cmap="gray")
plt.title("Mask background")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(mask_square, cmap="gray")
plt.title("Mask square")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(mask_circle, cmap="gray")
plt.title("Mask circle")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/regiones_sinteticas.png",
    dpi=150
)

plt.show()

# comparar imagen ideal y ruido
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(imagen_ideal, cmap="gray", vmin=0, vmax=1)
plt.title("Imagen ideal")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(imagen_ruido, cmap="gray", vmin=0, vmax=1)
plt.title("Ruido Poisson N=40")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/ruido_poisson.png",
    dpi=150
)

plt.show()