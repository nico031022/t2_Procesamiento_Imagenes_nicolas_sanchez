import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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




def crear_kernel_gaussiano(sigma):

    # radio de 3 sigma
    radio = int(np.ceil(3 * sigma))

    x = np.arange(-radio, radio + 1)
    y = np.arange(-radio, radio + 1)

    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    # normalizar
    kernel = kernel / np.sum(kernel)

    return kernel


# para probar kernel gaussiano
sigma_prueba = 1.3

kernel_gauss = crear_kernel_gaussiano(sigma_prueba)

print("\nKernel Gaussiano")
print("Sigma:", sigma_prueba)
print("Tamano:", kernel_gauss.shape)
print("Suma:", np.sum(kernel_gauss))


# filtro
imagen_filtrada = convolve2d(
    imagen_ruido,
    kernel_gauss,
    mode="same",
    boundary="symm"
)




def calcular_rmse(imagen_ref, imagen_test, mask=None):

    if mask is None:
        diferencia = imagen_ref - imagen_test
    else:
        diferencia = imagen_ref[mask] - imagen_test[mask]

    error = np.sqrt(np.mean(diferencia**2))

    return error

# rmse imagen con ruido
rmse_ruido_global = calcular_rmse(
    imagen_ideal,
    imagen_ruido
)

rmse_ruido_background = calcular_rmse(
    imagen_ideal,
    imagen_ruido,
    mask_background
)

rmse_ruido_square = calcular_rmse(
    imagen_ideal,
    imagen_ruido,
    mask_square
)

rmse_ruido_circle = calcular_rmse(
    imagen_ideal,
    imagen_ruido,
    mask_circle
)


# rmse imagen filtrada
rmse_gauss_global = calcular_rmse(
    imagen_ideal,
    imagen_filtrada
)

rmse_gauss_background = calcular_rmse(
    imagen_ideal,
    imagen_filtrada,
    mask_background
)

rmse_gauss_square = calcular_rmse(
    imagen_ideal,
    imagen_filtrada,
    mask_square
)

rmse_gauss_circle = calcular_rmse(
    imagen_ideal,
    imagen_filtrada,
    mask_circle
)


print("\nRMSE imagen ruido")
print("Global:", rmse_ruido_global)
print("Background:", rmse_ruido_background)
print("Square:", rmse_ruido_square)
print("Circle:", rmse_ruido_circle)

print("\nRMSE Gaussiano sigma=1.3")
print("Global:", rmse_gauss_global)
print("Background:", rmse_gauss_background)
print("Square:", rmse_gauss_square)
print("Circle:", rmse_gauss_circle)


check_rmse = rmse_gauss_global < rmse_ruido_global

if check_rmse:
    print("\nEl filtro reduce el RMSE global")
else:
    print("\nRevisar resultado del filtro")



# probar varios sigma
sigma_values = [0]

rmse_global_values = [rmse_ruido_global]
rmse_background_values = [rmse_ruido_background]
rmse_square_values = [rmse_ruido_square]
rmse_circle_values = [rmse_ruido_circle]


for sigma in np.arange(0.1, 4.1, 0.1):

    kernel = crear_kernel_gaussiano(sigma)

    imagen_sigma = convolve2d(
        imagen_ruido,
        kernel,
        mode="same",
        boundary="symm"
    )

    error_global = calcular_rmse(
        imagen_ideal,
        imagen_sigma
    )

    error_background = calcular_rmse(
        imagen_ideal,
        imagen_sigma,
        mask_background
    )

    error_square = calcular_rmse(
        imagen_ideal,
        imagen_sigma,
        mask_square
    )

    error_circle = calcular_rmse(
        imagen_ideal,
        imagen_sigma,
        mask_circle
    )

    sigma_values.append(sigma)

    rmse_global_values.append(error_global)
    rmse_background_values.append(error_background)
    rmse_square_values.append(error_square)
    rmse_circle_values.append(error_circle)


# sacar minimos
min_global = min(rmse_global_values)
indice_global = rmse_global_values.index(min_global)
sigma_min_global = sigma_values[indice_global]

min_background = min(rmse_background_values)
indice_background = rmse_background_values.index(min_background)
sigma_min_background = sigma_values[indice_background]

min_square = min(rmse_square_values)
indice_square = rmse_square_values.index(min_square)
sigma_min_square = sigma_values[indice_square]

min_circle = min(rmse_circle_values)
indice_circle = rmse_circle_values.index(min_circle)
sigma_min_circle = sigma_values[indice_circle]


print("\nMinimos RMSE")

print("Global:")
print("Sigma =", sigma_min_global)
print("RMSE =", min_global)

print("\nBackground:")
print("Sigma =", sigma_min_background)
print("RMSE =", min_background)

print("\nSquare:")
print("Sigma =", sigma_min_square)
print("RMSE =", min_square)

print("\nCircle:")
print("Sigma =", sigma_min_circle)
print("RMSE =", min_circle)


# revisar limite del rango
sigma_maximo = sigma_values[-1]

if sigma_min_global == sigma_maximo:
    print("\nRevisar rango para global")

if sigma_min_background == sigma_maximo:
    print("Revisar rango para background")

if sigma_min_square == sigma_maximo:
    print("Revisar rango para square")

if sigma_min_circle == sigma_maximo:
    print("Revisar rango para circle")









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

# comparar filtro gaussiano
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(imagen_ideal, cmap="gray", vmin=0, vmax=1)
plt.title("Imagen ideal")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(imagen_ruido, cmap="gray", vmin=0, vmax=1)
plt.title("Ruido Poisson")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(imagen_filtrada, cmap="gray", vmin=0, vmax=1)
plt.title("Gaussiano sigma=1.3")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/prueba_gaussiano.png",
    dpi=150
)

plt.show()


# curvas rmse
plt.figure(figsize=(9, 6))

plt.plot(
    sigma_values,
    rmse_background_values,
    label="Background"
)

plt.plot(
    sigma_values,
    rmse_square_values,
    label="Square"
)

plt.plot(
    sigma_values,
    rmse_circle_values,
    label="Circle"
)

plt.plot(
    sigma_values,
    rmse_global_values,
    label="Global"
)

plt.scatter(
    sigma_min_background,
    min_background,
    marker="o"
)

plt.scatter(
    sigma_min_square,
    min_square,
    marker="o"
)

plt.scatter(
    sigma_min_circle,
    min_circle,
    marker="o"
)

plt.scatter(
    sigma_min_global,
    min_global,
    marker="x",
    s=70
)

datos_rmse = np.column_stack((
    sigma_values,
    rmse_global_values,
    rmse_background_values,
    rmse_square_values,
    rmse_circle_values
))

np.savetxt(
    "resultados/pregunta1/rmse_sigma.csv",
    datos_rmse,
    delimiter=",",
    header="sigma,global,background,square,circle",
    comments=""
)



plt.xlabel("Sigma")
plt.ylabel("RMSE")
plt.title("RMSE segun sigma")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/rmse_sigma.png",
    dpi=150
)

plt.show()