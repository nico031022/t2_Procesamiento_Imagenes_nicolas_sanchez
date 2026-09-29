import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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


def calcular_rmse(imagen_ref, imagen_test, mask=None):

    if mask is None:
        diferencia = imagen_ref - imagen_test
    else:
        diferencia = imagen_ref[mask] - imagen_test[mask]

    error = np.sqrt(np.mean(diferencia**2))

    return error

def filtro_gaussiano_adaptativo(imagen, mapa_sigma):

    # radio para sigma mayor
    sigma_max = np.max(mapa_sigma)
    radio_max = int(np.ceil(3 * sigma_max))

    # bordes symmetric
    imagen_pad = np.pad(
        imagen,
        radio_max,
        mode="symmetric"
    )

    imagen_salida = np.zeros_like(imagen)

    # coordenadas kernel
    x = np.arange(-radio_max, radio_max + 1)
    y = np.arange(-radio_max, radio_max + 1)

    xx, yy = np.meshgrid(x, y)

    for fila in range(imagen.shape[0]):
        for columna in range(imagen.shape[1]):

            sigma_pixel = mapa_sigma[fila, columna]

            kernel = np.exp(
                -(xx**2 + yy**2) / (2 * sigma_pixel**2)
            )

            kernel = kernel / np.sum(kernel)

            region = imagen_pad[
                fila:fila + 2 * radio_max + 1,
                columna:columna + 2 * radio_max + 1
            ]

            valor = np.sum(region * kernel)

            imagen_salida[fila, columna] = valor

    return imagen_salida


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

    sigma = round(float(sigma), 1)

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

#----------------------

# estimar intensidad local
sigma_mu = 2.0

kernel_mu = crear_kernel_gaussiano(sigma_mu)

mu_local = convolve2d(
    imagen_ruido,
    kernel_mu,
    mode="same",
    boundary="symm"
)

# valores obtenidos del barrido
intensidades_ref = [0.15, 0.45, 0.80]

sigmas_ref = [
    sigma_min_background,
    sigma_min_square,
    sigma_min_circle
]
# crear mapa sigma
sigma_map = np.interp(
    mu_local,
    intensidades_ref,
    sigmas_ref
)

# check intensidad local
mu_background = np.mean(mu_local[mask_background])
mu_square = np.mean(mu_local[mask_square])
mu_circle = np.mean(mu_local[mask_circle])

print("\nIntensidad local estimada")
print("Background:", mu_background)
print("Square:", mu_square)
print("Circle:", mu_circle)


sigma_background = np.mean(sigma_map[mask_background])
sigma_square = np.mean(sigma_map[mask_square])
sigma_circle = np.mean(sigma_map[mask_circle])

print("\nSigma promedio del mapa")
print("Background:", sigma_background)
print("Square:", sigma_square)
print("Circle:", sigma_circle)

# filtro adaptativo
imagen_adaptativa = filtro_gaussiano_adaptativo(
    imagen_ruido,
    sigma_map
)

print("\nFiltro adaptativo")
print("Sigma minimo:", np.min(sigma_map))
print("Sigma maximo:", np.max(sigma_map))
print("Tamano imagen:", imagen_adaptativa.shape)

#--------------------------------------------------------
# mejor filtro global
kernel_best_global = crear_kernel_gaussiano(
    sigma_min_global
)

imagen_best_global = convolve2d(
    imagen_ruido,
    kernel_best_global,
    mode="same",
    boundary="symm"
)

# rmse adaptativo
rmse_adapt_global = calcular_rmse(
    imagen_ideal,
    imagen_adaptativa
)

rmse_adapt_background = calcular_rmse(
    imagen_ideal,
    imagen_adaptativa,
    mask_background
)

rmse_adapt_square = calcular_rmse(
    imagen_ideal,
    imagen_adaptativa,
    mask_square
)

rmse_adapt_circle = calcular_rmse(
    imagen_ideal,
    imagen_adaptativa,
    mask_circle
)

# rmse mejor global
rmse_best_global = calcular_rmse(
    imagen_ideal,
    imagen_best_global
)

rmse_best_background = calcular_rmse(
    imagen_ideal,
    imagen_best_global,
    mask_background
)

rmse_best_square = calcular_rmse(
    imagen_ideal,
    imagen_best_global,
    mask_square
)

rmse_best_circle = calcular_rmse(
    imagen_ideal,
    imagen_best_global,
    mask_circle
)


print("\nComparacion final")

print("\nSin filtro")
print("Global:", rmse_ruido_global)
print("Background:", rmse_ruido_background)
print("Square:", rmse_ruido_square)
print("Circle:", rmse_ruido_circle)

print("\nMejor Gaussiano global")
print("Sigma:", sigma_min_global)
print("Global:", rmse_best_global)
print("Background:", rmse_best_background)
print("Square:", rmse_best_square)
print("Circle:", rmse_best_circle)

print("\nGaussiano adaptativo")
print("Global:", rmse_adapt_global)
print("Background:", rmse_adapt_background)
print("Square:", rmse_adapt_square)
print("Circle:", rmse_adapt_circle)



# diferencia adaptativo - global
dif_global = rmse_adapt_global - rmse_best_global
dif_background = rmse_adapt_background - rmse_best_background
dif_square = rmse_adapt_square - rmse_best_square
dif_circle = rmse_adapt_circle - rmse_best_circle

print("\nDiferencia adaptativo - global")
print("Global:", dif_global)
print("Background:", dif_background)
print("Square:", dif_square)
print("Circle:", dif_circle)



datos_comparacion = np.array([
    [
        rmse_ruido_global,
        rmse_ruido_background,
        rmse_ruido_square,
        rmse_ruido_circle
    ],
    [
        rmse_best_global,
        rmse_best_background,
        rmse_best_square,
        rmse_best_circle
    ],
    [
        rmse_adapt_global,
        rmse_adapt_background,
        rmse_adapt_square,
        rmse_adapt_circle
    ]
])

np.savetxt(
    "resultados/pregunta1/comparacion_rmse.csv",
    datos_comparacion,
    delimiter=",",
    header="global,background,square,circle",
    comments=""
)
#-----------------------------------------------------


# perfil horizontal
fila_perfil = centro_y

x_perfil = np.arange(tamano)

perfil_ideal = imagen_ideal[fila_perfil, :]
perfil_ruido = imagen_ruido[fila_perfil, :]
perfil_global = imagen_best_global[fila_perfil, :]
perfil_adaptativo = imagen_adaptativa[fila_perfil, :]

perfil_sigma = sigma_map[fila_perfil, :]


# check sigma cerca de bordes
print("\nSigma cerca de bordes")

print("x=60:", perfil_sigma[60])
print("x=64:", perfil_sigma[64])
print("x=68:", perfil_sigma[68])

print("x=92:", perfil_sigma[92])
print("x=96:", perfil_sigma[96])
print("x=100:", perfil_sigma[100])

#------------------------------

# pixeles representativos
pixel_background = (32, 32)
pixel_square = (80, 80)
pixel_circle = (128, 128)


# mismo soporte de filtro adaptativo
sigma_max = np.max(sigma_map)
radio_max = int(np.ceil(3 * sigma_max))

x_kernel = np.arange(-radio_max, radio_max + 1)
y_kernel = np.arange(-radio_max, radio_max + 1)

xx_kernel, yy_kernel = np.meshgrid(
    x_kernel,
    y_kernel
)

# pixel background
fila = pixel_background[0]
columna = pixel_background[1]

mu_pixel_background = mu_local[fila, columna]
sigma_pixel_background = sigma_map[fila, columna]

kernel_background = np.exp(
    -(xx_kernel**2 + yy_kernel**2) /
    (2 * sigma_pixel_background**2)
)

kernel_background = (
    kernel_background /
    np.sum(kernel_background)
)

# pixel square
fila = pixel_square[0]
columna = pixel_square[1]

mu_pixel_square = mu_local[fila, columna]
sigma_pixel_square = sigma_map[fila, columna]

kernel_square = np.exp(
    -(xx_kernel**2 + yy_kernel**2) /
    (2 * sigma_pixel_square**2)
)

kernel_square = (
    kernel_square /
    np.sum(kernel_square)
)

# pixel circle
fila = pixel_circle[0]
columna = pixel_circle[1]

mu_pixel_circle = mu_local[fila, columna]
sigma_pixel_circle = sigma_map[fila, columna]

kernel_circle = np.exp(
    -(xx_kernel**2 + yy_kernel**2) /
    (2 * sigma_pixel_circle**2)
)

kernel_circle = (
    kernel_circle /
    np.sum(kernel_circle)
)


print("\nPixeles representativos")

print("\nBackground", pixel_background)
print("mu =", mu_pixel_background)
print("sigma =", sigma_pixel_background)
print("kernel shape =", kernel_background.shape)
print("kernel suma =", np.sum(kernel_background))

print("\nSquare", pixel_square)
print("mu =", mu_pixel_square)
print("sigma =", sigma_pixel_square)
print("kernel shape =", kernel_square.shape)
print("kernel suma =", np.sum(kernel_square))

print("\nCircle", pixel_circle)
print("mu =", mu_pixel_circle)
print("sigma =", sigma_pixel_circle)
print("kernel shape =", kernel_circle.shape)
print("kernel suma =", np.sum(kernel_circle))


print("\nKernel background")
print(
    np.array2string(
        kernel_background,
        precision=5,
        suppress_small=True
    )
)

print("\nKernel square")
print(
    np.array2string(
        kernel_square,
        precision=5,
        suppress_small=True
    )
)

print("\nKernel circle")
print(
    np.array2string(
        kernel_circle,
        precision=5,
        suppress_small=True
    )
)



































































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


# mostrar mapas
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(imagen_ruido, cmap="gray", vmin=0, vmax=1)
plt.title("Imagen con ruido")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mu_local, cmap="gray", vmin=0, vmax=1)
plt.title("Estimacion local")
plt.axis("off")

plt.subplot(1, 3, 3)
imagen_sigma = plt.imshow(sigma_map, cmap="viridis")
plt.title("Mapa sigma")
plt.axis("off")

plt.colorbar(
    imagen_sigma,
    fraction=0.046,
    pad=0.04
)

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/mapa_sigma.png",
    dpi=150
)

plt.show()


# mostrar funcion F
mu_values = np.linspace(0.10, 0.90, 200)

sigma_interpolado = np.interp(
    mu_values,
    intensidades_ref,
    sigmas_ref
)

plt.figure(figsize=(7, 5))

plt.plot(mu_values, sigma_interpolado)

plt.scatter(
    intensidades_ref,
    sigmas_ref
)

plt.xlabel("Intensidad local")
plt.ylabel("Sigma")
plt.title("Funcion sigma = F(mu)")

plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/funcion_sigma.png",
    dpi=150
)

plt.show()

# mostrar filtro adaptativo
plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
plt.imshow(
    imagen_ruido,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Imagen con ruido")
plt.axis("off")


plt.subplot(1, 3, 2)
mapa_plot = plt.imshow(
    sigma_map,
    cmap="viridis"
)
plt.title("Mapa sigma")
plt.axis("off")

plt.colorbar(
    mapa_plot,
    fraction=0.046,
    pad=0.04
)


plt.subplot(1, 3, 3)
plt.imshow(
    imagen_adaptativa,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Filtro adaptativo")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/filtro_adaptativo.png",
    dpi=150
)

plt.show()




# comparacion final
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(
    imagen_ideal,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Imagen ideal")
plt.axis("off")


plt.subplot(1, 4, 2)
plt.imshow(
    imagen_ruido,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Imagen con ruido")
plt.axis("off")


plt.subplot(1, 4, 3)
plt.imshow(
    imagen_best_global,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title(
    "Gaussiano global sigma="
    + str(sigma_min_global)
)
plt.axis("off")


plt.subplot(1, 4, 4)
plt.imshow(
    imagen_adaptativa,
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("Gaussiano adaptativo")
plt.axis("off")


plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/comparacion_global_adaptativo.png",
    dpi=150
)

plt.show()


#---------------------------------------
# perfil de intensidad
plt.figure(figsize=(11, 5))

plt.plot(
    x_perfil,
    perfil_ideal,
    label="Ideal"
)

plt.plot(
    x_perfil,
    perfil_ruido,
    label="Ruido",
    alpha=0.5
)

plt.plot(
    x_perfil,
    perfil_global,
    label="Global"
)

plt.plot(
    x_perfil,
    perfil_adaptativo,
    label="Adaptativo"
)


# bordes
plt.axvline(64, linestyle="--", alpha=0.4)
plt.axvline(96, linestyle="--", alpha=0.4)
plt.axvline(160, linestyle="--", alpha=0.4)
plt.axvline(192, linestyle="--", alpha=0.4)


plt.xlabel("Posicion x")
plt.ylabel("Intensidad")
plt.title("Perfil horizontal por el centro")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/perfil_bordes.png",
    dpi=150
)

plt.show()

# perfil sigma
plt.figure(figsize=(11, 4))

plt.plot(
    x_perfil,
    perfil_sigma
)

plt.axvline(64, linestyle="--", alpha=0.4)
plt.axvline(96, linestyle="--", alpha=0.4)
plt.axvline(160, linestyle="--", alpha=0.4)
plt.axvline(192, linestyle="--", alpha=0.4)

plt.xlabel("Posicion x")
plt.ylabel("Sigma")
plt.title("Perfil del mapa sigma")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/perfil_sigma.png",
    dpi=150
)

plt.show()

# zoom de bordes
plt.figure(figsize=(12, 4))


# borde background-square
plt.subplot(1, 2, 1)

plt.plot(
    x_perfil,
    perfil_ideal,
    label="Ideal"
)

plt.plot(
    x_perfil,
    perfil_global,
    label="Global"
)

plt.plot(
    x_perfil,
    perfil_adaptativo,
    label="Adaptativo"
)

plt.xlim(50, 80)
plt.ylim(0.05, 0.55)

plt.axvline(64, linestyle="--", alpha=0.4)

plt.title("Borde background-square")
plt.xlabel("Posicion x")
plt.ylabel("Intensidad")
plt.grid(alpha=0.3)


# borde square-circle
plt.subplot(1, 2, 2)

plt.plot(
    x_perfil,
    perfil_ideal,
    label="Ideal"
)

plt.plot(
    x_perfil,
    perfil_global,
    label="Global"
)

plt.plot(
    x_perfil,
    perfil_adaptativo,
    label="Adaptativo"
)

plt.xlim(82, 110)
plt.ylim(0.30, 0.90)

plt.axvline(96, linestyle="--", alpha=0.4)

plt.title("Borde square-circle")
plt.xlabel("Posicion x")
plt.ylabel("Intensidad")
plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "resultados/pregunta1/zoom_bordes.png",
    dpi=150
)

plt.show()
#-------------------


# kernels de pixeles representativos
plt.figure(figsize=(12, 4))

max_kernel = max(
    np.max(kernel_background),
    np.max(kernel_square),
    np.max(kernel_circle)
)


plt.subplot(1, 3, 1)
plt.imshow(
    kernel_background,
    cmap="viridis",
    vmin=0,
    vmax=max_kernel
)
plt.title(
    "Background\nsigma="
    + str(round(sigma_pixel_background, 3))
)
plt.axis("off")


plt.subplot(1, 3, 2)
plt.imshow(
    kernel_square,
    cmap="viridis",
    vmin=0,
    vmax=max_kernel
)
plt.title(
    "Square\nsigma="
    + str(round(sigma_pixel_square, 3))
)
plt.axis("off")


plt.subplot(1, 3, 3)
imagen_kernel = plt.imshow(
    kernel_circle,
    cmap="viridis",
    vmin=0,
    vmax=max_kernel
)
plt.title(
    "Circle\nsigma="
    + str(round(sigma_pixel_circle, 3))
)
plt.axis("off")


plt.colorbar(
    imagen_kernel,
    ax=plt.gcf().axes,
    fraction=0.025,
    pad=0.02
)

plt.savefig(
    "resultados/pregunta1/kernels_pixeles.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()




