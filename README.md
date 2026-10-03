# T2 Procesamiento de Imágenes

Nicolas Sanchez

## Aclaración

La Pregunta 1 está completada, sin incluir el bonus.

En la Pregunta 2, por limitaciones de tiempo, no alcancé a completar todo lo pedido en el enunciado. Dejé implementadas algunas partes base y funciones necesarias para la difusión anisotrópica, junto con pruebas simples para revisar su funcionamiento.


## Estructura

Actualmente el proyecto tiene la siguiente estructura:

```text
t2_Procesamiento_Imagenes_nicolas_sanchez/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── pregunta1/
│   └── pregunta1.py
│
├── pregunta2/
│   └── pregunta2.ipynb
│
└── resultados/
    └── pregunta1/
```

El código de cada pregunta se encuentra dentro de su carpeta correspondiente. En la Pregunta 1 los resultados generados se guardan en el directorio resultados/.

## Pregunta 1

En la p1 se trabaja con una imagen sintética con tres regiones de distinta intensidad. 
Se agrega ruido Poisson con `N = 40` y después se prueba un filtro Gaussiano para distintos valores de sigma. 
Se calcula el RMSE global y por región para encontrar los mejores valores de sigma. 
Después se construye un mapa de sigma a partir de una estimación local de intensidad y se aplica un filtro Gaussiano adaptativo. 
También se compara el filtro adaptativo con el mejor filtro Gaussiano global y se revisa el comportamiento cerca de los bordes.

## Pregunta 2

Para la p2 decidí trabajar en un notebook porque después de hacer la Pregunta 1 en un solo archivo .py, se me hizo difícil mantener ordenadas las distintas partes, pruebas y figuras.

Por limitaciones de tiempo no alcancé a completar toda la pregunta. Alcancé a implementar algunas partes base para la difusión anisotrópica, incluyendo las diferencias finitas en cuatro direcciones, el cálculo del gradiente y Laplaciano, el coeficiente de difusión TV y una implementación inicial de la difusión anisotrópica para el caso TV.

Para revisar estas funciones usé una imagen sintética simple con un cuadrado en el centro. También hice algunas pruebas cambiando el número de iteraciones y el valor de epsilon para revisar el comportamiento de la implementación.


### Requisitos

El código fue desarrollado con Python. Las librerías usadas están en el archivo `requirements.txt`. 
Para instalarlas, ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```

### Ejecución


Desde la carpeta principal del proyecto, ejecuta:

```bash
python pregunta1/pregunta1.py
```

Los resultados se guardan automáticamente en la ruta:
`resultados/pregunta1/`

El archivo de la Pregunta 2 se encuentra en:

`pregunta2/pregunta2.ipynb`

El notebook se entrega con las celdas ejecutadas para poder ver directamente los resultados de las pruebas realizadas.

Para ejecutar la Pregunta 2, se puede abrir el notebook y ejecutar las celdas en orden desde el inicio.

### Resultados principales

En la carpeta `resultados/pregunta1/` se guardan las imágenes y archivos CSV usados para analizar los resultados. Algunos de los archivos generados son:

- `rmse_sigma.png`
- `rmse_sigma.csv`
- `mapa_sigma.png`
- `comparacion_global_adaptativo.png`
- `comparacion_rmse.csv`
- `perfil_bordes.png`
- `kernels_pixeles.png`

