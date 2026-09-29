# T2 Procesamiento de Imágenes

Nicolas Sanchez

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
└── resultados/
    └── pregunta1/
```

El código de cada pregunta se encuentra dentro de su carpeta correspondiente y los resultados generados se guardan en el directorio `resultados/`.

## Pregunta 1

En la p1 se trabaja con una imagen sintética con tres regiones de distinta intensidad. 
Se agrega ruido Poisson con `N = 40` y después se prueba un filtro Gaussiano para distintos valores de sigma. 
Se calcula el RMSE global y por región para encontrar los mejores valores de sigma. 
Después se construye un mapa de sigma a partir de una estimación local de intensidad y se aplica un filtro Gaussiano adaptativo. 
También se compara el filtro adaptativo con el mejor filtro Gaussiano global y se revisa el comportamiento cerca de los bordes.

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

### Resultados principales

En la carpeta `resultados/pregunta1/` se guardan las imágenes y archivos CSV usados para analizar los resultados. Algunos de los archivos generados son:

- `rmse_sigma.png`
- `rmse_sigma.csv`
- `mapa_sigma.png`
- `comparacion_global_adaptativo.png`
- `comparacion_rmse.csv`
- `perfil_bordes.png`
- `kernels_pixeles.png`

> **Nota:** Se utilizó una semilla fija para el ruido Poisson con el objetivo de asegurar la reproducibilidad de los resultados.