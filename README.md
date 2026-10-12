# Reducción de dimensionalidad: PCA frente a SVD, t-SNE y UMAP

Material de apoyo para la semana 09 del curso de Data Mining (Sistemas de recomendación, parte 1). Responde a la actividad de la sesión sobre modelos de factores latentes: explicar la técnica, sus ventajas, sus casos de uso, interpretar las fórmulas y preparar una infografía.

**Visualizador interactivo:** [https://ricardoamiel.github.io/PCA_vs_SVD_Tsne_Umap/](https://USUARIO.github.io/PCA_vs_SVD_Tsne_Umap/).

## Contenido

| Carpeta o archivo                        | Qué contiene                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| visualizador                             | Herramienta con d3 v7 para recorrer cada algoritmo paso a paso; se publica en GitHub Pages.                         |
| implementacion/reduccion.py              | PCA, SVD, t-SNE y UMAP implementados desde cero con NumPy, sin llamar a np.linalg.svd ni a np.linalg.eig.           |
| notebook/reduccion_dimensionalidad.ipynb | Notebook ejecutado con los ejemplos numéricos, las tres aplicaciones de la SVD y la comparación con t-SNE y UMAP. |
| infografia                               | Póster científico A1 vertical en SVG y PDF, completamente vectorial.                                              |
| figs                                     | Figuras del notebook en PDF y SVG.                                                                                  |
| papers/LEEME.md                          | Índice comentado de los artículos de referencia, con sus enlaces.                                                 |

## Visualizador

Tiene cinco pestañas: PCA, SVD geometría, SVD compresión, t-SNE y UMAP. En cada una se avanza con el botón Siguiente o haciendo clic en un paso. En PCA se pueden arrastrar o agregar puntos; en SVD geometría se puede editar la matriz; en t-SNE y UMAP se ajustan la perplejidad, el número de vecinos y la distancia mínima. Funciona también abriendo visualizador/index.html directamente en el navegador, con conexión a internet para cargar d3.

### Publicación en GitHub Pages

GitHub Pages no detecta la carpeta visualizador por sí solo: con la opción de publicar desde una rama solo admite la raíz o la carpeta docs. Por eso el repositorio incluye el flujo .github/workflows/pages.yml, que publica esa carpeta. Pasos:

1. Subir el repositorio a GitHub con la rama main.
2. En el repositorio, abrir Settings, luego Pages, y en Build and deployment elegir Source: GitHub Actions.
3. Ir a la pestaña Actions y ejecutar el flujo Publicar visualizador en GitHub Pages con Run workflow (solo la primera vez; después se ejecuta solo en cada push que modifique el visualizador).
4. Al terminar, la dirección aparece en Settings, Pages, y en el resumen del flujo.

En la cuenta gratuita, GitHub Pages requiere que el repositorio sea público.

## Notebook

Desde la carpeta notebook:

    jupyter nbconvert --to notebook --execute --inplace reduccion_dimensionalidad.ipynb

La ejecución tarda cerca de 30 segundos y vuelve a escribir las figuras en la carpeta figs.

## Resultados principales

1. PCA de un ejemplo de cinco puntos: autovalores 9 y 1, la primera componente explica el 90 % de la varianza y el error de reconstrucción con una componente es 4, igual a (n − 1) λ₂.
2. SVD de una matriz de 2 × 3: valores singulares 5 y 3; la aproximación de rango 1 tiene error de Frobenius 3, igual a σ₂.
3. Relación exacta: la SVD de los datos centrados del primer ejemplo tiene σ = 6 y 2, y σ² / (n − 1) devuelve los autovalores 9 y 1.
4. Compresión de una imagen de 512 × 512: con k = 50 se guarda el 19,6 % de los números con un error relativo de 6,4 %.
5. Recomendación con SVD de rango 2 sobre una matriz con 30 % de calificaciones ocultas: el RMSE baja de 1,27 a 0,43.
6. Dígitos del 0 al 5 en dos dimensiones: coeficiente de silueta de 0,33 con PCA, 0,57 con t-SNE y 0,78 con UMAP.

## Dependencias

Python 3 con NumPy, Matplotlib, scikit-learn, scikit-image y Jupyter.
