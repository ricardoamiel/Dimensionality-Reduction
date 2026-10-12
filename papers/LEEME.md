# Papers de apoyo

Lectura sugerida en orden: primero el 02 y el 07 para la base, luego el 03 para la relación entre PCA y SVD, el 05 para recomendación, y el 08 y el 10 para los métodos no lineales.

| Archivo | Referencia | Para qué sirve |
|---|---|---|
| 01_Pearson_1901 | Pearson, K. (1901). On lines and planes of closest fit to systems of points in space. | Artículo fundacional de PCA como recta de mínimo error. |
| 02_Shlens_2014 | Shlens, J. (2014). A tutorial on principal component analysis. arXiv:1404.1100. | El tutorial más claro de PCA, con la relación con SVD. |
| 03_Wall_2003 | Wall, Rechtsteiner y Rocha (2003). Singular value decomposition and principal component analysis. | Capítulo breve que explica cómo PCA se obtiene de la SVD. |
| 04_Halko_2011 | Halko, Martinsson y Tropp (2011). Finding structure with randomness. SIAM Review. | SVD aleatorizada, el método que usa scikit-learn con matrices grandes. |
| 05_Koren_2009 | Koren, Bell y Volinsky (2009). Matrix factorization techniques for recommender systems. IEEE Computer. | Factores latentes en recomendación; el modelo ganador del premio Netflix. |
| 06_MMDS_cap09 | Leskovec, Rajaraman y Ullman. Mining of Massive Datasets, capítulo 9. | Sistemas de recomendación, base de las diapositivas de la sesión. |
| 07_MMDS_cap11 | Leskovec, Rajaraman y Ullman. Mining of Massive Datasets, capítulo 11. | Reducción de dimensionalidad: autovectores, PCA, SVD y CUR. |
| 08_vanderMaaten_Hinton_2008 | van der Maaten y Hinton (2008). Visualizing data using t-SNE. JMLR. | Artículo original de t-SNE. |
| 09_Kobak_Berens_2019 | Kobak y Berens (2019). The art of using t-SNE for single-cell transcriptomics. Nature Communications. | Buenas prácticas para usar t-SNE: inicialización, perplejidad y escalado. |
| 10_McInnes_2018 | McInnes, Healy y Melville (2018). UMAP. arXiv:1802.03426. | Artículo original de UMAP, con fundamento teórico y comparaciones. |
| 11_Malkov_2018 | Malkov y Yashunin (2018). Hierarchical navigable small world graphs. IEEE TPAMI. | Artículo original de HNSW. |
| 12_Johnson_2017 | Johnson, Douze y Jégou (2017). Billion-scale similarity search with GPUs. | IVF y cuantización de producto a gran escala (FAISS). |
| 13_Douze_2024 | Douze y colaboradores (2024). The Faiss library. arXiv:2401.08281. | Panorama actual de índices IVF, PQ y HNSW. |

Los PDF no se suben al repositorio porque tienen derechos de sus editoriales; cada referencia se puede buscar por su título o identificador arXiv. Dos de ellas, además, se consultan en línea en estas direcciones:

1. Jolliffe y Cadima (2016). Principal component analysis: a review and recent developments. [doi.org/10.1098/rsta.2015.0202](https://doi.org/10.1098/rsta.2015.0202), de acceso abierto.
2. Deerwester y colaboradores (1990). Indexing by latent semantic analysis. [doi.org/10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9](https://doi.org/10.1002/(SICI)1097-4571(199009)41:6%3C391::AID-ASI1%3E3.0.CO;2-9).
