# Ethnical-biases-in-face-detection

Tema: **Sesgos Etnicos en Clasificadores de Rostros**

Integrantes: Jorge Fabry y Jose Ignacio Musso


La red consta de tres archivos principales: importar_imagenes.py, cifar10mod.py y convnet.py

**importar_imagenes.py:**

Cumple la funcion de recibir los archivos de imagenes del dataset UTKFace y las comprime en 4 archivos de tipo pkl que seran usados por cifar10mod para alimentar la red. En las lineas 66 y 67 se encuentran las variables razamod y porcentaje, estas sirven para crear un desbalance en el dataset a comprimir e.g. razamod=1 y porcentaje=50 el dataset resultante tendra una presencia de raza Negros de solo un 50%.

**cifar10mod.py:**

Este archivo es ejecutado por convnet.py, no sirve ejecutarlo solo. En la linea 32 esta la variable DIR_BINARIES que debe especificar la carpeta donde se encuentran los archivos pkl generados por importar_imagenes.py

**convnet.py:**

Este archivo contiene la red convolucional. Tiene todo listo para llegar y ejecutar.

INSTRUCCIONES DE USO:

**SI SE TIENEN SOLO 8GB DE RAM SE RECOMIENDA CERRAR TODO (INCLUIDO CHROME) PARA EVITAR MEMORY ERROR**

1.  Descargar la base de datos UTKFace del siguiente link: https://drive.google.com/file/d/1xt1jX7Csq--6lxi7X7vOTyyTN6lj0eq6/view?usp=sharing
2.  Descomprimir el .zip y colocar la carpeta UTKFace en la carpeta donde se encuentran los tres archivos principales del           programa
3.  Ejecutar importar_imagenes.py para generar los archivos .pkl.
4.  Colocar los archivos .pkl dentro de una nueva carpeta Datos/
5.  Ejecutar convnet.py para comenzar el entrenamiento
6.  Finalmente se imprimen las matrices de confusion correspondientes en consola y numerosas figuras informativas
