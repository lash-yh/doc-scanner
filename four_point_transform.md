# four_point_transform

## Errores confirmados heredados del ordenamiento

La función obtiene las esquinas mediante `_order_points`. Si esa función
duplica u omite un vértice, la transformación utiliza correspondencias
degeneradas y no rectifica correctamente el cuadrilátero.

Se confirmaron dos entradas que provocan este problema:

- Un rombo con empates entre sumas o diferencias de coordenadas:
  `[(5, 0), (10, 5), (5, 10), (0, 5)]`.
- Un rectángulo válido representado como array `uint8`:
  `np.array([(1, 2), (8, 2), (8, 6), (1, 6)], dtype=np.uint8)`.

El primer caso produjo una salida de forma `(10, 7)` con solo cuatro valores
distintos al transformar una imagen de prueba con valores únicos. Los
detalles de ambos fallos se documentan en [_order_points.md](_order_points.md).

## Comportamientos que conviene definir

### Cantidad y validez geométrica de los puntos

No se exige que existan exactamente cuatro vértices distintos y no
colineales. En las pruebas se aceptaron tres y cinco puntos, así como
puntos colineales y cuatro puntos idénticos.

Con `image = np.arange(400, dtype=np.uint16).reshape(20, 20)` y
`pts = [(2, 2)] * 4`, la función devolvió una imagen constante de forma
`(20, 20)`, en lugar de rechazar la geometría degenerada.

Definir qué geometrías son válidas y validar esas condiciones antes de
calcular la transformación.

### Dimensiones mínimas y redondeo

El ancho y el alto se calculan a partir de distancias entre esquinas y se
truncan con `int`. Un rectángulo de lados `0.5` produjo dimensiones
calculadas de cero; OpenCV devolvió una salida del tamaño de la imagen
original. Un rectángulo con lados de un píxel produjo una salida `(1, 1)`
con correspondencias de destino degeneradas.

Definir las dimensiones mínimas admitidas y la regla para convertir
distancias geométricas en cantidades de píxeles.

### Coordenadas fuera de la imagen

Se aceptan esquinas exteriores y aparecen zonas con relleno negro en
la salida.

Definir si deben permitirse, rechazarse o ajustarse al área de la imagen,
y si el relleno debe poder configurarse.

### Entradas vacías y coordenadas no finitas

Una lista de puntos vacía genera `AxisError` desde `_order_points`.
Las coordenadas con `NaN` pueden terminar en
`ValueError: cannot convert float NaN to integer`.

Definir una validación explícita y excepciones consistentes para estos casos.

## Casos regulares verificados

Pasaron las pruebas de permutaciones de un cuadrilátero sin empates,
correspondencia de las cuatro esquinas e interpolación sobre una imagen
de coordenadas. También se comprobó la conservación de imágenes en gris,
RGB y RGBA, y de los tipos `uint8`, `uint16`, `float32` y `float64` probados.
