# _order_points

## Errores confirmados

### Esquinas duplicadas cuando hay empates

La selección mediante mínimos y máximos de `x + y` y `y - x` puede asignar
un mismo punto a varias posiciones y omitir otro vértice.

```python
_order_points([(5, 0), (10, 5), (5, 10), (0, 5)])
# array([[ 5.,  0.],
#        [ 5.,  0.],
#        [10.,  5.],
#        [ 5., 10.]], dtype=float32)
```

El resultado contiene tres vértices distintos en lugar de cuatro. Al probar
las 24 permutaciones de este rombo se obtuvieron 14 ordenamientos diferentes.
La salida depende del orden de entrada cuando hay empates.

### Desbordamiento con coordenadas `uint8`

`np.diff` conserva el tipo sin signo de la entrada. Las diferencias negativas
se desbordan y cambian la selección de esquinas.

```python
pts = np.array([(1, 2), (8, 2), (8, 6), (1, 6)], dtype=np.uint8)
_order_points(pts)
# array([[1., 2.],
#        [1., 2.],
#        [8., 6.],
#        [8., 6.]], dtype=float32)
```

Los mismos valores como lista o array de enteros con signo producen
correctamente `[(1, 2), (8, 2), (8, 6), (1, 6)]`.

## Comportamientos que conviene definir

### Cantidad y forma de los puntos

No se exige una entrada de forma `(4, 2)`. Se aceptaron tres puntos,
duplicando una esquina, y cinco puntos, descartando el punto interior probado.
Una lista vacía genera `AxisError` durante el cálculo de sumas.

Definir si deben exigirse exactamente cuatro coordenadas bidimensionales
y qué excepción corresponde a entradas inválidas.

### Validez geométrica y valores numéricos

Se aceptan puntos repetidos y colineales. Las entradas con `NaN` también
pueden producir esquinas repetidas con `NaN`.

Definir si se requieren coordenadas finitas, vértices distintos y un
cuadrilátero convexo con área positiva.

### Convención de orden y tipo de salida

La intención actual es devolver las esquinas superior izquierda, superior
derecha, inferior derecha e inferior izquierda, en un array `float32`.
Las pruebas regulares pasaron con coordenadas negativas y fraccionarias,
y con enteros con signo y números de punto flotante.

Definir cómo resolver orientaciones ambiguas y si `float32` debe formar
parte del contrato público de la función.
