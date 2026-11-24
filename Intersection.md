# Intersection

Diagnóstico de la clase y sus métodos en [src/core/entities.py](src/core/entities.py).
Las importaciones de `math_utils` quedan fuera de esta revisión.

## Errores confirmados

### `__init__`: evaluación booleana de una imagen NumPy

La condición `elif image` intenta convertir un array de varios elementos a
un booleano. Construir la clase con una imagen de forma `(20, 30)` genera:

```text
ValueError: The truth value of an array with more than one element is ambiguous.
```

Esto impide construir normalmente la instancia con una imagen. La condición
debe distinguir la ausencia de imagen mediante `image is not None`.

Prueba: `ConstructionTests.test_accepts_numpy_image`.

### `__init__` y `__corner_points`: ancho escalar incompatible

Pasar `x=20` termina en `TypeError: len() of unsized object`, porque
`__corner_points` entrega el escalar a `find_points_on_lines`, que espera
una secuencia de coordenadas X.

La ruta que recibe una imagen también asigna un escalar a `self.x` mediante
`image.shape[1]`. Por tanto, corregir solamente la condición booleana no
resuelve la incompatibilidad de esa ruta.

Con dos abscisas, como `x=(0, 20)`, la construcción probada funciona para
líneas no verticales exactas.

Prueba: `ConstructionTests.test_accepts_scalar_width`.

### `__corner_points`: extremos infinitos para una vertical exacta

Para `line_v=(0, 10)`, que representa `x=10`, se intenta obtener Y a partir
de X. El cálculo de la función auxiliar divide entre `sin(0)` y los extremos
verticales contienen `-inf` e `inf`.

La clase conserva esos extremos sin rechazarlos ni utilizar otra forma de
muestreo. No es posible emplearlos como extremos geométricos finitos.

Prueba: `ConstructionTests.test_axis_aligned_vertical_line_has_finite_corners`.

### `__corner_points` y `orientation`: direcciones incompatibles

`ORDER` declara `top, right, bottom, left`, pero `__corner_points` devuelve
`left, right, top, bottom`. `connectivity` conserva este último orden.

`orientation` utiliza los pares `(0, 1)`, `(0, 3)`, `(2, 3)` y `(2, 1)`.
Con el orden real, el primer par une izquierda y derecha; el tercero une
arriba y abajo. Ambos son pares de rayos opuestos, no esquinas.

Al aislar el cálculo con conectividad `(1, 1, 0, 0)`, los resultados son
`[2, 1, 0, 1]`. Una línea que continúa hacia izquierda y derecha obtiene
una puntuación máxima de esquina aunque no exista conexión vertical.

Pruebas: `ConstructionTests.test_corners_follow_declared_order` y
`OrientationAndSummaryTests.test_opposite_rays_do_not_form_a_corner`.

### `connectivity`: coordenadas X e Y intercambiadas

Los píxeles interpolados tienen la convención `(x, y)`, mientras que una
imagen NumPy se consulta mediante `[y, x]`. El método utiliza
`self.image[pixel[0], pixel[1]]`, intercambiando fila y columna.

Con un único píxel encendido en `image[2, 7]` y muestras `(7, 2)`, se
obtiene conectividad cero en lugar de uno. En imágenes rectangulares,
también pueden descartarse muestras válidas por índices fuera de rango.

Prueba: `ConnectivityDiagnosticTests.test_samples_xy_as_image_yx`.

### `connectivity`: índices negativos leen el borde opuesto

Capturar `IndexError` no basta para filtrar muestras exteriores. NumPy
acepta índices negativos y accede a posiciones desde el final del array.

En la prueba, las muestras `(-1, 2)` y `(2, -1)` leen píxeles encendidos
del borde opuesto. Junto a una muestra interior apagada, producen
conectividad `2/3`, aunque ninguna muestra interior esté encendida.

Es necesario comprobar explícitamente los límites de ambos ejes.

Prueba: `ConnectivityDiagnosticTests.test_negative_coordinates_do_not_wrap_around_image`.

### `connectivity`: división entre cero cuando no hay muestras válidas

Si todas las muestras quedan fuera de la imagen, `hits` y `longth`
permanecen en cero. La división final devuelve `(NaN, NaN, NaN, NaN)`.
Estos valores pueden propagarse a la orientación y al resumen.

La prueba propone devolver cero; la respuesta pública para la ausencia de
muestras debe definirse antes de fijar esa expectativa como contrato.

Prueba: `ConnectivityDiagnosticTests.test_no_valid_samples_produce_finite_zero_connectivity`.

### `orientation`: caché desactualizada después de recalcular conectividad

Cambiar `width` fuerza el recálculo de `__connectivity`, pero no invalida
`__orientation`. Las llamadas posteriores a `orientation` devuelven las
puntuaciones anteriores.

La prueba obtiene inicialmente `[2, 2, 2, 2]`, cambia el ancho y confirma
una conectividad nueva de `[0, 0, 0, 0]`. La orientación permanece en
`[2, 2, 2, 2]`.

Prueba: `ConnectivityDiagnosticTests.test_changed_width_recomputes_connectivity_and_orientation`.

## Comportamientos que conviene definir

### `__init__`: contrato de las entradas

Definir si `x` representa el ancho de imagen o una pareja de abscisas.
Si se admiten ambas formas, normalizarlas antes de calcular los extremos.
También debe decidirse qué entrada prevalece cuando se proporcionan
simultáneamente `x` e `image`.

Definir las dimensiones y los tipos de imagen admitidos, el tratamiento de
imágenes vacías y la validación de coordenadas y líneas finitas. La
validación actual de los puntos y las líneas comprueba su longitud, pero
no establece un contrato numérico completo.

### `__corner_points`: geometría y límites

Definir si los extremos son intersecciones con el borde de la imagen o
puntos de referencia sobre las líneas. Especificar cómo tratar verticales
exactas, intersecciones exteriores y extremos coincidentes con la
intersección, que implican una distancia cero en `connectivity`.

Unificar el orden de extremos, conectividad y etiquetas de orientación.

### `connectivity`: parámetros de muestreo

Sin `along_length`, el cálculo probado falla con `TypeError` al dividir
`None` entre una distancia. Sin imagen, falla con `TypeError` al intentar
indexar `None`. Definir si esos estados deben rechazarse con mensajes
explícitos o resolverse mediante valores predeterminados.

Los argumentos cero se ignoran por las condiciones `if along_length` e
`if width`. Los negativos probados se aceptan sin una validación explícita;
una longitud negativa puede invertir el sentido del muestreo.

Definir los rangos y tipos admitidos para ambos parámetros, el significado
del grosor y qué ocurre al redondear un segmento a longitud cero. La
limitación de la interpolación para extremos idénticos está documentada
en [interpolate_pixels_along_line.md](interpolate_pixels_along_line.md).

### `connectivity`: ausencia de muestras y denominador

Definir si las muestras exteriores se descartan o cuentan como ausencia
de borde. La implementación actual excluye los índices que provocan
`IndexError` del denominador, pero acepta índices negativos.

Cuando no queda ninguna muestra válida, decidir entre conectividad cero,
un estado explícito de ausencia de datos o una excepción controlada.

### `connectivity` y `orientation`: invalidación de resultados

Definir cuándo se invalidan ambas cachés: cambios en parámetros, imagen,
intersección o líneas. Decidir si se permite modificar directamente esos
atributos y si los cambios internos del array de imagen deben detectarse
o requieren una operación explícita de recálculo.

### `orientation`: significado de las etiquetas

Definir si `top-left` describe la posición de una esquina del documento
o las direcciones de sus rayos. Una esquina superior izquierda del
documento conecta hacia la derecha y hacia abajo; la interpretación
contraria cambia la correspondencia entre etiquetas y puntuaciones.

Definir también cómo se interpretan empates y puntuaciones insuficientes.
Actualmente se devuelven cuatro sumas, sin elegir una orientación.

### `summary` y `__repr__`: formato y errores heredados

El formato probado de ambos métodos funciona. `summary` depende de los
cálculos anteriores y puede propagar sus excepciones o mostrar `NaN`.
Definir cómo representar la ausencia de datos y si el formato, el orden
de las cifras y las dos modalidades de salida forman parte del contrato.

## Problema corregido desde la revisión inicial

El acceso a `self.intersections` en `connectivity` provocaba
`AttributeError`. La versión actual utiliza `self.intersection` y la
prueba `PublicConnectivityTests.test_connectivity_uses_stored_intersection`
ya pasa. No se considera un error pendiente.

## Verificación

Las pruebas están en [test/test_entities.py](test/test_entities.py).
Las pruebas de diagnóstico aíslan los bloqueos de construcción y algunas
usan muestras controladas para comprobar cada problema por separado.
Las expectativas de validación representan comportamientos propuestos,
no contratos que ya estén definidos en la implementación.

```bash
uv run python -m unittest test.test_entities -v
```

La ejecución sobre la versión actual realizó 22 pruebas y reportó
`failures=12, errors=4`. Los contadores incluyen fallos de subpruebas,
por lo que no equivalen a 16 métodos distintos fallidos.

Pasaron los casos probados de construcción con dos abscisas y líneas
inclinadas, pertenencia de los extremos a sus líneas, longitudes de
entradas inválidas, ausencia simultánea de imagen y abscisas, `__repr__`,
formato de `summary`, conectividad uniforme en imágenes llenas y vacías,
reutilización de caché con parámetros idénticos y acceso al atributo
corregido de la intersección. Los casos de muestreo aislado no validan
por sí solos el flujo completo desde el constructor.
