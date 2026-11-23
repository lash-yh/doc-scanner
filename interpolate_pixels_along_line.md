# interpolate_pixels_along_line

## Error confirmado: extremos idénticos

Cuando ambos puntos coinciden, `dx` y `dy` son cero. La operación
`gradient = dy / dx` impide completar la interpolación.

```python
interpolate_pixels_along_line((2, 3), (2, 3))
# ZeroDivisionError: division by zero
```

Con extremos de tipo `np.ndarray`, la división genera `NaN` y una advertencia;
la conversión posterior a enteros termina en `ValueError`.

```python
interpolate_pixels_along_line(np.array([2, 3]), np.array([2, 3]))
# ValueError: cannot convert float NaN to integer
```

La prueba existente `test_zero_length_segment_contains_its_pixel` espera que
la salida incluya el punto coincidente y actualmente termina con error.

## Comportamientos que conviene definir

### Significado y uniformidad de `width`

En el interior del segmento se generan `2 * width` píxeles por posición del
eje de recorrido, pero en cada extremo se generan únicamente dos.
Por ejemplo, para `(0, 0)` a `(5, 0)` con `width=3`, las cantidades por columna
son `2, 6, 6, 6, 6, 2`.

Definir si `width` representa el grosor total o un alcance lateral, y si
ese grosor debe mantenerse también en los extremos.

### Validación de `width`

- `width=0` y los valores negativos omiten los píxeles interiores y devuelven
  solo los correspondientes a los extremos.
- `width=1.5` genera `TypeError` cuando hay posiciones interiores, pero se
  acepta en segmentos cortos que no ejecutan ese bucle.

Definir los valores admitidos y una respuesta consistente para los inválidos.

### Orden de los píxeles

La función normaliza el sentido de recorrido intercambiando los extremos
cuando lo necesita. Por ello, la salida puede comenzar en `p2`.

Definir si el orden debe seguir siempre el sentido de `p1` a `p2` o conservar
el recorrido normalizado actual.

### Coordenadas decimales

Aunque las anotaciones de las tuplas indican coordenadas enteras, la función
acepta puntos decimales y devuelve coordenadas enteras. Utiliza `np.round`
para los extremos y `np.floor` para las posiciones interiores.

Definir si se permiten coordenadas decimales y, en ese caso, qué regla de
conversión a píxeles debe aplicarse.
