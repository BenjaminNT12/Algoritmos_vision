def saltar_numeros(iteraciones, numeros_a_saltar):
    contador = 0
    numero_actual = 0

    while contador < iteraciones:
        numero_actual += 1

        if numero_actual in numeros_a_saltar:
            continue

        print(numero_actual)
        contador += 1
    return numero_actual

iteraciones = 10
numeros_a_saltar = [3, 5, 7]

ultimo = saltar_numeros(iteraciones, numeros_a_saltar)
print("ultimo", ultimo)