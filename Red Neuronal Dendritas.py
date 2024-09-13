import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tabulate import tabulate
# Función para construir, entrenar y evaluar el modelo
def build_train_evaluate_model(X_train, y_train, num_capas, num_neuronas_por_capa, num_epocas):
    model = Sequential()
    model.add(Dense(num_neuronas_por_capa[0], activation='relu', input_dim=X_train.shape[1]))  # Capa de entrada
    for i in range(1, num_capas):  # Capas ocultas
        model.add(Dense(num_neuronas_por_capa[i], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Capa de salida

    # Compilar el modelo
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=num_epocas, verbose=0)

    # Evaluar el modelo
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # Calcular predicciones
    y_pred = model.predict(X_train)

    # Imprimir resultados
    headers = [f'x{i+1}' for i in range(X_train.shape[1])] + ['y', 'y_cal']
    data = []
    for x, y_true, y_pred in zip(X_train, y_train, y_pred):
        row = [f'{val:.2f}' for val in x] + [f'{y_true:.2f}', f'{y_pred[0]:.2f}']
        data.append(row)

    # Imprimir tabla utilizando tabulate
    print(tabulate(data, headers=headers, tablefmt='fancy_grid'))

# Menú principal
while True:
    print("\nSeleccione una opción:")
    print("1. Función lógica AND")
    print("2. Función lógica OR")
    print("3. Función lógica XOR")
    print("4. Ejercicio 1")
    print("5. Mayoría simple")
    print("6. Paridad de 4 bits")
    print("7. Salir")
    opcion = input("Ingrese el número de la opción deseada: ")

    if opcion == "7":
        print("Saliendo...")
        break
    elif opcion in ["1", "2", "3", "4", "5", "6"]:
        num_capas = int(input("Ingrese el número de capas de la red neuronal: "))
        num_neuronas_por_capa = [int(input(f"Ingrese el número de neuronas para la capa {i+1}: ")) for i in range(num_capas)]
        num_epocas = int(input("Ingrese el número de épocas: "))

        if opcion == "1":  # AND
            X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
            y_and = np.array([0, 0, 0, 1], dtype="float32")
            build_train_evaluate_model(X_and, y_and, num_capas, num_neuronas_por_capa, num_epocas)

        elif opcion == "2":  # OR
            X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
            y_or = np.array([0, 1, 1, 1], dtype="float32")
            build_train_evaluate_model(X_or, y_or, num_capas, num_neuronas_por_capa, num_epocas)
        elif opcion == "3":  # XOR
            X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
            y_xor = np.array([0, 1, 1, 0], dtype="float32")
            build_train_evaluate_model(X_xor, y_xor, num_capas, num_neuronas_por_capa, num_epocas)
        elif opcion == "4":  # Ejercicio 1
            X_ejercicio1 = np.array([[2, 0], [0, 0], [2, 2], [0, 1], [1, 1], [1, 2]], dtype="float32")
            y_ejercicio1 = np.array([1, 0, 1, 0, 1, 0], dtype="float32")
            build_train_evaluate_model(X_ejercicio1, y_ejercicio1, num_capas, num_neuronas_por_capa, num_epocas)
        elif opcion == "5":  # Mayoría simple
            X_mayoria_simple = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                          [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype="float32")
            y_mayoria_simple = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype="float32")
            build_train_evaluate_model(X_mayoria_simple, y_mayoria_simple, num_capas, num_neuronas_por_capa, num_epocas)
        elif opcion == "6":  # Paridad de 4 bits
            X_paridad = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                                  [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
                                  [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                                  [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]], dtype="float32")
            y_paridad = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype="float32")
            build_train_evaluate_model(X_paridad, y_paridad, num_capas, num_neuronas_por_capa, num_epocas)
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")

