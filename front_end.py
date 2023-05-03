import tkinter as tk
import numpy as np
import pickle

from back_end import X_svm, y_svm

# Função para obter os valores dos campos de entrada e fazer previsões
def fazer_previsao():
    # Obter os valores de cada campo de entrada e transformá-los em um array NumPy
    entrada = np.array([
        float(radius_mean_entry.get()),
        float(texture_mean_entry.get()),
        float(perimeter_mean_entry.get()),
        float(area_mean_entry.get()),
        float(smoothness_mean_entry.get())
    ]).reshape(1, -1)

    print(entrada)

    with open('modelo_treinado.pkl', 'rb') as arquivo:
        modelo = pickle.load(arquivo)

        #modelo.fit(X_svm, y_svm)

    # Fazer a previsão com o modelo treinado
    previsao = modelo.predict(entrada)

    # Exibir a previsão na janela
    resultado_label.config(text=f"Diagnóstico: {previsao[0]}")

# Criação da janela principal
root = tk.Tk()
root.title("Predizer")

# Criação dos rótulos e campos de entrada
radius_mean_label = tk.Label(root, text="radius_mean")
radius_mean_label.grid(row=0, column=0)
radius_mean_entry = tk.Entry(root)
radius_mean_entry.grid(row=0, column=1)

texture_mean_label = tk.Label(root, text="texture_mean")
texture_mean_label.grid(row=1, column=0)
texture_mean_entry = tk.Entry(root)
texture_mean_entry.grid(row=1, column=1)

perimeter_mean_label = tk.Label(root, text="perimeter_mean")
perimeter_mean_label.grid(row=2, column=0)
perimeter_mean_entry = tk.Entry(root)
perimeter_mean_entry.grid(row=2, column=1)

area_mean_label = tk.Label(root, text="area_mean")
area_mean_label.grid(row=3, column=0)
area_mean_entry = tk.Entry(root)
area_mean_entry.grid(row=3, column=1)

smoothness_mean_label = tk.Label(root, text="smoothness_mean")
smoothness_mean_label.grid(row=4, column=0)
smoothness_mean_entry = tk.Entry(root)
smoothness_mean_entry.grid(row=4, column=1)

# Criação do botão de predição
predict_button = tk.Button(root, text="Predizer 'diagnosis'", command=fazer_previsao)
predict_button.grid(row=5, column=0, columnspan=2)

# Criação do rótulo para exibir o resultado
resultado_label = tk.Label(root, text="")
resultado_label.grid(row=6, column=0, columnspan=2)

# Iniciar o loop da interface
root.mainloop()
