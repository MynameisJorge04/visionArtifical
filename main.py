import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

global var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY

_Lambda = 255
_Gamma = 0.5


# Identidad
def FP_Iden(M1):  # by columns
    high = M1.shape[0]  # r
    width = M1.shape[1]  # c
    M2 = np.zeros([high, width, 3])
    for k in range(3):  # canales
        for i in range(high):
            for j in range(width):
                M2[i][j][k] = M1[i][j][k]

    # print M2
    return M2


# Negativo
def FP_Neg(M1):
    high = M1.shape[0]  # r
    width = M1.shape[1]  # c
    M2 = np.zeros([high, width, 3])
    print('r=', len(M2))
    print('c=', len(M2[0]))
    for k in range(3):  # canales
        for i in range(high):
            for j in range(width):
                M2[i, j, k] = _Lambda - M1[i, j, k]

    # print M2
    return M2


def FP_Gamma(M1, gg):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = _Lambda * pow(M1[i, j, k] / _Lambda, gg)

    return M2


def FP_Logaritmo(M1, gg):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    fact = _Lambda / (np.log(_Lambda + 1))
    for i in range(high):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = fact * np.log(M1[i, j, k] + 1)

    return M2


def FP_Sin(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    fact = _Lambda / (np.log(_Lambda + 1))
    for i in range(high):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = _Lambda * np.sin(M1[i, j, k] * np.pi / (2 * _Lambda))

    return M2


def FP_Cosnoidal(M1, _L):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    fact = _Lambda / (np.log(_Lambda + 1))
    for i in range(high):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = _L * (1 - np.cos(M1[i, j, k] * np.pi / (2 * _L)))

    return M2


def FP_BorderX(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high):
        for j in range(width - 1):
            for k in range(3):
                M2[i][j][k] = abs(M1[i][j + 1][k] - M1[i, j, k])

    return M2


def FP_BorderY(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high - 1):
        for j in range(width):
            for k in range(3):
                M2[i][j][k] = abs(M1[i + 1][j][k] - M1[i, j, k])

    return M2

def grey_filter2(M1):
    high = M1.shape[0]
    width = M1.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high):
        for j in range(width):
            for k in range(3):
                M2[i, j, k] = np.mean(M1[i, j])
    return M2







def read():
    img = cv2.imread('lighthouse.bmp')  # cv2.IMREAD_GRAYSACALE
    img1 = cv2.imread('BOLAS.BMP')  # cv2.IMREAD_GRAYSACALE

    ventana = tk.Tk()

    # Crear la variable para almacenar el valor de la casilla de verificación
    var_FP_Neg = tk.BooleanVar()
    var_FP_Gamma = tk.BooleanVar()
    var_FP_Logaritmo = tk.BooleanVar()
    var_FP_Sin = tk.BooleanVar()
    var_FP_Cosnoidal = tk.BooleanVar()
    var_FP_BorderX = tk.BooleanVar()
    var_FP_BorderY = tk.BooleanVar()

    # Crear el estilo para los checkbox
    estilo_checkbox = ttk.Style()
    estilo_checkbox.configure("TCheckbutton", background="#99cceb", font=("Arial", 12), borderwidth=2, relief="groove")

    # Crear los checkbox
    checkbutton_FP_Neg = ttk.Checkbutton(ventana, text="Aplica FP_Neg", variable=var_FP_Neg, style="TCheckbutton",
                                         command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_Gamma = ttk.Checkbutton(ventana, text="Aplica FP_Gamma", variable=var_FP_Gamma, style="TCheckbutton",
                                           command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_Logaritmo = ttk.Checkbutton(ventana, text="Aplica FP_Logaritmo", variable=var_FP_Logaritmo,
                                               style="TCheckbutton", command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_Sin = ttk.Checkbutton(ventana, text="Aplica FP_Sin", variable=var_FP_Sin, style="TCheckbutton",
                                         command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_Cosnoidal = ttk.Checkbutton(ventana, text="Aplica FP_Cosnoidal", variable=var_FP_Cosnoidal,
                                               style="TCheckbutton", command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_BorderX = ttk.Checkbutton(ventana, text="Aplica FP_BorderX", variable=var_FP_BorderX,
                                             style="TCheckbutton", command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))
    checkbutton_FP_BorderY = ttk.Checkbutton(ventana, text="Aplica FP_BorderY", variable=var_FP_BorderY,
                                             style="TCheckbutton", command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo, var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX, var_FP_BorderY))



    # Ubicar los checkbox en la ventana
    checkbutton_FP_Neg.place(relx=0.5, rely=0.1, anchor="center")
    checkbutton_FP_Gamma.place(relx=0.5, rely=0.2, anchor="center")
    checkbutton_FP_Logaritmo.place(relx=0.5, rely=0.3, anchor="center")
    checkbutton_FP_Sin.place(relx=0.5, rely=0.4, anchor="center")
    checkbutton_FP_Cosnoidal.place(relx=0.5, rely=0.5, anchor="center")
    checkbutton_FP_BorderX.place(relx=0.5, rely=0.6, anchor="center")
    checkbutton_FP_BorderY.place(relx=0.5, rely=0.7, anchor="center")

    # Cargar la imagen con PIL
    imagen = Image.open('lighthouse.bmp')

    # Crear un objeto PhotoImage a partir de la imagen
    imagen_tk = ImageTk.PhotoImage(imagen)

    # Crear un widget Label para mostrar la imagen
    label_imagen = tk.Label(ventana, image=imagen_tk)

    # Ubicar el widget Label en la ventana
    label_imagen.pack(side="left", padx=10, pady=10)

    # Crear un widget Label para mostrar el texto debajo de la imagen
    label_texto = tk.Label(ventana, text="Imagen original")
    label_texto.pack(side="left", pady=10)

    # Crear la figura de Matplotlib y agregar una subparcela
    fig, ax = plt.subplots(figsize=(4, 4))

    # Crear el objeto FigureCanvasTkAgg y agregarlo a la ventana
    canvas = FigureCanvasTkAgg(fig, master=ventana)
    canvas.get_tk_widget().pack(side="right", padx=100, pady=10)

    # Crear el botón de aplicar filtros y agregarlo a la ventana
    boton_aplicar_filtros = tk.Button(ventana, text="Aplicar filtros")

    imagen_original = img

    # Definir la función que se ejecutará cuando se presione el botón
    def aplicar_filtros():

        # Verificar si la checkbox de FP_Neg está activada
        if var_FP_Neg.get():
            # Aplicar el filtro
            imagen_filtrada = FP_Neg(imagen_original)
        elif var_FP_Gamma.get():
            # Aplicar el filtro
            imagen_filtrada = FP_Gamma(imagen_original, 0.5)
        elif var_FP_Logaritmo.get():
            # Aplicar el filtro
            imagen_filtrada = FP_Logaritmo(imagen_original, 0.5)
        elif var_FP_Sin.get():
            # Aplicar el filtro
            imagen_filtrada = FP_Sin(imagen_original)
        elif var_FP_Cosnoidal.get():
            # Aplicar el filtro
            imagen_filtrada = FP_Cosnoidal(imagen_original, 18)
        elif var_FP_BorderX.get():
            # Aplicar el filtro
            imagen_filtrada = var_FP_BorderX(imagen_original)
        elif var_FP_BorderY.get():
            # Aplicar el filtro
            imagen_filtrada = var_FP_BorderY(imagen_original)
        else:
            imagen_filtrada = imagen_original

        # Escalar los valores de la imagen filtrada al rango [0, 1]
        imagen_filtrada = np.clip(imagen_filtrada, 0, 255) / 255

        # Mostrar la imagen filtrada en la subparcela
        ax.imshow(imagen_filtrada)

        # Actualizar el canvas
        canvas.draw()

    # Mostrar la imagen filtrada en la subparcela
    ax.imshow(imagen_original)

    # Actualizar el canvas
    canvas.draw()

    # Asignar la función de aplicar filtros al botón
    boton_aplicar_filtros.config(command=aplicar_filtros)
    boton_aplicar_filtros.pack(side="bottom", padx=10)

    label_texto = tk.Label(ventana, text="Imagen con filtro")
    label_texto.pack(side="right")

    # Ejecutar la ventana
    ventana.configure(bg="#99cceb")
    ventana.geometry("300x300")
    ventana.title("Mi ventana de checkbox")
    ventana.mainloop()



def select_only_one(*vars_to_check):
    # Deseleccionar los otros checkboxes si el actual está seleccionado
    for var in vars_to_check:
        if var.get() and sum(v.get() for v in vars_to_check) > 1:
            for v in vars_to_check:
                if v != var:
                    v.set(False)
            break



# main()
read()
