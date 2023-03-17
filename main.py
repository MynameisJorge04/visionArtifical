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

LAMBDA = 255
GAMMA = 0.5
L = 18


# Identity filter - 1
def identity_filter(matrix):
    return matrix.copy()


# Filtro negativo
def negative_filter(matrix):
    return LAMBDA - matrix


# Filtro Gris (R+G+B)/3

def grey_filter(matrix):
    grey = np.mean(matrix, axis=-1)

    grey = np.expand_dims(grey, axis=-1)
    grey = np.repeat(grey, 3, axis=-1)

    return grey


# Filtro gamma
def gamma_filter(matrix, gamma):
    """
    Aplica el filtro gamma a la matriz de entrada.
    """
    return LAMBDA * np.power(matrix / LAMBDA, gamma)


# Filtro rango dinámico
def logarithmic_filter(matrix):
    """
    Aplica el filtro logarítmico a la matriz de entrada.
    """
    factor = LAMBDA / np.log(LAMBDA + 1)
    return factor * np.log(matrix + 1e-6)  # Agrega un valor constante pequeño para prevenir la división por cero


# Filtro rango dinámico parametrizado
def dynamic_range_filter(matrix, alpha):
    """
    Aplica un filtro de rango dinámico a la matriz de entrada.
    El filtro comprime el rango dinámico de la matriz, realzando detalles y contraste.
    """
    max_val = np.max(matrix)
    factor = (max_val / (1 - alpha)) ** alpha
    return factor * (matrix ** alpha)


# Filtro seno
def sine_filter(matrix):
    """
    Aplica el filtro seno a la matriz de entrada.
    """
    return LAMBDA * np.sin(matrix * np.pi / (2 * LAMBDA))


# Filtro coseno
def cosine_filter(matrix, lamda):
    """
    Aplica el filtro coseno a la matriz de entrada.
    """
    return lamda * (1 - np.cos(matrix * np.pi / (2 * lamda)))


# Filtro exponencial
def exponential_filter(matrix, gamma):
    """
    Aplica el filtro exponencial a la matriz de entrada.
    """
    return LAMBDA * (1 - np.exp(-matrix / (LAMBDA * gamma)))


# Filtro sigmoidal seno
def sigmoid_sine_filter(matrix, a=2, b=2):
    """
    Aplica el filtro sigmoidal seno a la matriz de entrada.
    """
    return LAMBDA / (1 + np.exp(-a * (matrix - LAMBDA / 2))) * np.sin(b * matrix)


# Filtro sigmoidal tangente hiperbólica
def hyperbolic_tangent_sigmoid_filter(matrix, a=1, b=1):
    """
    Aplica el filtro de tangente hiperbólica sigmoidal a la matriz de entrada.
    """
    return np.tanh(a * matrix) * (1 / (1 + np.exp(-b * matrix)))


# Función para obtener los bordes en el eje Y de una matriz

def FP_BorderX(matrix):
    high = matrix.shape[0]
    width = matrix.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high):
        for j in range(width - 1):
            for k in range(3):
                M2[i][j][k] = abs(matrix[i][j + 1][k] - matrix[i, j, k])

    return M2


# Función para obtener los bordes en el eje Y de una matriz

def FP_BorderY(matrix):
    high = matrix.shape[0]
    width = matrix.shape[1]
    M2 = np.zeros([high, width, 3])
    for i in range(high - 1):
        for j in range(width):
            for k in range(3):
                M2[i][j][k] = abs(matrix[i + 1][j][k] - matrix[i, j, k])

    return M2


# Función para obtener los bordes en el eje Y de una matriz

def ecualizate_histogram(matrix):
    """
    Función para obtener los bordes en el eje Y de una matriz
    """
    # Calcular el histograma de la matriz de entrada
    hist, bins = np.histogram(matrix.flatten(), 256, [0, 256])

    # Calcular el histograma de la matriz de entrada
    cdf = hist.cumsum()

    cdf_normalized = cdf * hist.max() / cdf.max()

    equalized_matrix = np.interp(matrix.flatten(), bins[:-1], cdf_normalized).reshape(matrix.shape)

    return equalized_matrix.astype(np.uint8)


def read():
    img = cv2.imread('lighthouse.bmp')  # cv2.IMREAD_GRAYSACALE
    img1 = cv2.imread('BOLAS.BMP')  # cv2.IMREAD_GRAYSACALE

    ventana = tk.Tk()

    # Crear la variable para almacenar el valor de la casilla de verificación
    var_FP_Identy = tk.BooleanVar()
    var_FP_Neg = tk.BooleanVar()
    var_FP_Gamma = tk.BooleanVar()
    var_FP_Logaritmo = tk.BooleanVar()
    var_FP_Sin = tk.BooleanVar()
    var_FP_Cosnoidal = tk.BooleanVar()
    var_FP_BorderX = tk.BooleanVar()
    var_FP_BorderY = tk.BooleanVar()
    var_FP_ecualizate_histogram = tk.BooleanVar()
    var_FP_grey_filter = tk.BooleanVar()
    var_FP_exponential_filter = tk.BooleanVar()
    var_FP_sigmoid_sine_filter = tk.BooleanVar()
    var_FP_hyperbolic_tangent_sigmoid_filter = tk.BooleanVar()

    # Crear el estilo para los checkbox
    estilo_checkbox = ttk.Style()
    estilo_checkbox.configure("TCheckbutton", background="#99cceb", font=("Arial", 12), borderwidth=2, relief="groove")

    # Crear los checkbox
    checkbutton_FP_Identy = ttk.Checkbutton(ventana, text="Aplica FP_Identy", variable=var_FP_Identy, style="TCheckbutton",
                                            command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                            var_FP_Sin, var_FP_Cosnoidal,
                                                                            var_FP_BorderX, var_FP_BorderY,
                                                                            var_FP_Identy, var_FP_ecualizate_histogram,
                                                                            var_FP_grey_filter,
                                                                            var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                            var_FP_sigmoid_sine_filter))
    checkbutton_FP_Neg = ttk.Checkbutton(ventana, text="Aplica FP_Neg", variable=var_FP_Neg, style="TCheckbutton",
                                         command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                         var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX,
                                                                         var_FP_BorderY, var_FP_Identy,
                                                                         var_FP_ecualizate_histogram,
                                                                         var_FP_grey_filter,
                                                                         var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                         var_FP_sigmoid_sine_filter))
    checkbutton_FP_grey_filter = ttk.Checkbutton(ventana, text="Aplica grey_filter", variable=var_FP_grey_filter,
                                                 style="TCheckbutton",
                                                 command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                                 var_FP_Logaritmo, var_FP_Sin,
                                                                                 var_FP_Cosnoidal, var_FP_BorderX,
                                                                                 var_FP_BorderY, var_FP_Identy,
                                                                                 var_FP_ecualizate_histogram,
                                                                                 var_FP_grey_filter,
                                                                                 var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                                 var_FP_sigmoid_sine_filter))
    checkbutton_FP_Gamma = ttk.Checkbutton(ventana, text="Aplica FP_Gamma", variable=var_FP_Gamma, style="TCheckbutton",
                                           command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                           var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX,
                                                                           var_FP_BorderY, var_FP_Identy,
                                                                           var_FP_ecualizate_histogram,
                                                                           var_FP_grey_filter,
                                                                           var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                           var_FP_sigmoid_sine_filter))
    checkbutton_FP_Logaritmo = ttk.Checkbutton(ventana, text="Aplica FP_Logaritmo", variable=var_FP_Logaritmo,
                                               style="TCheckbutton",
                                               command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                               var_FP_Logaritmo, var_FP_Sin,
                                                                               var_FP_Cosnoidal, var_FP_BorderX,
                                                                               var_FP_BorderY, var_FP_Identy,
                                                                               var_FP_ecualizate_histogram,
                                                                               var_FP_grey_filter,
                                                                               var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                               var_FP_sigmoid_sine_filter))
    checkbutton_FP_Sin = ttk.Checkbutton(ventana, text="Aplica FP_Sin", variable=var_FP_Sin, style="TCheckbutton",
                                         command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                         var_FP_Sin, var_FP_Cosnoidal, var_FP_BorderX,
                                                                         var_FP_BorderY, var_FP_Identy,
                                                                         var_FP_ecualizate_histogram,
                                                                         var_FP_grey_filter,
                                                                         var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                         var_FP_sigmoid_sine_filter))
    checkbutton_FP_Cosnoidal = ttk.Checkbutton(ventana, text="Aplica FP_Cosnoidal", variable=var_FP_Cosnoidal,
                                               style="TCheckbutton",
                                               command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                               var_FP_Logaritmo, var_FP_Sin,
                                                                               var_FP_Cosnoidal, var_FP_BorderX,
                                                                               var_FP_BorderY, var_FP_Identy,
                                                                               var_FP_ecualizate_histogram,
                                                                               var_FP_grey_filter,
                                                                               var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                               var_FP_sigmoid_sine_filter))
    checkbutton_FP_BorderX = ttk.Checkbutton(ventana, text="Aplica FP_BorderX", variable=var_FP_BorderX,
                                             style="TCheckbutton",
                                             command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                             var_FP_Sin, var_FP_Cosnoidal,
                                                                             var_FP_BorderX, var_FP_BorderY,
                                                                             var_FP_Identy, var_FP_ecualizate_histogram,
                                                                             var_FP_grey_filter,
                                                                             var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                             var_FP_sigmoid_sine_filter))
    checkbutton_FP_BorderY = ttk.Checkbutton(ventana, text="Aplica FP_BorderY", variable=var_FP_BorderY,
                                             style="TCheckbutton",
                                             command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma, var_FP_Logaritmo,
                                                                             var_FP_Sin, var_FP_Cosnoidal,
                                                                             var_FP_BorderX, var_FP_BorderY,
                                                                             var_FP_Identy, var_FP_ecualizate_histogram,
                                                                             var_FP_grey_filter,
                                                                             var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                             var_FP_sigmoid_sine_filter))
    checkbutton_FP_ecualizate_histogram = ttk.Checkbutton(ventana, text="Aplica ecualizate_histogram",
                                                          variable=var_FP_ecualizate_histogram,
                                                          style="TCheckbutton",
                                                          command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                                          var_FP_Logaritmo, var_FP_Sin,
                                                                                          var_FP_Cosnoidal,
                                                                                          var_FP_BorderX,
                                                                                          var_FP_BorderY, var_FP_Identy,
                                                                                          var_FP_ecualizate_histogram,
                                                                                          var_FP_grey_filter,
                                                                                          var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                                          var_FP_sigmoid_sine_filter))
    checkbutton_FP_exponential_filter = ttk.Checkbutton(ventana, text="Aplica exponential_filter",
                                                        variable=var_FP_exponential_filter,
                                                        style="TCheckbutton",
                                                        command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                                        var_FP_Logaritmo, var_FP_Sin,
                                                                                        var_FP_Cosnoidal,
                                                                                        var_FP_BorderX, var_FP_BorderY,
                                                                                        var_FP_Identy,
                                                                                        var_FP_ecualizate_histogram,
                                                                                        var_FP_grey_filter,
                                                                                        var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                                        var_FP_sigmoid_sine_filter))
    checkbutton_FP_sigmoid_sine_filter = ttk.Checkbutton(ventana, text="Aplica sigmoid_sine_filter",
                                                         variable=var_FP_sigmoid_sine_filter,
                                                         style="TCheckbutton",
                                                         command=lambda: select_only_one(var_FP_Neg, var_FP_Gamma,
                                                                                         var_FP_Logaritmo, var_FP_Sin,
                                                                                         var_FP_Cosnoidal,
                                                                                         var_FP_BorderX, var_FP_BorderY,
                                                                                         var_FP_Identy,
                                                                                         var_FP_ecualizate_histogram,
                                                                                         var_FP_grey_filter,
                                                                                         var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                                         var_FP_sigmoid_sine_filter))
    checkbutton_FP_hyperbolic_tangent_sigmoid_filter = ttk.Checkbutton(ventana, text="Aplica tangent_sigmoid_filter",
                                                                       variable=var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                       style="TCheckbutton",
                                                                       command=lambda: select_only_one(var_FP_Neg,
                                                                                                       var_FP_Gamma,
                                                                                                       var_FP_Logaritmo,
                                                                                                       var_FP_Sin,
                                                                                                       var_FP_Cosnoidal,
                                                                                                       var_FP_BorderX,
                                                                                                       var_FP_BorderY,
                                                                                                       var_FP_Identy,
                                                                                                       var_FP_ecualizate_histogram,
                                                                                                       var_FP_grey_filter,
                                                                                                       var_FP_hyperbolic_tangent_sigmoid_filter,
                                                                                                       var_FP_sigmoid_sine_filter))

    # Ubicar los checkbox en la ventana
    checkbutton_FP_exponential_filter.place(relx=0.5, rely=0.1, anchor="center")
    checkbutton_FP_sigmoid_sine_filter.place(relx=0.5, rely=0.2, anchor="center")
    checkbutton_FP_hyperbolic_tangent_sigmoid_filter.place(relx=0.5, rely=0.3, anchor="center")
    checkbutton_FP_ecualizate_histogram.place(relx=0.5, rely=0.4, anchor="center")
    checkbutton_FP_Identy.place(relx=0.5, rely=0.5, anchor="center")
    checkbutton_FP_Neg.place(relx=0.5, rely=0.6, anchor="center")
    checkbutton_FP_grey_filter.place(relx=0.5, rely=0.7, anchor="center")
    checkbutton_FP_Gamma.place(relx=0.5, rely=0.8, anchor="center")
    checkbutton_FP_Logaritmo.place(relx=0.5, rely=0.9, anchor="center")
    checkbutton_FP_Sin.place(relx=0.7, rely=0.9, anchor="center")
    checkbutton_FP_Cosnoidal.place(relx=0.3, rely=0.9, anchor="center")
    checkbutton_FP_BorderX.place(relx=0.7, rely=0.8, anchor="center")
    checkbutton_FP_BorderY.place(relx=0.3, rely=0.8, anchor="center")

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
    imagen_original2 = img1

    # Definir la función que se ejecutará cuando se presione el botón
    def aplicar_filtros():

        # Verificar si la checkbox de FP_Neg está activada
        if var_FP_Identy.get():
            # Aplicar el filtro
            imagen_filtrada = identity_filter(imagen_original)
        elif var_FP_Neg.get():
            # Aplicar el filtro
            imagen_filtrada = negative_filter(imagen_original)
        elif var_FP_grey_filter.get():
            # Aplicar el filtro
            imagen_filtrada = grey_filter(imagen_original)
        elif var_FP_Gamma.get():
            # Aplicar el filtro
            imagen_filtrada = gamma_filter(imagen_original, 0.5)
        elif var_FP_Logaritmo.get():
            # Aplicar el filtro
            imagen_filtrada = logarithmic_filter(imagen_original)
        elif var_FP_Sin.get():
            # Aplicar el filtro
            imagen_filtrada = sine_filter(imagen_original)
        elif var_FP_Cosnoidal.get():
            # Aplicar el filtro
            imagen_filtrada = cosine_filter(imagen_original, 18)
        elif var_FP_BorderX.get():
            # Aplicar el filtro
            imagen_filtrada = FP_BorderX(imagen_original2)
        elif var_FP_BorderY.get():
            # Aplicar el filtro
            imagen_filtrada = FP_BorderY(imagen_original2)
        elif var_FP_ecualizate_histogram.get():
            # Aplicar el filtro
            imagen_filtrada = ecualizate_histogram(imagen_original)
        elif var_FP_hyperbolic_tangent_sigmoid_filter.get():
            # Aplicar el filtro
            imagen_filtrada = hyperbolic_tangent_sigmoid_filter(imagen_original)
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