#Compuerta lógica AND negada.

import numpy as np
import tensorflow as tf
import math as math

# X = np.zeros((4, 2))

Y = np.zeros((4, 1))

X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

Y = np.array([1, 1, 1, 0])

W = np.random.rand(3)
W = [0, 0, 0]
synapsis = np.zeros((4, 1))
# axion = []
er = 0
print('X: ')
print(X)

# print(Y)
print('W: ')
print(W)


#Multiplicax*w; es la entrada asumiendo un peso; puede aplicarse normalización
# de ser necesario
def dendrita(x):
    n = np.multiply(x, W)
    return n


#Suma los resultados features x; este resultado nos dice el coeficiente intelectual de la neurona
#, gracias al factor de error; nos permite calibrar su inteligencia.
# Tratando de minimizar la sumatoria real contra la de la sumatoria
def neuron(n):
    return np.sum(n)

#Es la funcion de activación de la neurona; dependiendo de la actividad puede ser modificada
#incluso puede aplicarse una regularización en esta sálida.
def axon(s):
    return 1 / (1 + math.exp(-s))
    
def axon_derivative(s):
    y = axon(s)
    return y * (1 - y)

#Pensar es decir usar la neurona
def think(x):
    d = dendrita(x)
#    print('dendrita: ', d)
    n = neuron(d)
#    print('neuron: ', n)
    a = axon(n)
    return a



# Ejecuta todas las muestras (lecciones) en una interacción; devuelve el error
# cuadrático medio; dependiendo de la cantidad de lecciones que tome la neurona; ajusta su calibraje.
# mejorando su coeficiente intelectual
def lesson(x,y):
    #print(x)
    return think(x) - y
        

# Este es el profesor de la neurona; regula las lecciones y le da una calificación por su trabajo.
# A medida que se entrena, la calificación es mayor.
# de acuerdo a su learning rate; es el tiempo que el trainer le da a la neurona para asimilar
# conocimientos.
def trainer(lessons, learning_rate):
    for l in range(0, lessons):                         #Cantidad de iteracciones
        for i, training_set in enumerate(X):            #Recorrer cada Training set
            grade = lesson(training_set,Y[i])           #Probar la hipotesis; y obtener la diferencia entre la hipotesis y lo ideal
            print(grade, l)             
            for j, x in enumerate(training_set):                 #Recorrer cada Feature del trainingset
                W[j] = W[j] - ( (learning_rate * grade) * x )    #Cada Feature (x) tiene su Peso (w); por lo que dependiendo del tamaño
                                                                 #del feature(x) ponderamos el error. Asi entre mayor sea x, afectaos 
                                                                 #mas su peso(w).            
                                                                 #por eso learning_rate; representa cuanto arriesgamos para acercarnos 
                                                                 #a la convergencia-



trainer(5, 10)

print('W: ', W)
print('think 1: ', think(X[0]))
print('think 2: ', think(X[1]))
print('think 3: ', think(X[2]))
print('think 4: ', think(X[3]))