#Ejercicio realizado "from scratch", usando conocimientos adquiridos en los cursos de AndreNG.
#El código esta completamente documentado en español por mi.

# By Carlos Manuel Patiño

import numpy as np 
import math as math


Y = np.zeros((4, 1))

X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

Y = np.array([0, 0.1, 0.5, 0])

#Filas = Neuronas, Columnas = Entradas (Features?)
#Dos neuronas de tres entradas; Capa 1
global W1
global W2
W1 = np.random.rand(2, 3)
W2 = np.random.rand(1, 3)[0]
s = 2, 3
#W1 = np.ones(s)
#W2 = np.ones(3)
#W1 = np.array([[-30, 20, 20], [10, -20, -20]])
#W2 = np.array([-10, 20, 20])
er = 0
print('X: ')
print(X)



#Multiplicax*w; es la entrada asumiendo un peso; puede aplicarse normalización
# de ser necesario
def dendrita(x, w):
    n = np.multiply(x, w)
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
    return s * (1 - s)
    
#Pensar es decir usar la neurona
def think(x, w):
    d = dendrita(x, w)
    n = neuron(d)
    a = axon(n)
    return a



# Ejecuta todas las muestras (lecciones) en una interacción; devuelve el error
# cuadrático medio; dependiendo de la cantidad de lecciones que tome la neurona; ajusta su calibraje.
# mejorando su coeficiente intelectual
def lesson(x, y, w):
    return y - think(x, w)

def test(x):
    a2_1 = think(x, W1[0])
    a2_2 = think(x, W1[1])
    a3 = think([1,a2_1,a2_2], W2)
    return a3
# Este es el profesor de la neurona; regula las lecciones y le da una calificación por su trabajo.
# A medida que se entrena, la calificación es mayor.
# de acuerdo a su learning rate; es el tiempo que el trainer le da a la neurona para asimilar
# conocimientos.
def trainer(lessons, learning_rate):
    global W1
    global W2
    a3 = 0

    for m in range(0, lessons):
        s = 2, 3
        t1_delta = np.zeros(s)
        t2_delta = np.zeros(3)
        d2 = np.zeros(3)
        d3 = 0
        for i, a1 in enumerate(X):
            #Intento vectorizar la formula del Costo...

            a2_1 = think(a1, W1[0])           #capa 2; neurona 1
            a2_2 = think(a1, W1[1])           #capa 3; neurona 2
            a2 = np.array([1, a2_1, a2_2])    #acá agregamos el bias en el puest 0; (la neurona tiene tres entradas)
                                              #ademas en la segunda capa entra ya en el vector X
            
            a3 = think(a2, W2)
   
            d3 = a3 - Y[i]                     #delta de la capa tres
            
            #Delta debe tener un elemento por nodo de la capa respectiva
            #es decir tenemos dos neuronas; seria d2 con dos elementos. (sobra el primero d[0])
            #lo ideal es truncarlo con t1_delta???
            #El error d3; multiplicado por el peso de la capa anterior que lo produce
            #multiplicado por la derivada que lo produjo; de la entrada que lo produjo a2
            #Asi el tamaño del error es tan grande como los pesos que lo produjo y sus entradas
            #Al restarselo al peso; el error se va reduciendo en cada iteraccion
            #La formuna de Andrew para d; funciona si esta vectorizada
            
            #Simplemente, el peso que produjo el error final multiplicado por el error producido
            #Luego por la entrada que produjo el error INTERESANTE!!!!!!!!!!!!!!!!
            d2[0] = ( W2[0] * d3 ) * axon_derivative(a2[0])     #delta bias de la capa dos; ignorado más adelante
            d2[1] = ( W2[1] * d3 ) * axon_derivative(a2[1])     #delta 1 de la capa dos  
            d2[2] = ( W2[2] * d3 ) * axon_derivative(a2[2])     #delta 2 de la capa dos
            
            #Los tx_delta, acumulan cada error de cada una de las muestras (m)
            #
            t2_delta[0] = t2_delta[0] + (d3 * a2[0])
            t2_delta[1] = t2_delta[1] + (d3 * a2[1])
            t2_delta[2] = t2_delta[2] + (d3 * a2[2])
            
            
            #Esto debe tener la misma arquitectora de W1 (theta1)
            #Vemos como delta2 usamos uno por cada fila de t1
            t1_delta[0][0] = t1_delta[0][0] + (d2[1] * a1[0])
            t1_delta[0][1] = t1_delta[0][1] + (d2[1] * a1[1])
            t1_delta[0][2] = t1_delta[0][2] + (d2[1] * a1[2])
            
            t1_delta[1][0] = t1_delta[1][0] + (d2[2] * a1[0])
            t1_delta[1][1] = t1_delta[1][1] + (d2[2] * a1[1])
            t1_delta[1][2] = t1_delta[1][2] + (d2[2] * a1[2])
            
            
        #Aca al terminarse los trainingsets; tenemos la sumatoria de los errores
        #cuya proporcionalidad se la restamos a los pesos.
        W1[0][0] = W1[0][0] - (learning_rate * (t1_delta[0][0] / 4))
        W1[0][1] = W1[0][1] - (learning_rate * (t1_delta[0][1] / 4))
        W1[0][2] = W1[0][2] - (learning_rate * (t1_delta[0][2] / 4))
        
        W1[1][0] = W1[1][0] - (learning_rate * (t1_delta[1][0] / 4))
        W1[1][1] = W1[1][1] - (learning_rate * (t1_delta[1][1] / 4))
        W1[1][2] = W1[1][2] - (learning_rate * (t1_delta[1][2] / 4))    
         
        W2[0] = W2[0] - (learning_rate * (t2_delta[0] / 4))
        W2[1] = W2[1] - (learning_rate * (t2_delta[1] / 4))
        W2[2] = W2[2] - (learning_rate * (t2_delta[2] / 4))


trainer(500000, 0.1) 
 
print('W1: ', W1)
print('W2: ', W2)
t1 = test([1, 0, 0])
print('t1: ', t1)
t2 = test([1, 0, 1])
print('t2: ', t2)
t3 = test([1, 1, 0])
print('t3: ', t3)
t4 = test([1, 1, 1])
print('t4: ', t4)
