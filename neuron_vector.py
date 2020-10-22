#Compuerta lógica AND negada.
#Version de neurona vectorizada con numpy
import numpy as np 
import math as math


Y = np.zeros((4, 1))

X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

Y = np.array([0, 0.1, 0.5, 1])

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
    print (x, w)
    n = np.dot(x, w.T)
    return n


#Suma los resultados features x; este resultado nos dice el coeficiente intelectual de la neurona
#, gracias al factor de error; nos permite calibrar su inteligencia.
# Tratando de minimizar la sumatoria real contra la de la sumatoria
def neuron(n):
    #print('n; ', n)
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

            a2_1 = think(a1, W1)           #capa 2; neurona 1
            print('a2_1: ', a2_1)
            a2_2 = think(a1, W1[1])           #capa 3; neurona 2
            a2 = np.array([1, a2_1, a2_2])    #acá agregamos el bias en el puest 0; (la neurona tiene tres entradas)
                                              #ademas en la segunda capa entra ya en el vector X
            
            a3 = think(a2, W2)
   
            d3 = a3 - Y[i]                     #delta de la capa tres, es uno por capa   
            d2 = ( W2 * d3 ) * axon_derivative(a2)     #Error * pesos que lo produjeron * la deriviada de la entrada. es unpo por capa

            
            #Los tx_delta, acumulan cada error de cada una de las muestras (m), es uno por capa
            t2_delta = t2_delta + (d3 * a2)
            t1_delta = t1_delta + (d2 * a1)

            
        #Aca al terminarse los trainingsets; tenemos la sumatoria de los errores
        #cuya proporcionalidad se la restamos a los pesos.
        W1 = W1 - (learning_rate * (t1_delta / 4))
        W2 = W2 - (learning_rate * (t2_delta / 4))



trainer(1, 0.1) 
 
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