import cv2
import numpy as np
import sys #daqui pra baixo referente a contar pixel
from multiprocessing import Queue
import matplotlib.pyplot as plt

image = cv2.imread('B9 (3).jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
blur = cv2.medianBlur(hsv, 11)
kernel = np.ones((5,5), np.uint8)

lower = np.array([0,100,100])
upper = np.array([20,255,255])

mask = cv2.inRange(blur, lower, upper)
#erosao = cv2.erode(mask, kernel, 1)
dilatacao = cv2.dilate(mask, kernel, 1)
fechamento = cv2.morphologyEx(dilatacao, cv2.MORPH_CLOSE, kernel)
res = cv2.bitwise_and(image,image, mask= mask)

altura = fechamento.shape[0]
largura = fechamento.shape[1]

#Algum problema nessa função, por exemplo para a ir_10047 ela aponta 101 objetos totais = Descobri o que foi
def contar(fechamento):
    contador = 0
    visitado = fechamento < 255

    for x in range(0, altura):
        for y in range(0, largura):
            if not visitado[x,y]:
                contador = contador + 1
                q = Queue()
                q.put([x,y])
                vizinhanca(fechamento, q, contador, visitado)
                cv2.putText(fechamento, str(contador), (x,y), cv2.FONT_ITALIC, 0.4, 255, 2)
    print("Total: ",contador, "objetos")

def vizinhanca(fechamento, q, contador, visitado):
    while (q.qsize() > 0):
        pixel = q.get()
        x = pixel[0]
        y = pixel[1]
        if (x > 0 and x < altura and y > 0 and y < largura):
            if (not visitado[x,y]):
                fechamento[x,y] = 200 - contador*2
                visitado[x,y] = True
                q.put([x+1, y])
                q.put([x-1, y])
                q.put([x, y+1])
                q.put([x, y-1])

def areas(fechamento):
    hist, bin = np.histogram(fechamento.ravel(), 256, [1,254])
    contador = 0

    maior = 0
    i = 0

    for i in range(254, 1, -1):
        if hist[i] > 0:
            contador = contador + 1
            print("Objeto {0:3d} - Nível de cinza: ".format(contador))
            print("{1:3d} - área: {0:7d} pixels".format(hist[i], i))
            #print(hist[i])
            #print(bin)
            #condição para apontar qual a maior área na Imagem
            if hist[i] > maior:
                maior = hist[i]
            else:
                maior = maior
            i = i + 1

    print("A maior área é igual a",maior, "pixels")
    print("Área total: ", np.sum(hist), "pixels")
    estimarPeso(maior)
    return contador, np.max(hist)

def estimarPeso(maior):
    peso = 0.0045*maior - 34.405
    print("O suíno tem aproximadamente", "%.2f" % peso, "Kg.")

contar(fechamento)
contador, max = areas(fechamento)

plt.hist(fechamento.ravel(), 256, [1,254])
plt.xlabel("Nível de Cinza")
plt.ylabel("Pixels")
plt.title("Histograma de Níveis de cinza")
plt.axis([200-(1+contador*2), 200, 0, max+100])
plt.show()

cv2.imshow("contados", fechamento)
cv2.waitKey()
cv2.destroyAllWindows()

#Salvar resultado
#cv2.imwrite('medianBlur.jpg', res)

cv2.imshow("mask ",fechamento)
cv2.imshow('stack', np.hstack([image, res]))
cv2.waitKey(0)
