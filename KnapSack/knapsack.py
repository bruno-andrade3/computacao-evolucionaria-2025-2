import time
import numpy as np

#crossover
#mutação
#selecaodospais
#aptidao

pesos_dos_objetos = np.array([350, 250, 160, 120, 200, 100, 120, 220, 40, 80, 100, 300, 180, 250, 220, 150, 280, 310, 120, 160, 110, 210])
valor_dos_objetos = np.array([300, 400, 450, 350, 250, 300, 200, 250, 150, 400, 350, 300, 450, 500, 350, 400, 200, 300, 250, 300, 150, 200])

PESO_MAX = 3000

tamanho_da_populacao = 10
tamanho_do_genoma = np.size(pesos_dos_objetos)

taxa_de_mutacao = 0.05

populacao = np.random.randint(2, size=(tamanho_da_populacao, tamanho_do_genoma))
print(populacao)

def fitness(populacao):
    pesos = np.zeros()
    valores = np.zeros()
    for i in len(populacao):
        peso = np.dot(populacao[i], pesos_dos_objetos)
        valor = np.dot(populacao[i], valor_dos_objetos)
        peso_total = np.sum(peso)
        if peso_total > PESO_MAX:
            #fitness = 0
        pesos.append(peso)
        valores.append(valor)
    

    


inicio= time.time()

ponto_de_crossover = np.random.randint(tamanho_do_genoma - 1)

# Mutação
for i in range(tamanho_da_populacao):
    for j in range(tamanho_do_genoma):
        if np.random.rand() < taxa_de_mutacao:
            # Inverte o bit
            offspring[i, j] = 1 - offspring[i, j]




fim = time.time()
demorou = fim - inicio

#################################################################################
import time
import numpy as np

#