import time
import numpy as np

#crossover
#mutação
#selecaodospais
#aptidao

pesos_dos_objetos = np.array([350, 250, 160, 120, 200, 100, 120, 220, 40, 80, 100, 300, 180, 250, 220, 150, 280, 310, 120, 160, 110, 210])
valor_dos_objetos = np.array([300, 400, 450, 350, 250, 300, 200, 250, 150, 400, 350, 300, 450, 500, 350, 400, 200, 300, 250, 300, 150, 200])

PESO_MAX = 3000
tamanho_do_genoma = len(pesos_dos_objetos)
tamanho_da_populacao = 10
taxa_de_mutacao = 0.05
geracoes = 500 #Mudar se demorar muito pra rodar
tamanho_torneio = 3

populacao = np.random.randint(2, size=(tamanho_da_populacao, tamanho_do_genoma))
print(populacao)

def fitness(individuo):
    peso_total = np.dot(individuo, pesos_dos_objetos)
    valor_total = np.dot(individuo, valor_dos_objetos)
    if peso_total > PESO_MAX:
        return 0  # Soluções inválidas
    return valor_total
    

def selecao_por_torneio(populacao, aptidoes):
    competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
    melhor = competidores[np.argmax(aptidoes[competidores])]
    return populacao[melhor]

def crossover(pai, mae):
    ponto = np.random.randint(1, tamanho_do_genoma-1)
    filho1 = np.concatenate((pai[:ponto], mae[ponto:]))
    filho2 = np.concatenate((mae[:ponto], pai[ponto:]))
    return filho1, filho2

def mutacao(individuo):
    for i in range(tamanho_do_genoma):
        if np.random.rand() < taxa_de_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo


inicio= time.time()

populacao = np.random.randint(2, size=(tamanho_da_populacao, tamanho_do_genoma))

melhor_valor = 0
melhor_solucao = None
geracao_melhor = 0

for g in range(geracoes):
    aptidoes = np.array([fitness(ind) for ind in populacao])
    
    # Guardar melhor solução
    max_fit = np.max(aptidoes)
    if max_fit > melhor_valor:
        melhor_valor = max_fit
        melhor_solucao = populacao[np.argmax(aptidoes)].copy()
        geracao_melhor = g
    
    nova_populacao = []
    while len(nova_populacao) < tamanho_da_populacao:
        pai1 = selecao_por_torneio(populacao, aptidoes)
        pai2 = selecao_por_torneio(populacao, aptidoes)
        filho1, filho2 = crossover(pai1, pai2)
        filho1 = mutacao(filho1)
        filho2 = mutacao(filho2)
        nova_populacao.append(filho1)
        nova_populacao.append(filho2)
    
    populacao = np.array(nova_populacao[:tamanho_da_populacao])

fim = time.time()

#RESULTADOS

itens_escolhidos = np.where(melhor_solucao == 1)[0]
peso_total = np.dot(melhor_solucao, pesos_dos_objetos)

print("=== Melhor Solução Encontrada ===")
print(f"Itens escolhidos: {itens_escolhidos}")
print(f"Peso total: {peso_total} g")
print(f"Valor total: {melhor_valor}")
print(f"Geração em que foi encontrada: {geracao_melhor}")
print(f"Tempo de execução: {fim - inicio:.2f} segundos")
