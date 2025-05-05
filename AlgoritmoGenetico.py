import pygad
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


lista_alunos = pd.read_csv('alunos_habilidades.csv')

# Habilidades necessárias para o projeto e suas prioridades
habilidades_Projeto = {
    3: ['INF2', 'DSL'],  # Alta prioridade
    2: ['AS', 'DT'],  # Média prioridade
    1: ['GP']  # Baixa prioridade
}

# Definir os pesos das prioridades
pesos_prioridade = {
    3: 3,  # Alta prioridade tem peso 3
    2: 2,  # Média prioridade tem peso 2
    1: 1   # Baixa prioridade tem peso 1
}

# Variáveis para monitoramento
tempo_por_geracao = []
combinacoes_realizadas_por_geracao = []
start_time = time.time()

# Função de fitness
def fitness_func(ga_instance,solution, solution_idx):
 
    # Seleciona os alunos da equipe com base na solução
    equipe = lista_alunos.iloc[solution]

    # Conjunto para armazenar habilidades cobertas pela equipe
    habilidades_cobertas = set()

    # Avalia as habilidades cobertas pela equipe
    for _, aluno in equipe.iterrows():
        for prioridade in [3, 2, 1]:  # Avalia em ordem de prioridade
            for habilidade in habilidades_Projeto[prioridade]:
                if habilidade in aluno and aluno[habilidade] > 0:
                    habilidades_cobertas.add(habilidade)

    # Agora calcula o fitness com base nas habilidades cobertas e suas prioridades
    fitness = 0
    for prioridade, habilidades in habilidades_Projeto.items():
        for habilidade in habilidades:
            if habilidade in habilidades_cobertas:
                fitness += pesos_prioridade[prioridade]

    return fitness

num_alunos = lista_alunos.shape[0]
print(num_alunos)
num_genes = 5  # Número de membros na equipe

ga_instance = pygad.GA(
    num_generations=5000,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=5,
    num_genes=num_genes,
    gene_type=int,
    gene_space={'low': 0, 'high': num_alunos - 1},  # Índices dos alunos
    parent_selection_type="rws",
    mutation_percent_genes=30,
    mutation_type="random",
    on_generation=lambda ga: track_progress(ga)
)

# Função para rastrear o tempo por geração
def track_progress(ga_instance):
    tempo_por_geracao.append(time.time() - start_time)
    combinacoes_realizadas_por_geracao.append(ga_instance.generations_completed)

inicio = time.time()
# Executa o GA
ga_instance.run()

final = time.time()

tempo_total = final - inicio 
minutos = int(tempo_total // 60)
segundos = tempo_total % 60

solucao, fitness, _ = ga_instance.best_solution()

# Obter e printar os nomes dos alunos na equipe selecionada
equipe_selecionada = lista_alunos.iloc[solucao]  # Seleciona os alunos pela solução
nomes_equipe = equipe_selecionada['Aluno']  # Assumindo que a coluna com os nomes é 'Nome'

print(f'Tempo Gasto ao formar equipes: {minutos}:{segundos:.1f}')
print("Equipe selecionada:")
print(nomes_equipe.tolist())  # Exibe os nomes dos alunos na equipe
print("Pontuação da solução:", fitness)
