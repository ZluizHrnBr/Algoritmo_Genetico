import pygad
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


alunos = pd.read_csv('matriz_habilidades.csv')

# Habilidades necessárias para o projeto e suas prioridades
habilidades_prioridade = {
    3: ['DSL', 'DM', 'DB'],  # Alta prioridade
    2: ['AS', 'TS', 'INF1'],  # Média prioridade
    1: ['DF']  # Baixa prioridade
}

# Definir os pesos das prioridades
pesos_prioridade = {
    3: 3,  # Alta prioridade tem peso 3
    2: 2,  # Média prioridade tem peso 2
    1: 1   # Baixa prioridade tem peso 1
}


start_time = time.time()

# Função de fitness
def fitness_func(ga_instance,solution, solution_idx):
    # Seleciona os alunos de acordo com a solução (índices dos alunos)
    equipe = alunos.iloc[solution]
    # Inicializar o fitness
    fitness = 0
    # Percorrer cada aluno na equipe
    for _, aluno in equipe.iterrows():
        # Verificar as habilidades do aluno em relação às prioridades
        for prioridade, habilidades in habilidades_prioridade.items():
            for habilidade_equipe in habilidades:
                # Verificar se o aluno possui a habilidade (valor > 0)
                if aluno[habilidade_equipe] > 0:
                    fitness += pesos_prioridade[prioridade]  # Soma o peso da prioridade

    return fitness

num_alunos = alunos.shape[0]
num_genes = 5  # Número de membros na equipe

ga_instance = pygad.GA(
    num_generations=10000,
    num_parents_mating=5,
    fitness_func=fitness_func,
    sol_per_pop=5,
    num_genes=num_genes,
    gene_type=int,
    gene_space={'low': 0, 'high': num_alunos - 1},  # Índices dos alunos
    parent_selection_type="rws",
    mutation_percent_genes=10,
    on_generation=lambda ga: track_progress(ga)
)

inicio = time.time()
# Executa o GA
ga_instance.run()

final = time.time()

tempo_total = final - inicio 
minutos = int(tempo_total // 60)
segundos = tempo_total % 60

solucao, fitness, _ = ga_instance.best_solution()

# Obter e printar os nomes dos alunos na equipe selecionada
equipe_selecionada = alunos.iloc[solucao]  # Seleciona os alunos pela solução
nomes_equipe = equipe_selecionada['Aluno']  # Assumindo que a coluna com os nomes é 'Nome'

print(f'Tempo Gasto ao formar equipes: {minutos}:{segundos:.1f}')
print("Equipe selecionada:")
print(nomes_equipe.tolist())  # Exibe os nomes dos alunos na equipe
print("Pontuação da solução:", fitness)
