import matplotlib.pyplot as plt

def plot_metrics(fitness_history):
    plt.figure(figsize=(10,5))

    # График для среднего фитнеса
    plt.subplot(1, 2, 1)
    plt.plot(fitness_history, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    plt.show()