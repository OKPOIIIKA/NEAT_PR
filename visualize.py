import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(fitness_history):
    plt.figure(figsize=(10,5))

    plt.subplot(1, 2, 1)
    plt.plot(fitness_history, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Fitness per Generation')
    plt.show()

def visualize_network_with_weight_categories(network, epoch, filename="neural_network_structure_epoch.png"):

    input_size = network.input_size
    hidden_size = len([g for g in network.genes if g.input_node < input_size])  
    output_size = network.output_size
    
    in_x, in_y = [0.0] * input_size, np.linspace(0.1, 0.9, input_size)
    hid_x, hid_y = [0.5] * hidden_size, np.linspace(0.05, 0.95, hidden_size)
    out_x, out_y = [1.0] * output_size, np.linspace(0.3, 0.7, output_size)

    # Создание графика
    plt.figure(figsize=(8, 6))

    for i, (x0, y0) in enumerate(zip(in_x, in_y)):
        for j, (x1, y1) in enumerate(zip(hid_x, hid_y)):
            weight = next((gene.weight for gene in network.genes if gene.input_node == i and gene.output_node == input_size + j), 0)
            line_width = 0.5 + 4.5 * abs(weight) / np.max(np.abs([g.weight for g in network.genes]))  
            line_color = 'red' if abs(weight) > 1 else 'gray'  
            plt.plot([x0, x1], [y0, y1], color=line_color, alpha=0.7, linewidth=line_width)


    for i, (x0, y0) in enumerate(zip(hid_x, hid_y)):
        for j, (x1, y1) in enumerate(zip(out_x, out_y)):
            weight = next((gene.weight for gene in network.genes if gene.input_node == input_size + i and gene.output_node == input_size + hidden_size + j), 0)
            line_width = 0.5 + 4.5 * abs(weight) / np.max(np.abs([g.weight for g in network.genes]))  
            line_color = 'red' if abs(weight) > 1 else 'gray'  
            plt.plot([x0, x1], [y0, y1], color=line_color, alpha=0.7, linewidth=line_width)

    plt.scatter(in_x, in_y, s=200, color='blue', edgecolors='black', zorder=3)
    for i in range(input_size):
        plt.text(in_x[i] - 0.03, in_y[i], str(i), fontsize=9, ha='right', va='center', color='white')

    plt.scatter(hid_x, hid_y, s=200, color='orange', edgecolors='black', zorder=3)
    for i in range(hidden_size):
        plt.text(hid_x[i], hid_y[i], str(i), fontsize=9, ha='center', va='center', color='black')

    plt.scatter(out_x, out_y, s=200, color='green', edgecolors='black', zorder=3)
    for i in range(output_size):
        plt.text(out_x[i] + 0.03, out_y[i], str(i), fontsize=9, ha='left', va='center', color='black')

    plt.axis('off')
    plt.title('Network Structure')
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.show()
