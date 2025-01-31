import copy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx


def create_colormap():
    # Create a custom colormap to prevent repeating colors
    colors = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#5254a3',
        '#6b4c9a', '#8ca252', '#bd9e39', '#ad494a', '#636363',
        '#8c6d8c', '#9c9ede', '#cedb9c', '#e7ba52', '#e7cb94',
        '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173',
        '#a55194', '#ce6dbd', '#de9ed6', '#f1b6da', '#fde0ef',
        '#636363', '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0',
        '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d',
        '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
        '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc',
        '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
        '#f0f0f0', '#a63603', '#e6550d', '#fd8d3c', '#fdae6b',
        '#fdd0a2', '#31a354', '#74c476', '#a1d99b', '#c7e9c0',
        '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363',
        '#969696', '#bdbdbd', '#d9d9d9', '#f0f0f0', '#6a3d9a',
        '#8e7cc3', '#b5a0d8', '#ce6dbd', '#de9ed6', '#f1b6da',
        '#fde0ef', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'
    ]
    return mcolors.ListedColormap(colors)


def plot_gantt_chart(JobShop):
    # Plot the Gantt chart of the job shop schedule
    fig, ax = plt.subplots()
    colormap = create_colormap()

    for machine in JobShop.machines:
        machine_operations = sorted(machine._processed_operations, key=lambda op: op.scheduling_information['start_time'])
        for operation in machine_operations:
            operation_start = operation.scheduling_information['start_time']
            operation_end = operation.scheduling_information['end_time']
            operation_duration = operation_end - operation_start
            operation_label = f"{operation.operation_id}"

            # Set color based on job ID
            color_index = operation.job_id % len(JobShop.jobs)
            if color_index >= colormap.N:
                color_index = color_index % colormap.N
            color = colormap(color_index)

            ax.broken_barh(
                [(operation_start, operation_duration)],
                (machine.machine_id - 0.4, 0.8),
                facecolors=color,
                edgecolor='black'
            )

            setup_start = operation.scheduling_information['start_setup']
            setup_time = operation.scheduling_information['setup_time']
            if setup_time != None:
                ax.broken_barh(
                    [(setup_start, setup_time)],
                    (machine.machine_id - 0.4, 0.8),
                    facecolors='grey',
                    edgecolor='black', hatch='/')
            middle_of_operation = operation_start + operation_duration / 2
            ax.text(
                middle_of_operation,
                machine.machine_id,
                operation_label,
                ha='center',
                va='center',
                fontsize=8
            )

    fig = ax.figure
    fig.set_size_inches(12, 6)

    ax.set_yticks(range(JobShop.nr_of_machines))
    ax.set_yticklabels([f'M{machine_id+1}' for machine_id in range(JobShop.nr_of_machines)])
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    ax.grid(True)

    return plt


def draw_precedence_relations(JobShop):
    colormap = create_colormap()

    # Convert precedence relations into a usable format
    precedence_relations = copy.deepcopy(JobShop.precedence_relations_operations)
    for key, value in precedence_relations.items():
        value = [i.operation_id for i in value]
        precedence_relations[key] = value

    # Add nodes and edges to the graph
    G = nx.DiGraph()
    for key, value in precedence_relations.items():
        for successor in value:
            G.add_edge(successor, key)  # Reverse the edge direction

    # Assign levels to nodes using breadth-first search (BFS)
    levels = {}
    queue = [node for node in G.nodes if not list(G.predecessors(node))]
    for node in queue:
        levels[node] = 0

    while queue:
        current = queue.pop(0)
        for successor in G.successors(current):
            levels[successor] = max(levels.get(successor, 0), levels[current] + 1)
            queue.append(successor)

    # Group nodes by their levels
    level_nodes = {}
    for node, level in levels.items():
        level_nodes.setdefault(level, []).append(node)

    # Set positions for nodes
    pos = {}
    for level, nodes in level_nodes.items():
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (level, i - len(nodes) / 2)

    # Draw the graph
    options = {
        "font_size": 8,
        "node_size": 500,
        "node_color": [],
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    # Assign colors to nodes based on job ID
    for node in G.nodes:
        job_id = JobShop.get_operation(node).job_id
        options["node_color"].append(colormap(job_id % colormap.N))

    nx.draw_networkx(G, pos, **options)

    # Adjust plot settings
    plt.gca().margins(0.20)
    plt.gcf().set_size_inches(16, 8)
    plt.axis("off")
    plt.show()
