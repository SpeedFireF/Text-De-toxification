import matplotlib.pyplot as plt


def build_chart(data, path='/Users/damirabdulaev/Desktop/Hypothesis2.png'):
    # X-axis labels (you can customize them)
    labels = ['Toxicity', 'Similarity', 'Fluency', 'J metric']

    # Create a bar chart
    plt.bar(labels, data)
    plt.ylim(0, 1)

    # Add labels and a title
    plt.xlabel('Metrics')
    plt.ylabel('Metric values')
    plt.title('Hypothesis 2')

    # Save the chart as a PNG file
    plt.savefig(path)

    # Show the chart
    plt.show()
