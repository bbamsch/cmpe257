import sys
import pandas as pd
import matplotlib.pyplot as plt

def main(inputFile, columns):
    data = pd.read_csv(
        inputFile,
        sep=',',
        header=0)

    # Transform all columns to lowercase
    data = data.rename(columns=str.lower)

    # Select columns & print mean, median and variance
    selected = data[columns]
    print(selected.mean())
    print(selected.median())
    print(selected.var())

    # Build Histogram
    plt.figure(0)
    selected.plot(kind='hist', alpha=0.8)
    plt.title('Histogram')
    plt.xlabel('Metric')
    plt.ylabel('Frequency')

    plt.figure(1)
    selected.plot(kind='box')
    plt.title('Box')
    plt.xlabel('Metric')
    plt.ylabel('Value')

    plt.show()

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage:")
        print("  analysis.py <filename> <column> [<column> ...]")
        sys.exit(0)

    inputFile = sys.argv[1]
    data = main(inputFile, [x.lower() for x in sys.argv[2:]])
