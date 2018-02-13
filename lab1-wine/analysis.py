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
    selected.plot.hist()
    plt.xlabel('Metric')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    if (len(sys.argv) < 3):
        print("Usage:")
        print("  analysis.py <filename> <column> [<column> ...]")
        sys.exit(0)

    inputFile = sys.argv[1]
    data = main(inputFile, [x.lower() for x in sys.argv[2:]])
