import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def get_statistical_results(data, name, run_num):
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    std = np.std(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    minimum = np.min(data)
    maximum = np.max(data)

    print(f"{name} stats:")
    print("Mean: ", mean)
    print("Median: ", median)
    print("Variance: ", variance)
    print("Standard Deviation: ", std)
    print("25th Percentile: ", percentile_25)
    print("75th Percentile: ", percentile_75)
    print("Minimum: ", minimum)
    print("Maximum: ", maximum)

    sns.set_theme(style='whitegrid')
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.figure(figsize=(12, 6))

    # Create a Seaborn plot
    sns.lineplot(x=range(data.size), y=data, err_style='bars', errorbar='sd')
    plt.xlabel('Sample #')
    plt.ylabel('Time (ms)')
    plt.title(f'{name} (Run #{run_num})')
    # plt.savefig(f'{name}_results.png')
    
    plt.axhline(mean, color='orange', linestyle='--', label='Mean')
    plt.axhline(median, color='red', linestyle='--', label='Median')
    plt.axhline(percentile_25, color='green', linestyle='--', label='25th Percentile')
    plt.axhline(percentile_75, color='purple', linestyle='--', label='75th Percentile')
    # plt.ylim(median - variance*25, median + variance*25)
    plt.legend()
    # plt.savefig(f'{name}_results_zoom.png')
    os.makedirs(f'run{run_num}', exist_ok=True)
    plt.savefig(f'./run{run_num}/{name}.png')
    print()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run number selection from 1 to 5")
    parser.add_argument('run_num', type=int, help="An integer from 1 to 5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.run_num < 1 or args.run_num > 5:
        print("Error: Run selected must be between 1 and 5")
        exit(1)
    
    print("Selected run: ", args.run_num)

    df = pd.read_csv(f'./data/run{args.run_num}.csv')
    
    if list(df.columns) != ['Arrival time', 'Preprocessing start', 'Preprocessing end/Classification start', 'Classification end']:
        print("Error: Unexpected data headers")
        exit(1)
    
    # Seperate 'Preprocessing end' and 'Classification start'
    df = df.rename(columns={'Preprocessing end/Classification start': 'Preprocessing end'})
    df['Classification start'] = df['Preprocessing end']
    
    # The difference between arrival times denotes the time taken by the blutooth transmission 
    df['Bluetooth transmission duration'] = df['Arrival time'].diff()
    
    # To avoid the synchronization issues between the time on smartwatch and on the phone,
    # The Arrival time was recorded AFTER bluetooth transmission, but in reality,
    # the transaction begins before the transmission.
    # To address this discrepancy we can shift the Arrival times down by 1, and eliminate the first row
    df['Arrival time'] = df['Arrival time'].shift(-1)
    df.drop(df.index[0], inplace=True)
    
    # Service times
    df['Preprocessing service time'] = df['Preprocessing end'] - df['Preprocessing start']
    df['Classification service time'] = df['Classification end'] - df['Classification start']

    df["Phone busy time"] = df['Classification end'] - df['Preprocessing start']
    df["Phone idle time"] = df['Bluetooth transmission duration'] - df["Phone busy time"]
    
    get_statistical_results(df['Bluetooth transmission duration'], f'Bluetooth Transmission Duration', args.run_num)
    get_statistical_results(df['Preprocessing service time'], f'Preprocessing Service Time', args.run_num)
    get_statistical_results(df['Classification service time'], f'Classification Service Time', args.run_num)
    get_statistical_results(df['Phone busy time'], f'Phone Busy Time', args.run_num)
    get_statistical_results(df['Phone idle time'], f'Phone Idle Time', args.run_num)





