import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run number selection from 1 to 5")
    parser.add_argument('run_num', nargs='?', type=int, help="An integer from 1 to 5")
    return parser.parse_args()


def run_calculations(run_num):
    print("Selected run: ", run_num)

    df = pd.read_csv(f'./data/run{run_num}.csv')
    
    if list(df.columns) != ['Arrival time', 'Preprocessing start', 'Preprocessing end/Classification start', 'Classification end']:
        print("Error: Unexpected data headers")
        exit(1)
    
    # Seperate 'Preprocessing end' and 'Classification start'
    df = df.rename(columns={'Preprocessing end/Classification start': 'Preprocessing end'})
    df['Classification start'] = df['Preprocessing end']
    
    # The difference between arrival times denotes the time taken by the Bluetooth transmission 
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
    
    return df


def get_statistical_results(data, name, run_num):
    mean = np.mean(data)
    median = np.median(data)
    variance = np.var(data)
    std = np.std(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    minimum = np.min(data)
    maximum = np.max(data)

    print(f"{name} (Run #{run_num}) stats:")
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
    plt.xlabel('Sample')
    plt.ylabel('Time (ms)')
    plt.title(f'{name} (Run #{run_num})')
    
    plt.axhline(mean, color='orange', linestyle='--', label='Mean')
    plt.axhline(median, color='red', linestyle='--', label='Median')
    plt.axhline(percentile_25, color='green', linestyle='--', label='25th Percentile')
    plt.axhline(percentile_75, color='purple', linestyle='--', label='75th Percentile')
    plt.legend()
    os.makedirs(f'run{run_num}', exist_ok=True)
    plt.savefig(f'./run{run_num}/{name}.png')
    plt.close()
    print()
    
    return {
        "run": run_num,
        "mean": mean,
        "median": median,
        "variance": variance,
        "std": std,
        "percentile_25": percentile_25,
        "percentile_75": percentile_75,
        "minimum": minimum,
        "maximum": maximum,
    }


def cross_run_results():
    multi_run_df = pd.DataFrame()
    target_metrics = ['Bluetooth transmission duration', 'Preprocessing service time', 'Classification service time', 'Phone busy time', 'Phone idle time']
    plot_columns = [[] for _ in range(len(target_metrics))]
    statistics = {}
    for run in range(1, 6):
        df = run_calculations(run)
        for i, target in enumerate(target_metrics):
            statistics[(run, target)] = get_statistical_results(df[target], target.title(), run)
            multi_run_df[f'{target} (Run #{run})'] = df[target]
            plot_columns[i].append(f'{target} (Run #{run})')
    
    for pc in plot_columns:
        multi_series_plot(multi_run_df, pc)
        
    statistical_plots(statistics)
    
    
def multi_series_plot(df, columns):
    x = range(df.shape[0])
    name = columns[0][:-9].title()
    
    sns.set_theme(style='whitegrid')
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.figure(figsize=(12, 6))
    
    for col in columns:
        sns.lineplot(x=x, y=df[col], label=col)

    plt.xlabel('Sample')
    plt.ylabel('Time (ms)')
    plt.title(name)

    # Show the plot
    plt.legend()
    plt.savefig(f'{name}.png')
    plt.close()
    
    plt.figure(figsize=(6, 7))
    sns.boxplot(data=df[columns])
    # Replace labels for better readability
    plt.xticks(ticks=plt.xticks()[0], labels=['Run #1', 'Run #2', 'Run #3', 'Run #4', 'Run #5'], rotation=45, ha='right')
    plt.title(name)
    os.makedirs(f'boxplots', exist_ok=True)
    plt.savefig(f'./boxplots/{name} Box Plot.png')
    plt.close()
    

def statistical_plots(stats):
    # Pie plot w/ phone busy/idle time
    target_metrics = ['Phone idle time', 'Phone busy time']
    
    # Gets the mean from each run and gets an overall mean for each of the selected metrics
    means = [np.mean([stats[k]['mean'] for k in stats if k[1] == target]) for target in target_metrics]

    # explode the smallest mean value
    min_index = means.index(min(means))
    explode = [0] * len(target_metrics)
    explode[min_index] = 0.1  # Set explode value to 0.1 for the smallest value

    plt.figure(figsize=(11, 8))
    wedges, _, _ = plt.pie(means, labels=target_metrics, explode=explode, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
    total = np.sum(means)
    legend_labels = [f'Mean {metric.title()}: {value:.3f}ms ({value/total*100:.1f}%)' for metric, value in zip(target_metrics, means)]
    plt.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=10)
    plt.title('Phone Busy and Idle Time (ms)')
    plt.axis('equal')
    plt.savefig(f'Phone Busy and Idle Time Pie Plot.png')
    plt.close()
    
    
    # Pie plot w/ transaction stages mean time
    target_metrics = ['Bluetooth transmission duration', 'Preprocessing service time', 'Classification service time']
    
    # Gets the mean from each run and gets an overall mean for each of the selected metrics
    means = [np.mean([stats[k]['mean'] for k in stats if k[1] == target]) for target in target_metrics]

    # explode the smallest mean value
    min_index = means.index(min(means))
    explode = [0] * len(target_metrics)
    explode[min_index] = 0.1  # Set explode value to 0.1 for the smallest value

    # Create a pie chart using Matplotlib
    plt.figure(figsize=(11, 8))
    # plt.pie(means, labels=target_metrics, autopct='%1.1f%%', startangle=140)
    wedges, _, _ = plt.pie(means, labels=target_metrics, explode=explode, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12})
    total = np.sum(means)
    legend_labels = [f'Mean {metric.title()}: {value:.3f}ms ({value/total*100:.1f}%)' for metric, value in zip(target_metrics, means)]
    plt.legend(wedges, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), fontsize=10)
    plt.title('Mean Service Times (ms)')
    plt.axis('equal')
    plt.savefig(f'Mean Service Times Pie Plot.png')
    plt.close()
    
    
    # Bar plot w/ mean values and CI
    # target_metrics = ['Bluetooth transmission duration', 'Preprocessing service time', 'Classification service time']
    target_metrics = ['Preprocessing service time', 'Classification service time']
    repeated_targets = target_metrics * 5
    
    # Gets the mean from each run
    means = [stats[k]['mean'] for k in stats for target in target_metrics if k[1] == target]
    
    plt.figure(figsize=(7, 7))
    sns.barplot(x=repeated_targets, y=means, errorbar=("ci", 95), width=0.5)
    plt.title('Mean Preprocessing and Classification Service Time (95% CI)')
    plt.ylabel('Time (ms)')
    plt.savefig(f'Mean Preprocessing and Classification Service Time CI.png')
    plt.close()
    
    
    # Bar plot w/ mean values and CI
    target_metrics = ['Bluetooth transmission duration']
    repeated_targets = target_metrics * 5
    
    # Gets the mean from each run
    means = [stats[k]['mean'] for k in stats for target in target_metrics if k[1] == target]
    
    plt.figure(figsize=(7, 7))
    sns.barplot(x=repeated_targets, y=means, errorbar=("ci", 95), width=0.25)
    plt.title('Mean Bluetooth Transmission Duration (95% CI)')
    plt.ylabel('Time (ms)')
    plt.savefig(f'Mean Bluetooth Transmission Duration CI.png')
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.run_num is None:
        cross_run_results()
    elif args.run_num < 1 or args.run_num > 5:
        df = run_calculations(args.run_num)
        target_metrics = ['Bluetooth transmission duration', 'Preprocessing service time', 'Classification service time', 'Phone busy time', 'Phone idle time']
        for target in target_metrics:
            get_statistical_results(df[target], target.title(), args.run_num)

    else:
        print("Error: Run selected must be between 1 and 5")
        exit(1)

