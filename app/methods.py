from cmath import sqrt

import pandas as pd
import numpy as np
import matplotlib
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg, AR
from statsmodels.tsa.arima.model import ARIMA

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import io
from pandas.plotting import lag_plot

color_pal = sns.color_palette()







def get_max_glucose_value(start_date,end_date,df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    glucoseList=list(graph['BG'])
    return max(glucoseList)

def get_min_glucose_value(start_date,end_date,df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    glucoseList=list(graph['BG'])
    return min(glucoseList)






def get_areas(start_date, end_date, minimum, maximum, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    normal_duration = calculate_period_duration(graph, minimum, maximum)
    high_duration = calculate_period_duration(graph, maximum + 1, float('inf'))
    low_duration = calculate_period_duration(graph, float('-inf'), minimum - 1)


    high_duration_formatted = convert_to_hours_minutes(high_duration)
    normal_duration_formatted = convert_to_hours_minutes(normal_duration)
    low_duration_formatted = convert_to_hours_minutes(low_duration)

    result = [
        f"{high_duration_formatted} ",
        f"{normal_duration_formatted} ",
        f"{low_duration_formatted} "
    ]

    return result


def calculate_period_duration(graph, minimum, maximum):
    duration = pd.Timedelta(0)
    start_time = None

    for i in range(len(graph)):
        if minimum <= graph['BG'][i] <= maximum:
            if start_time is None:
                start_time = graph.index[i]
        else:
            if start_time is not None:
                end_time = graph.index[i - 1]
                next_start_time = graph.index[i] if i < len(graph) - 1 else None

                if next_start_time is not None:
                    duration += next_start_time - start_time
                else:
                    duration += end_time - start_time

                start_time = None


    if start_time is not None:
        end_time = graph.index[-1]
        duration += end_time - start_time

    return duration.total_seconds() // 60





def convert_to_hours_minutes(duration_minutes):
    hours = duration_minutes // 60
    minutes = duration_minutes % 60

    if hours == 0:
        return f"{minutes} minute"
    elif minutes == 0:
        return f"{hours} ore"
    else:
        return f"{hours} ore și {minutes} minute"


def get_average_glucose(df,start_date,end_date):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    average = graph.mean()
    return int(average.iloc[0]) if int else 0

def calculate_percentages(df,minimum,maximum,start_date,end_date):
    numbers=df.loc[(df.index >= start_date) & (df.index <= end_date)]
    total_count = len(numbers)
    less_than_min_count = len([num for num in range(0, len(numbers['BG'])) if numbers['BG'][num] < minimum])
    normal_count = len([num for num in range(0, len(numbers['BG'])) if minimum <= numbers['BG'][num] <= maximum])
    greater_than_max_count = len([num for num in range(0, len(numbers['BG'])) if numbers['BG'][num] > maximum])

    less_than_min_percentage = round((less_than_min_count / total_count) * 100,2)
    normal_percentage = round((normal_count / total_count) * 100,2)
    greater_than_max_percentage = round((greater_than_max_count / total_count) * 100,2)
    return less_than_min_percentage, normal_percentage, greater_than_max_percentage




def create_simple_graph(start_date, end_date, minimum, maximum, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    maxListDates = []
    minListDates = []
    minListBG = []
    maxListBG = []


    graph.plot(style='-', figsize=(15, 5))
    plt.axhline(y=minimum, color='r', linestyle='-')
    plt.axhline(y=maximum, color='r', linestyle='-')
    for i in range(0, len(graph['BG'])):
        if graph['BG'][i] < minimum:
            if graph['BG'][i - 1] > maximum:
                plt.fill_between(maxListDates, maxListBG, maximum, facecolor="red", alpha=0.5)

                maxListDates = []
                maxListBG = []

            minListDates.append(graph['BG'].index[i])
            minListBG.append(graph['BG'][i])
            plt.scatter(graph['BG'].index[i], graph['BG'][i], c="red", zorder=10, alpha=0.5)


        elif graph['BG'][i] > maximum:
            if graph['BG'][i - 1] < minimum:
                plt.fill_between(minListDates, minListBG, minimum, facecolor="red", alpha=0.5)

                minListDates = []
                minListBG = []

            maxListDates.append(graph['BG'].index[i])
            plt.scatter(graph['BG'].index[i], graph['BG'][i], c="red", zorder=10, alpha=0.5)
            maxListBG.append(graph['BG'][i])
        else:

            plt.scatter(graph['BG'].index[i], graph['BG'][i], c="green", zorder=10)
            if graph['BG'][i - 1] < minimum:
                plt.fill_between(minListDates, minListBG, minimum, facecolor="red", alpha=0.5)

                minListDates = []
                minListBG = []


            elif graph['BG'][i - 1] > maximum:
                plt.fill_between(maxListDates, maxListBG, maximum, facecolor="red", alpha=0.5)

                maxListDates = []
                maxListBG = []
        if i == len(graph['BG']) - 1:
            if len(maxListDates) != 0:
                plt.fill_between(maxListDates, maxListBG, maximum, facecolor="red", alpha=0.5)
            else:
                plt.fill_between(minListDates, minListBG, minimum, facecolor="red", alpha=0.5)
    plt.title('Grafic zone anormale de la ' + start_date + ' până la ' + end_date + " glicemie minimă="+str(minimum)+" glicemie maximă="+str(maximum))
    plt.ylabel('Glucose')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf



def create_lag_plot(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    lag_plot(graph)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf


def create_acf_plot(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    plot_acf(graph, lags=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf

def create_pacf_plot(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    plot_pacf(graph, lags=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf


def create_graph(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    graph.plot(style='-', figsize=(15, 5))
    plt.title('Glucose measurements from ' + start_date + ' to ' + end_date)
    plt.ylabel('Glucose')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf




def find_best_p(df, train_ratio=0.8):

    n = len(df)
    train_size = int(n * train_ratio)
    train_data = df.iloc[:train_size]



    p_values = range(0, 6)
    best_aic = float('inf')
    best_p = None

    for p in p_values:
        model = ARIMA(train_data['BG'], order=(p, 0, 0))
        model_fit = model.fit()
        aic = model_fit.aic
        if aic < best_aic:
            best_aic = aic
            best_p = p

    return best_p

def create_ar_automated_prediction(df, start_date,end_date,train_ratio=0.8):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    p = find_best_p(graph, train_ratio)

    predicted_values = []
    actual_values = []


    n = len(graph)
    train_size = int(n * train_ratio)
    train_data = graph.iloc[:train_size].dropna()
    test_data = graph.iloc[train_size:].dropna()


    model = AutoReg(train_data['BG'], lags=p)


    model_fit = model.fit()



    test_predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) -1)

    plt.figure(figsize=(15, 5))

    counter=0
    for i in range(0,len(test_data)):
        if i==0:
            plt.scatter(test_data.index[i], test_data['BG'][i], c="green",label="Valori reale")

            if len(actual_values) <= 10:
                actual_values.append(test_data['BG'][i])
        else:
            plt.scatter(test_data.index[i], test_data['BG'][i], c="green")

            if len(actual_values) <= 10:
                actual_values.append(test_data['BG'][i])
        counter+=1
        if counter==10:
            break
    counter=0
    trend=None
    trend_changed=False


    if not isinstance(test_predictions.index,pd.core.indexes.datetimes.DatetimeIndex):
        for i in test_predictions.index:
            if counter == 0:
                plt.scatter(test_data.index[counter], test_predictions[i], c="red", label="Valori prezise")

                if len(predicted_values) <= 10:
                    predicted_values.append(int(test_predictions[i]))
            else:
                plt.scatter(test_data.index[counter], test_predictions[i], c="red")

                if len(predicted_values) <= 10:
                    predicted_values.append(int(test_predictions[i]))
            if test_predictions[i] < test_predictions[i + 1]:
                if trend is None:
                    trend = "ascendent"
                elif trend == "descendent":
                    trend_changed = True

            elif test_predictions[i] > test_predictions[i + 1]:
                if trend is None:
                    trend = "descendent"
                elif trend == "ascendent":
                    trend_changed = True
            plt.plot([test_data.index[counter], test_data.index[counter]],
                     [test_data['BG'][counter], test_predictions[i]], c="blue")
            counter += 1
            if counter == 10:
                break
        if trend_changed:
            trend = "mix"
        else:
            pass
    else:
        for i in range(0,len(test_predictions)):
            if counter == 0:
                plt.scatter(test_predictions.index[i], test_predictions[i], c="red", label="Valori prezise")

            else:
                plt.scatter(test_predictions.index[i], test_predictions[i], c="red")
            if test_predictions[i] < test_predictions[i + 1]:
                if trend is None:
                    trend = "ascendent"
                elif trend == "descendent":
                    trend_changed = True

            elif test_predictions[i] > test_predictions[i + 1]:
                if trend is None:
                    trend = "descendent"
                elif trend == "ascendent":
                    trend_changed = True
            plt.plot([test_data.index[counter], test_data.index[counter]],
                     [test_data['BG'][counter], test_predictions[i]], c="blue")
            counter += 1
            if counter == 10:
                break
        if trend_changed:
            trend = "mix"
        else:
            pass



    plt.xlabel('Date')
    plt.ylabel('BG')
    plt.title(f'Predictii AR cu valoarea ordinului p de {p}, trend {trend} detectat')

    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf,predicted_values,actual_values



def create_ma_model(start_date, end_date, df, num_col, window):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    window = int(window)

    rolling_avg = graph[num_col].rolling(window=window).mean()

    plt.figure(figsize=(15, 5))
    plt.plot(graph[num_col], label=num_col)
    plt.plot(rolling_avg, label=f'Medie mobilă, de la ({start_date} până la {end_date})')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.title(f"Medie mobilă cu valoarea a ferestrei de {window}")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf


def find_best_window(df,start_date,end_date, target_column):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    graph.sort_index(inplace=True)
    min_rmse = float('inf')
    best_window = None

    max_window_size = 30

    for window in range(3, max_window_size):
        rolling_means = graph[target_column].rolling(window=window).mean().dropna()
        overlapping_data = graph.iloc[window-1:][target_column]
        rmse = np.sqrt(mean_squared_error(overlapping_data, rolling_means))
        if rmse < min_rmse:
            min_rmse = rmse
            best_window = window

    return best_window


def create_ar_automated_model(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    graph.sort_index(inplace=True)
    best_aic = float('inf')
    best_p = None
    for p in range(0, 5):
        try:
            model = AutoReg(graph, lags=p)
            results = model.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_p = p
        except:
            continue
    model = AutoReg(graph, lags=best_p)
    results = model.fit()
    plt.figure(figsize=(15, 6))
    plt.plot(graph, label='Valori înainte de prelucrare')
    plt.plot(results.fittedvalues, label='Valori după prelucrare')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.legend()
    plt.title('Model AR')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf

def create_arma_automated_model(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    graph.sort_index(inplace=True)
    best_aic = float('inf')
    best_p = None
    best_q = None
    for p in range(0, 5):
        for q in range(0, 5):
            try:
                model = ARIMA(graph, order=(p, 0, q))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
            except:
                continue
    model = ARIMA(graph, order=(best_p, 0, best_q), trend=None)
    results = model.fit()
    plt.figure(figsize=(15, 6))
    plt.plot(graph, label='Valori înainte de prelucrare')
    plt.plot(results.fittedvalues, label='Valori după prelucrare')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.legend()
    plt.title('Model ARMA')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf


def create_arma_automated_prediction(start_date, end_date, df, train_ratio=0.8, plot=True):
    predicted_values = []
    actual_values = []

    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    n = len(graph)
    train_size = int(n * train_ratio)
    train_data = graph.iloc[:train_size].dropna()
    test_data = graph.iloc[train_size:].dropna()


    p_range = range(0, 5)
    q_range = range(0, 5)


    best_aic = float('inf')
    best_p = None
    best_q = None
    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(train_data, order=(p, 0, q))
                results = model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic = aic
                    best_p = p
                    best_q = q
            except:
                continue


    model = ARIMA(train_data, order=(best_p, 0, best_q), trend=None)
    results = model.fit()


    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) -1)

    if plot:
        plt.figure(figsize=(15, 5))

        counter = 0
        for i in range(0, len(test_data)):
            if i == 0:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green", label="Valori reale")

                if len(actual_values) <= 10:
                    actual_values.append(test_data['BG'][i])
            else:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green")

                if len(actual_values) <= 10:
                    actual_values.append(test_data['BG'][i])
            counter += 1
            if counter == 10:
                break
        counter = 0
        trend = None
        trend_changed = False

        if not isinstance(test_predictions.index, pd.core.indexes.datetimes.DatetimeIndex):
            for i in test_predictions.index:
                if counter == 0:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red", label="Valori prezise")
                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))
                else:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red")
                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))
                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True

                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True
                plt.plot([test_data.index[counter], test_data.index[counter]],
                         [test_data['BG'][counter], test_predictions[i]], c="blue")
                counter += 1
                if counter == 10:
                    break
            if trend_changed:
                trend = "mix"
            else:
                pass
        else:
            for i in range(0, len(test_predictions)):
                if counter == 0:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red", label="Valori prezise")

                else:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red")
                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True

                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True
                plt.plot([test_data.index[counter], test_data.index[counter]],
                         [test_data['BG'][counter], test_predictions[i]], c="blue")
                counter += 1
                if counter == 10:
                    break
            if trend_changed:
                trend = "mix"
            else:
                pass


        plt.xlabel('Date')
        plt.ylabel('BG')
        plt.title(f'Predicții ARMA cu AR ({best_p}) și MA ({best_q}), trend {trend} detectat')

        plt.legend()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.clf()
        return buf,predicted_values,actual_values


def create_arima_automated_model(start_date, end_date, df):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    graph.sort_index(inplace=True)
    best_aic = float('inf')
    best_p = None
    best_q = None
    best_d = None

    for p in range(0, 5):
        for d in range(0, 3):
            for q in range(0, 5):
                try:
                    model = ARIMA(graph, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_d = d
                        best_q = q
                except:
                    continue

    model = ARIMA(graph, order=(best_p, best_d, best_q))
    results = model.fit()

    plt.figure(figsize=(15, 6))
    plt.plot(graph, label='Valori înainte de prelucrare')
    plt.plot(results.fittedvalues, label='Valori după prelucrare')
    plt.xlabel('Timp')
    plt.ylabel('Valoare')
    plt.legend()
    plt.title('Model ARIMA')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf

def create_arima_automated_prediction(start_date, end_date, df, train_ratio=0.8, plot=True):

    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    predicted_values=[]
    actual_values=[]

    n = len(graph)
    train_size = int(n * train_ratio)
    train_data = graph.iloc[:train_size].dropna()
    test_data = graph.iloc[train_size:].dropna()


    p_range = range(0, 5)
    d_range = range(0, 3)
    q_range = range(0, 5)


    best_aic = float('inf')
    best_p = None
    best_d = None
    best_q = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_d = d
                        best_q = q
                except:
                    continue




    model = ARIMA(train_data, order=(best_p, best_d, best_q))
    results = model.fit()


    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    if plot:
        plt.figure(figsize=(15, 5))

        counter = 0
        for i in range(0, len(test_data)):
            if i == 0:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green", label="Valori reale")

                if len(actual_values) <=10:
                    actual_values.append(test_data['BG'][i])
            else:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green")

                if len(actual_values) <=10:
                    actual_values.append(test_data['BG'][i])

            counter += 1
            if counter == 10:
                break
        counter = 0
        trend = None
        trend_changed = False

        if not isinstance(test_predictions.index, pd.core.indexes.datetimes.DatetimeIndex):
            for i in test_predictions.index:
                if counter == 0:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red", label="Valori prezise")

                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))
                else:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red")

                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))
                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True

                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True
                plt.plot([test_data.index[counter], test_data.index[counter]],
                         [test_data['BG'][counter], test_predictions[i]], c="blue")
                counter += 1
                if counter == 10:
                    break
            if trend_changed:
                trend = "mix"
            else:
                pass
        else:
            for i in range(0, len(test_predictions)):
                if counter == 0:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red", label="Valori prezise")


                else:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red")
                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True

                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True
                plt.plot([test_data.index[counter], test_data.index[counter]],
                         [test_data['BG'][counter], test_predictions[i]], c="blue")
                counter += 1
                if counter == 10:
                    break
            if trend_changed:
                trend = "mix"
            else:
                pass





    plt.xlabel('Date')
    plt.ylabel('BG')
    plt.title(f'Predicții ARIMA cu AR({best_p}), I({best_d}), MA({best_q}), trend {trend} detectat')

    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf,predicted_values,actual_values


def create_predictions_option(start_date, end_date, df, train_ratio=0.8, plot=True):
    graph = df.loc[(df.index >= start_date) & (df.index <= end_date)]

    predicted_values = []
    actual_values = []

    n = len(graph)
    train_size = int(n * train_ratio)
    train_data = graph.iloc[:train_size].dropna()
    test_data = graph.iloc[train_size:].dropna()

    p_range = range(0, 5)
    d_range = range(0, 3)
    q_range = range(0, 5)

    best_aic = float('inf')
    best_p = None
    best_d = None
    best_q = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_p = p
                        best_d = d
                        best_q = q
                except:
                    continue

    model = ARIMA(train_data, order=(best_p, best_d, best_q))
    results = model.fit()

    test_predictions = results.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

    if plot:
        plt.figure(figsize=(15, 5))

        counter = 0
        for i in range(len(test_data)):
            if i == 0:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green", label="Valori reale")
                if len(actual_values) <= 10:
                    actual_values.append(test_data['BG'][i])
            else:
                plt.scatter(test_data.index[i], test_data['BG'][i], c="green")
                if len(actual_values) <= 10:
                    actual_values.append(test_data['BG'][i])

            counter += 1
            if counter == 10:
                break

        counter = 0
        trend = None
        trend_changed = False

        if not isinstance(test_predictions.index, pd.core.indexes.datetimes.DatetimeIndex):
            for i in test_predictions.index:
                if counter == 0:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red", label="Valori prezise")
                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))
                else:
                    plt.scatter(test_data.index[counter], test_predictions[i], c="red")
                    if len(predicted_values) <= 10:
                        predicted_values.append(int(test_predictions[i]))

                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True
                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True


                plt.plot([test_data.index[counter], test_data.index[counter]], [test_data['BG'][counter], test_predictions[i]], c="blue")

                counter += 1
                if counter == 10:
                    break

            if trend_changed:
                trend = "mixed"
            else:
                pass
        else:
            for i in range(len(test_predictions)):
                if counter == 0:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red", label="Valori prezise")
                else:
                    plt.scatter(test_predictions.index[i], test_predictions[i], c="red")

                if test_predictions[i] < test_predictions[i + 1]:
                    if trend is None:
                        trend = "ascendent"
                    elif trend == "descendent":
                        trend_changed = True
                elif test_predictions[i] > test_predictions[i + 1]:
                    if trend is None:
                        trend = "descendent"
                    elif trend == "ascendent":
                        trend_changed = True


                plt.plot([test_data.index[counter], test_data.index[counter]], [test_data['BG'][counter], test_predictions[i]], c="blue")

                counter += 1
                if counter == 10:
                    break

            if trend_changed:
                trend = "mix"
            else:
                pass

    plt.xlabel('Date')
    plt.ylabel('BG')
    plt.title(f'Predicții realizate pe baza datelor între {start_date} și {end_date}, trend {trend} detectat')

    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.clf()
    return buf, predicted_values, actual_values
