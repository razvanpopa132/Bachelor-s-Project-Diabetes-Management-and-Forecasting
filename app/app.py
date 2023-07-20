import base64

from flask import Flask, render_template, request, jsonify
from methods import *
app = Flask(__name__,template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
import warnings
warnings.filterwarnings("ignore")

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global df

    if request.method == 'POST':
        file = request.files['csv_file']
        df = pd.read_csv(file, nrows=46000)
        df = df.set_index('DAYTIME')
        df.index = pd.to_datetime(df.index)

    return jsonify({'message': 'CSV uploaded successfully'})

@app.route('/simple-graph', methods=['POST'])
def simple_graph():
    global df
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        min_glucose = request.form['min_glucose']
        max_glucose = request.form['max_glucose']
        average_max_min=[]
        average_glucose=get_average_glucose(df,start_date,end_date)
        img = create_simple_graph(start_date, end_date, int(min_glucose), int(max_glucose), df)
        plot_data = base64.b64encode(img.getvalue()).decode('ascii')
        Result = get_areas(start_date, end_date, int(min_glucose), int(max_glucose), df)
        maxGlucose = get_max_glucose_value(start_date, end_date, df)
        minGlucose = get_min_glucose_value(start_date, end_date, df)
        average_max_min.append(average_glucose)
        average_max_min.append(maxGlucose)
        average_max_min.append(minGlucose)
        percentages=calculate_percentages(df,int(min_glucose),int(max_glucose),start_date,end_date)
        average_max_min.append(percentages[0])
        average_max_min.append(percentages[1])
        average_max_min.append(percentages[2])
        return jsonify({'plot_data': plot_data,
                        'Result': Result,
                        'average_max_min':average_max_min
                        })

    return jsonify({'message': 'Invalid request'})


@app.route('/create-predictions', methods=['POST'])
def create_predictions():
    global df

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        predictions=create_predictions_option(start_date, end_date, df)

        arima_prediction= predictions[0]
        predicted_values = [int(value) for value in predictions[1]]
        actual_values = [int(value) for value in predictions[2]]



        plot_data = base64.b64encode(arima_prediction.getvalue()).decode('ascii')


        return jsonify({'plot_data': plot_data,
                        "predicted_values":predicted_values,
                        "actual_values":actual_values})

    return jsonify({'message': 'Invalid request'})

@app.route('/create-predictionsAR', methods=['POST'])
def create_predictionsAR():
    global df

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        pacf_plot=create_pacf_plot(start_date,end_date,df)
        plot_data1 = base64.b64encode(pacf_plot.getvalue()).decode('ascii')

        ar_model=create_ar_automated_model(start_date,end_date,df)
        plot_data2=base64.b64encode(ar_model.getvalue()).decode('ascii')
        ar_predictions=create_ar_automated_prediction(df, start_date,end_date)

        predictions = ar_predictions[0]
        predicted_values = [int(value) for value in ar_predictions[1]]
        actual_values = [int(value) for value in ar_predictions[2]]
        plot_data3=base64.b64encode(predictions.getvalue()).decode('ascii')



        return jsonify({'plot_data1': plot_data1,
                        "plot_data2":plot_data2,
                        "plot_data3":plot_data3,
                        "predicted_values": predicted_values,
                        "actual_values": actual_values})

    return jsonify({'message': 'Invalid request'})

@app.route('/create-modelMA', methods=['POST'])
def create_modelMA():
    global df
    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        acf_plot=create_acf_plot(start_date,end_date,df)
        plot_data1 = base64.b64encode(acf_plot.getvalue()).decode('ascii')
        ma_model=create_ma_model(start_date,end_date,df,'BG',find_best_window(df,start_date,end_date,'BG'))
        plot_data2=base64.b64encode(ma_model.getvalue()).decode('ascii')
        return jsonify({'plot_data1': plot_data1,
                        "plot_data2":plot_data2,
                        })
    return jsonify({'message': 'Invalid request'})


@app.route('/create-modelARMA', methods=['POST'])
def create_modelARMA():
    global df

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        arma=create_arma_automated_prediction(start_date,end_date,df)

        arma_model=create_arma_automated_model(start_date,end_date,df)
        plot_data1 = base64.b64encode(arma_model.getvalue()).decode('ascii')

        arma_predictions = arma[0]
        predicted_values = [int(value) for value in arma[1]]
        actual_values = [int(value) for value in arma[2]]





        plot_data2=base64.b64encode(arma_predictions.getvalue()).decode('ascii')



        return jsonify({'plot_data1': plot_data1,
                        "plot_data2":plot_data2,
                        "predicted_values":predicted_values,
                        "actual_values":actual_values
                        })

    return jsonify({'message': 'Invalid request'})


@app.route('/create-modelARIMA', methods=['POST'])
def create_modelARIMA():
    global df

    if request.method == 'POST':
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        arima=create_arima_automated_prediction(start_date,end_date,df)

        arima_model=create_arima_automated_model(start_date,end_date,df)
        plot_data1 = base64.b64encode(arima_model.getvalue()).decode('ascii')

        arima_predictions = arima[0]
        predicted_values = [int(value) for value in arima[1]]
        actual_values = [int(value) for value in arima[2]]





        plot_data2=base64.b64encode(arima_predictions.getvalue()).decode('ascii')



        return jsonify({'plot_data1': plot_data1,
                        "plot_data2":plot_data2,
                        "predicted_values":predicted_values,
                        "actual_values":actual_values
                        })

    return jsonify({'message': 'Invalid request'})





if __name__ == '__main__':
    app.run(debug=True)