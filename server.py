import os

from flask import Flask, render_template, request, redirect, url_for, session
from config import Config
from forms import GAForm
from ga_interface import GAInterface

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY


@app.get('/')
def get_main_page():
    return render_template('index.html')


@app.get('/result')
def get_result_page():
    data = session['result']
    print(data)
    return render_template('result.html', data=data)


@app.route('/process_data', methods=['POST'])
def process_data():
    print(request.form)
    ga_form = GAForm(request.form)
    if not ga_form.validate():
        return redirect(url_for('get_main_page'))
    ga_data = ga_form.data
    ga = GAInterface(ga_data.pop('equation'))
    ga_data['parallel_processing'] = os.cpu_count() if ga_data.pop('parallel_processing') else 1
    ga.build_solver(**ga_data)
    execution_time = ga.run_solver()
    result_data = ga.get_result()
    result_data['execution_time'] = execution_time
    return redirect(url_for('get_result_page'))


if __name__ == '__main__':
    app.run()
