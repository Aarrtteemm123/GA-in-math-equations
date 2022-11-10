import os

from flask import Flask, render_template, request, redirect, url_for, session
from config import Config
from forms import GAForm
from ga_interface import GAInterface

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY


@app.get('/')
def get_main_page():
    view_data = {
        'num_generations': Config.NUM_GENERATIONS,
        'num_parents_mating': Config.NUM_PARENTS_MATING,
        'num_genes': Config.NUM_GENES,
        'sol_per_pop': Config.SOL_PER_POP,
        'crossover_type': Config.CROSSOVER_TYPE,
        'mutation_probability': Config.MUTATION_PROBABILITY,
        'parallel_processing': Config.PARALLEL_PROCESSING,
        'accuracy': Config.ACCURACY
    }
    return render_template('index.html', data=view_data)


@app.get('/result')
def get_result_page():
    data = session['result']
    print(data)
    return render_template('result.html', data=data)


@app.route('/process_data', methods=['POST'])
def process_data():
    ga_form = GAForm(request.form)
    if not ga_form.validate():
        return redirect(url_for('get_main_page'))
    ga_data = ga_form.data
    ga = GAInterface(ga_data.pop('equation'))
    ga_data['parallel_processing'] = os.cpu_count() if ga_data.pop('parallel_processing') else 1
    if Config.PARALLEL_PROCESSING > 1 and ga_data['parallel_processing'] > 1:
        ga_data['parallel_processing'] = Config.PARALLEL_PROCESSING
    ga.build_solver(**ga_data)
    execution_time = ga.run_solver()
    result_data = ga.get_result()
    ga.get_progress_figure().savefig('static/my_plot.png')
    result_data['execution_time'] = execution_time
    session['result'] = result_data
    return redirect(url_for('get_result_page'))


if __name__ == '__main__':
    app.run()
