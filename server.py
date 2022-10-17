from flask import Flask, render_template, request, redirect, url_for, session
from config import Config

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
    form_data = request.form.to_dict()
    print(form_data)
    data = form_data
    session['result'] = data
    return redirect(url_for('get_result_page'))


if __name__ == '__main__':
    app.run()
