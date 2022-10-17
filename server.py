from flask import Flask, render_template, request

app = Flask(__name__)


@app.get('/',)
def get_main_page():
    return render_template('index.html')


@app.route('/process_data', methods=['POST'])
def process_data():
    print(request.form)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
