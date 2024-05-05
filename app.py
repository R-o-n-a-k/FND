from flask import Flask, request, jsonify, render_template
import pickle
import os
from flask import send_from_directory

app = Flask(__name__)
load_model = pickle.load(open('NPAfinal_model.sav', 'rb'))

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')
    
@app.route("/")
def hello():
    return render_template("index.html")


@app.route("/predicted", methods=['POST'])
def predicted():
    # FOR RENDING RESULT
    input_news = request.form["news"]
    output = load_model.predict([input_news])

    return render_template('index.html', prediction_text='The News is: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
