from loadModelCNN import *

from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('watchyourlang.html')

@app.route("/normalize/<myText>", methods=['POST'])
def normalize(myText):
	if len(myText) == 0:
		return "Text missing. Please provide text to speak."
	return normalize_text(myText)

app.run(debug=True)