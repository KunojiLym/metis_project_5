import flask
from flask import request
from flask_cors import CORS
from poetry_flask_api import generate_poem_api, generate_poem, to_html

# Initialize the app

app = flask.Flask(__name__)
CORS(app)

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!
@app.route("/")
def mainpage():
    return flask.render_template('index.html')


@app.route("/generate", methods=["POST", "GET"])
def generate():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    req = request.args
    poem_list = generate_poem_api(req)
    print("params = ", req)
    return to_html(poem_list)
    
@app.route("/generate_json", methods=["POST", "GET"])
def generate_json():
    req = request.get_json()
    poem_list = generate_poem(req)
    print("params = ", req)
    return to_html(poem_list)

# Start the server, continuously listen to requests.
# We'll have a running web app!

# For local development:
app.run(debug=True)

# For public web serving:
# app.run(host='0.0.0.0')
