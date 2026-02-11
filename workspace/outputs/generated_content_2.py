from flask import Flask, jsonify

app = Flask(__name__)


@app.errorhandler(401)
def unauthorized_error(e):
    return jsonify({"error": "Unauthorized"}), 401


@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({"error": "Internal Server Error"}), 500
