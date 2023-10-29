# app/api.py
from flask import Flask, request, jsonify
from modules.preprocessing import preprocess_text
from modules.clustering import cluster_data_nmf
import os
import tempfile


app = Flask(__name__)


@app.route("/requests", methods=["POST"])
def cluster_issues():
    if request.method == "POST":
        # Check if "input_file" and "model_name" parameters are in the request
        if "input_file" not in request.args:
            return jsonify({"error": "'input_file' parameter is required!"}), 400
        if "model_name" not in request.args:
            return jsonify({"error": "'model_name' parameter is required!"}), 400

        input_file = request.args.get("input_file")
        model_name = request.args.get("model_name")
        # Check if the model_name is valid (e.g., 'kmeans', 'nmf')
        if model_name not in ["kmeans", "nmf"]:
            return jsonify({"error": "Invalid model_name"}), 400

        # Check if the input file name exists
        if not os.path.exists(input_file):
            return jsonify({"error": "Input file not found"}), 404

        data = preprocess_text(input_file)

        # Load the model based on model_name (implement model loading)
        if model_name == "nmf":
            clustered_data = cluster_data_nmf(data)

        # Save the results
        result_file = os.path.join("./data/output/", "output.csv")
        clustered_data.to_csv(result_file, index=False)

        return jsonify({"result_path": f"Results saved in: {result_file}"})

    else:
        return jsonify({"error": "not a post request"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
