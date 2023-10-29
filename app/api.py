from flask import Flask, request, jsonify
from modules.preprocessing import preprocess_text
from modules.clustering import cluster_data_nmf, cluster_data_bert
import os
from datetime import datetime

app = Flask(__name__)


@app.route("/topic_modeling", methods=["POST"])
def topic_modeling():
    """
    Perform topic modeling on text data and save the results.

    Returns:
        JSON response with the path to the saved result file.
    """
    if request.method == "POST":
        # Check if "input_df_path" and "model_name" parameters are in the request
        if "input_df_path" not in request.args:
            return jsonify({"error": "'input_df_path' parameter is required!"}), 400
        if "model_name" not in request.args:
            return jsonify({"error": "'model_name' parameter is required!"}), 400

        input_df_path = request.args.get("input_df_path")
        model_name = request.args.get("model_name")
        # Check if the input file name exists
        if not os.path.exists(input_df_path):
            return jsonify({"error": "Input file not found"}), 404

        data = preprocess_text(input_df_path)

        # Load the model based on model_name (implement model loading)
        if model_name == "nmf":
            clustered_data = cluster_data_nmf(data)
        else:
            model_name = "bert"
            # BERT is used as the default model
            clustered_data = cluster_data_bert(data)

        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Modify the result filename to include the timestamp
        result_file = os.path.join(
            "./data/output/", f"output__{model_name}_{timestamp}.csv"
        )
        clustered_data.to_csv(result_file, index=False)

        return jsonify({"result_path": f"Results saved in: {result_file}"})
    else:
        return jsonify({"error": "Not a POST request!"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
