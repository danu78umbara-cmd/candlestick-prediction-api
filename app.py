from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import preprocessor as pp
import joblib
import os
import io

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")


def feature_engineering(data_raw):
    df_processed = pp.process_dataframe(data_raw)
    return df_processed


def load_best_model(candle_pattern):
    folder = "saved_models"
    safe_pattern = candle_pattern.replace(" ", "_")

    for file in os.listdir(folder):
        if candle_pattern in file or safe_pattern in file:
            model_path = os.path.join(folder, file)
            model = joblib.load(model_path)
            return model

    raise FileNotFoundError(f"Model untuk pola '{candle_pattern}' tidak ditemukan.")


@app.route("/predict_csv", methods=["GET"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "File CSV belum diunggah."}), 400

        file = request.files["file"]
        if file.filename.strip() == "":
            return jsonify({"status": "error", "message": "Nama file tidak valid."}), 400

        df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")),
                         thousands=",", decimal=".")

        if len(df) < 20:
            return jsonify({"status": "error", "message": "Dibutuhkan minimal 20 data OHCLV."}), 400

        df_feat = feature_engineering(df)
        if df_feat.empty:
            return jsonify({"status": "error", "message": "Preprocessing menghasilkan data kosong."}), 400

        if "CandlePattern" not in df_feat.columns:
            return jsonify({"status": "error", "message": "Kolom 'CandlePattern' tidak ditemukan."}), 400

        last_pattern = df_feat.iloc[-1]["CandlePattern"]

        model = load_best_model(last_pattern)

        X_last = df_feat.tail(1).drop(columns=["CandlePattern"], errors="ignore")
        prediction = model.predict(X_last)[0]

        interpretation = (
            "Harga emas diperkirakan naik besok"
            if prediction == 1 else
            "Harga emas diperkirakan turun besok"
        )

        return jsonify({
            "status": "success",
            "CandlePattern": str(last_pattern),
            "prediction": int(prediction),
            "interpretation": interpretation
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
