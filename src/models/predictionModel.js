const axios = require("axios");

class PredictionModel {
  constructor() {
    this.pythonServiceUrl =
      process.env.PYTHON_SERVICE_URL || "http://localhost:5000";
    this.timeout = 10000; // 10 second timeout
  }

  async predict(features) {
    try {
      const response = await axios.post(
        `${this.pythonServiceUrl}/predict`,
        {
          features: features,
        },
        {
          timeout: this.timeout,
        }
      );

      return response.data;
    } catch (error) {
      console.error("Error calling Python service:", error.message);

      if (error.code === "ECONNREFUSED") {
        throw new Error("Prediction service is not running");
      } else if (error.response) {
        // Python service returned an error
        throw new Error(
          `Prediction service error: ${error.response.data.error}`
        );
      } else if (error.request) {
        throw new Error("No response from prediction service");
      } else {
        throw new Error("Error configuring prediction request");
      }
    }
  }

  async healthCheck() {
    try {
      const response = await axios.get(`${this.pythonServiceUrl}/health`, {
        timeout: 5000,
      });
      return response.data;
    } catch (error) {
      return { status: "ERROR", error: error.message };
    }
  }
}

module.exports = PredictionModel;
