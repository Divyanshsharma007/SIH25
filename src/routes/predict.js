const express = require("express");
const router = express.Router();
const PredictionModel = require("../models/predictionModel");

// Initialize model connector
const predictionModel = new PredictionModel();

// Prediction endpoint
router.post("/", async (req, res) => {
  try {
    const { features } = req.body;

    if (!features || !Array.isArray(features)) {
      return res.status(400).json({
        error: "Features array is required in the request body",
      });
    }

    // Forward prediction request to Python service
    const prediction = await predictionModel.predict(features);

    res.json({
      success: true,
      ...prediction,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({
      error: "Prediction failed",
      details: error.message,
    });
  }
});

// Model info endpoint
router.get("/model-info", (req, res) => {
  res.json({
    model_type: "CatBoost",
    input_features: 35, // adjust based on your dataset
    target_classes: ["Dropout", "Graduate", "Enrolled"],
  });
});

module.exports = router;
