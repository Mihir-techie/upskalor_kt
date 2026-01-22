import pandas as pd
import numpy as np
import json
import os
from collections import deque  

EMOTION_MAP = {
    "Distressed": 1,
    "Exhausted": 1,
    "Anxious": 1,
    "Stressed": 2,
    "Tired": 2,
    "Bored": 2,
    "Calm": 3,
    "Content": 3,
    "Relaxed": 4,
    "Serene": 4,
    "Happy": 5,
    "Energized": 5,
    "Accomplished": 5,
    "Invigorated": 5,
    "Strong": 5,
    "Empowered": 5,
    "Fulfilled": 5,
    "Flexible": 4,
    "Challenged": 3,
    "Refreshed": 4,
    "Understood": 4,
    "Rejuvenated": 5,
    "Recharged": 4,
    "Agile": 4,
}
ORIGINAL_RANGES = {
    "energy_level": (1, 10),
    "mind_clarity": (1, 10),
    "stress_level": (1, 10),
    "emotion_score": (1, 5),
}
EWMA_LAMBDA = 0.1
MAX_RAW_DRIFT = 10.0 
WEEKLY_HISTORY_FILE = "weekly_history.json"
WEEKLY_WINDOW = 7





def _normalize_input(input_data):
    """Maps emotion and normalizes the four metrics based on their known ranges."""

    emotion_val = EMOTION_MAP.get(input_data["emotion_state"], 3)
    input_data["emotion_score"] = emotion_val

    normalized_input = {}

    
    for metric in ["energy_level", "mind_clarity", "stress_level"]:
        min_val, max_val = ORIGINAL_RANGES[metric]
        normalized_input[metric] = (input_data[metric] - min_val) / (max_val - min_val)

 
    min_val, max_val = ORIGINAL_RANGES["emotion_score"]
    normalized_input["emotion_score"] = (emotion_val - min_val) / (max_val - min_val)

    return normalized_input


def _calculate_z_scores(normalized_input, baseline_stats):
    """Calculates Z-score for each metric."""
    z_scores = {}

    for metric, X in normalized_input.items():
        mu = baseline_stats.loc[metric, "Baseline_Mean"]
        variance = baseline_stats.loc[metric, "Baseline_Variance"]
        sigma = np.sqrt(variance)

        if sigma == 0:
            Z = 0.0
        else:
            Z = (X - mu) / sigma

        z_scores[metric] = Z

    return z_scores


def _calculate_combined_drift(z_scores):
    """Combines individual Z-scores into a single 0-100 score."""

    raw_drift_index = sum(np.abs(list(z_scores.values())))

    drift_score = min(100, (raw_drift_index / MAX_RAW_DRIFT) * 100)

    if drift_score >= 80:
        classification = "severe_drift"
        anomaly_flag = True
    elif drift_score >= 40:
        classification = "moderate_drift"
        anomaly_flag = True
    else:
        classification = "minimal_drift"
        anomaly_flag = False

    return drift_score, classification, anomaly_flag


def _update_baseline_ewma(normalized_input, baseline_stats):
    """Updates the baseline Mean and Variance using EWMA."""

    updated_stats = baseline_stats.copy()

    for metric, X in normalized_input.items():
        mu_old = updated_stats.loc[metric, "Baseline_Mean"]
        var_old = updated_stats.loc[metric, "Baseline_Variance"]

        # Mean EWMA formula
        mu_new = (EWMA_LAMBDA * X) + ((1 - EWMA_LAMBDA) * mu_old)

        #  Variance
        variance_new = (EWMA_LAMBDA * (X - mu_new) ** 2) + ((1 - EWMA_LAMBDA) * var_old)

        updated_stats.loc[metric, "Baseline_Mean"] = mu_new
        updated_stats.loc[metric, "Baseline_Variance"] = variance_new

    updated_stats.to_csv("user_baseline_stats.csv")

    return updated_stats





def generate_explanation(z_scores):
    """
    Generates a rules-based text explanation based on the Z-scores.
    """
    driver_metric = max(z_scores, key=lambda key: abs(z_scores[key]))
    driver_z_score = z_scores[driver_metric]
    direction = "above" if driver_z_score > 0 else "below"

    metric_names = {
        "energy_level": "Energy",
        "mind_clarity": "Clarity",
        "stress_level": "Stress",
        "emotion_score": "Mood",
    }

    friendly_name = metric_names.get(driver_metric, driver_metric)

    if abs(driver_z_score) < 0.5:
        return "Your mood metrics are currently in line with your personal baseline."

 
    if friendly_name == "Stress":
        if direction == "above":
            return f" High Stress Alert: Your {friendly_name} level is significantly {direction} your normal baseline."
        else:
            return f" Positive Shift: Your {friendly_name} is lower than your average, suggesting a period of calm."

 
    else:
        if direction == "above":
            return f" Positive Drift: Your {friendly_name} score is notably {direction} your personal baseline, indicating a strong positive state."
        else:
            return f" Key Drop: Your {friendly_name} is significantly {direction} your normal baseline, signaling a potential low point."





def _get_weekly_history():
    """Retrieves and manages the last 7 days of history."""
    if os.path.exists(WEEKLY_HISTORY_FILE):
        with open(WEEKLY_HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
    return deque(history, maxlen=WEEKLY_WINDOW)


def _save_weekly_history(history):
    """Saves the history back to the file."""
    with open(WEEKLY_HISTORY_FILE, "w") as f:
        json.dump(list(history), f, indent=4)


def generate_weekly_summary(new_day_data, baseline_stats):
    """
    Generates the weekly summary based on the last 7 days of data.
    """

    history = _get_weekly_history()


    history.append(
        {
            "date": new_day_data["timestamp"],
            "drift_score": new_day_data["drift_score"],
            "normalized_input": new_day_data["normalized_input"],
            "z_scores": new_day_data["z_scores"],
        }
    )

    _save_weekly_history(history)

    
    if len(history) < WEEKLY_WINDOW:
        return {
            "avg_drift": 0,
            "trend": "data_gathering",
            "biggest_improvement": "N/A",
            "biggest_drop": "N/A",
        }

    #  Calculate Average Drift 
    avg_drift = round(np.mean([d["drift_score"] for d in history]))

    #  Calculate Trend 
    first_half_drift = np.mean([d["drift_score"] for d in list(history)[:3]])
    second_half_drift = np.mean([d["drift_score"] for d in list(history)[-3:]])

    if second_half_drift > first_half_drift + 5:
        trend = "upward more drift"
    elif second_half_drift < first_half_drift - 5:
        trend = "downward less drift"
    else:
        trend = "stable"

    

    metric_averages = {}
    for metric in baseline_stats.index:
        metric_averages[metric] = np.mean(
            [
                d["normalized_input"][metric]
                for d in history
                if metric in d["normalized_input"]
            ]
        )

  
    deviations = {}
    for metric, avg_value in metric_averages.items():
        ewma_mean = baseline_stats.loc[metric, "Baseline_Mean"]
        deviations[metric] = avg_value - ewma_mean

   
    biggest_improvement = max(deviations, key=deviations.get)
    biggest_drop = min(deviations, key=deviations.get)

    friendly_names = {
        "energy_level": "Energy",
        "mind_clarity": "Clarity",
        "stress_level": "Stress",
        "emotion_score": "Mood",
    }

    return {
        "avg_drift": avg_drift,
        "trend": trend,
        "biggest_improvement": friendly_names[biggest_improvement],
        "biggest_drop": friendly_names[biggest_drop],
    }





def detect_mood_drift(daily_input):
    """
    Main function to detect drift, calculate scores, and update baseline,
    and generate the explanation and weekly summary.
    """

    try:
        baseline_stats = pd.read_csv("user_baseline_stats.csv", index_col=0)
    except FileNotFoundError:
        return {"error": "Baseline stats file not found. Run baseline_engine.py first."}

    
    normalized_input = _normalize_input(daily_input)

    
    z_scores = _calculate_z_scores(normalized_input, baseline_stats)

   
    drift_score, classification, anomaly_flag = _calculate_combined_drift(z_scores)

    
    explanation_text = generate_explanation(z_scores)

    
    updated_stats = _update_baseline_ewma(normalized_input, baseline_stats)

    # Prepare preliminary output for Step 5
    daily_output = {
        "timestamp": daily_input["timestamp"],
        "drift_score": round(drift_score),
        "classification": classification,
        "anomaly_flag": anomaly_flag,
        "z_scores": z_scores,
        "normalized_input": normalized_input,
    }

    #  Generate Weekly Summary 
    weekly_summary = generate_weekly_summary(daily_output, updated_stats)

    # Prepare the final output dictionary
    final_output = {
        "drift_score": daily_output["drift_score"],
        "classification": daily_output["classification"],
        "anomaly_flag": daily_output["anomaly_flag"],
        "explanation": explanation_text,
        "weekly_summary": weekly_summary,
    }

    return final_output


if __name__ == "__main__":
  

    # DAY 1: Bad Day High Drift
    example_bad_input = {
        "energy_level": 2,
        "mind_clarity": 3,
        "emotion_state": "Exhausted",
        "stress_level": 9,
        "timestamp": "2025-12-01",
    }

    print("DAY 1 Simulating First Bad Day")
    result_d1 = detect_mood_drift(example_bad_input)
    print(json.dumps(result_d1, indent=4))

    # DAY 2-6 Stable Days
    for i in range(2, 7):
        example_stable_input = {
            "energy_level": 5,
            "mind_clarity": 6,
            "emotion_state": "Content",
            "stress_level": 5,
            "timestamp": f"2025-12-0{i}",
        }
        detect_mood_drift(example_stable_input)
        print(f"\n DAY {i} Stable ")

    # DAY 7: Good Day Generates Summary
    example_summary_input = {
        "energy_level": 9,
        "mind_clarity": 8,
        "emotion_state": "Accomplished",
        "stress_level": 2,
        "timestamp": "2025-12-07",
    }

    print("\n DAY 7 Final Day - Generates Weekly Summary) ")
    result_d7 = detect_mood_drift(example_summary_input)
    print(json.dumps(result_d7, indent=4))