import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


# ==========================================
# HOUSEHOLD EXPENSE PATTERN ANALYZER
# ==========================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_DIR, "household_expenses.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)


# ==========================================
# 1. GENERATE SAMPLE DATASET
# ==========================================

def generate_sample_data():
    """Create a reproducible sample dataset of 2,937 transactions."""

    np.random.seed(42)

    households = {
        "H001": 35000,
        "H002": 45000,
        "H003": 55000,
        "H004": 65000,
        "H005": 75000,
    }

    categories = [
        "Food",
        "Transport",
        "Housing",
        "Utilities",
        "Healthcare",
        "Education",
        "Shopping",
        "Entertainment",
    ]

    category_weights = [0.25, 0.12, 0.20, 0.10, 0.08, 0.08, 0.10, 0.07]
    payment_modes = ["Cash", "UPI", "Debit Card", "Credit Card"]

    rows = []

    for i in range(2937):
        household = np.random.choice(list(households.keys()))
        category = np.random.choice(categories, p=category_weights)

        # Generate dates throughout 2025.
        date = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            np.random.randint(0, 365), unit="D"
        )

        # Generate realistic sample amounts for different categories.
        amount_ranges = {
            "Food": (100, 1800),
            "Transport": (50, 1000),
            "Housing": (3000, 15000),
            "Utilities": (300, 3500),
            "Healthcare": (200, 5000),
            "Education": (300, 6000),
            "Shopping": (150, 5000),
            "Entertainment": (100, 2500),
        }

        low, high = amount_ranges[category]
        amount = round(np.random.uniform(low, high), 2)

        payment_mode = np.random.choice(payment_modes)

        rows.append({
            "Transaction_ID": f"T{i + 1:04d}",
            "Household_ID": household,
            "Date": date.strftime("%Y-%m-%d"),
            "Category": category,
            "Amount": amount,
            "Payment_Mode": payment_mode,
            "Monthly_Income": households[household],
        })

    df = pd.DataFrame(rows)

    # Add a few missing values to demonstrate data cleaning.
    missing_amount_rows = np.random.choice(df.index, 15, replace=False)
    df.loc[missing_amount_rows, "Amount"] = np.nan

    missing_payment_rows = np.random.choice(df.index, 25, replace=False)
    df.loc[missing_payment_rows, "Payment_Mode"] = np.nan

    # Create duplicate records for demonstration.
    duplicate_rows = df.sample(10, random_state=7)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    df.to_csv(DATA_FILE, index=False)
    print("Sample dataset created:", DATA_FILE)


# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

def load_and_clean_data():
    """Load the dataset and clean invalid or duplicate records."""

    if not os.path.exists(DATA_FILE):
        generate_sample_data()

    df = pd.read_csv(DATA_FILE)

    original_rows = len(df)

    required_columns = [
        "Transaction_ID",
        "Household_ID",
        "Date",
        "Category",
        "Amount",
        "Payment_Mode",
        "Monthly_Income",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Monthly_Income"] = pd.to_numeric(
        df["Monthly_Income"], errors="coerce"
    )

    # Remove duplicate transactions.
    df = df.drop_duplicates(
        subset=["Transaction_ID"], keep="first"
    )

    # Remove rows without essential transaction information.
    df = df.dropna(
        subset=[
            "Household_ID",
            "Date",
            "Category",
            "Amount",
            "Monthly_Income",
        ]
    )

    # Expenses and income must be positive.
    df = df[
        (df["Amount"] > 0)
        & (df["Monthly_Income"] > 0)
    ].copy()

    # Replace missing payment modes.
    df["Payment_Mode"] = (
        df["Payment_Mode"].fillna("Unknown")
    )

    df["Category"] = df["Category"].astype(str).str.strip()
    df["Payment_Mode"] = (
        df["Payment_Mode"].astype(str).str.strip()
    )

    # Feature engineering.
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")
    df["Day_Name"] = df["Date"].dt.day_name()
    df["Is_Weekend"] = df["Date"].dt.dayofweek >= 5

    df = df.sort_values("Date").reset_index(drop=True)

    print("\n========== DATA CLEANING ==========")
    print("Original rows:", original_rows)
    print("Cleaned rows:", len(df))
    print("Rows removed:", original_rows - len(df))
    print("Missing values after cleaning:")
    print(df[required_columns].isnull().sum())

    df.to_csv(
        os.path.join(OUTPUT_DIR, "cleaned_expenses.csv"),
        index=False,
    )

    return df


# ==========================================
# 3. CATEGORY-WISE SPENDING
# ==========================================

def analyze_categories(df):
    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    category_totals.to_csv(
        os.path.join(OUTPUT_DIR, "category_spending.csv")
    )

    plt.figure(figsize=(10, 6))
    category_totals.sort_values().plot(
        kind="barh", title="Category-wise Household Spending"
    )
    plt.xlabel("Total Spending")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "category_spending.png")
    )
    plt.close()

    print("\n========== CATEGORY ANALYSIS ==========")
    print(category_totals.round(2))

    return category_totals


# ==========================================
# 4. MONTHLY SPENDING AND FORECAST
# ==========================================

def analyze_monthly_spending(df):
    monthly = (
        df.groupby(df["Date"].dt.to_period("M"))["Amount"]
        .sum()
        .sort_index()
    )

    monthly_df = monthly.rename(
        "Total_Expense"
    ).reset_index()

    monthly_df["Month"] = monthly_df["Date"].astype(str)

    monthly_df.to_csv(
        os.path.join(OUTPUT_DIR, "monthly_spending.csv"),
        index=False,
    )

    # Forecast the next three months.
    monthly_values = monthly.to_numpy(dtype=float)
    month_numbers = np.arange(len(monthly_values)).reshape(-1, 1)

    future_numbers = np.arange(
        len(monthly_values),
        len(monthly_values) + 3,
    ).reshape(-1, 1)

    model = LinearRegression()
    model.fit(month_numbers, monthly_values)

    predictions = model.predict(future_numbers)
    predictions = np.maximum(predictions, 0)

    last_month = monthly.index.max()

    future_dates = pd.period_range(
        start=last_month + 1, periods=3, freq="M"
    )

    forecast_df = pd.DataFrame({
        "Month": future_dates.astype(str),
        "Predicted_Expense": predictions.round(2),
    })

    forecast_df.to_csv(
        os.path.join(OUTPUT_DIR, "expense_forecast.csv"),
        index=False,
    )

    # Plot historical spending and forecast.
    plt.figure(figsize=(11, 6))

    plt.plot(
        monthly.index.astype(str),
        monthly_values,
        marker="o",
        label="Actual Spending",
    )

    plt.plot(
        forecast_df["Month"],
        forecast_df["Predicted_Expense"],
        marker="o",
        linestyle="--",
        label="Forecast",
    )

    plt.title("Monthly Expenses and Three-Month Forecast")
    plt.xlabel("Month")
    plt.ylabel("Total Expenses")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "monthly_trend_forecast.png")
    )
    plt.close()

    print("\n========== MONTHLY SPENDING ==========")
    print(monthly.round(2))

    print("\n========== NEXT THREE MONTHS FORECAST ==========")
    print(forecast_df.to_string(index=False))

    return monthly, forecast_df


# ==========================================
# 5. HOUSEHOLD INCOME AND EXPENSE ANALYSIS
# ==========================================

def analyze_households(df):
    household_expenses = (
        df.groupby("Household_ID")["Amount"]
        .sum()
        .rename("Annual_Expenses")
    )

    household_income = (
        df.groupby("Household_ID")["Monthly_Income"]
        .first()
        .rename("Monthly_Income")
    )

    household_summary = pd.concat(
        [household_income, household_expenses], axis=1
    )

    household_summary["Annual_Income"] = (
        household_summary["Monthly_Income"] * 12
    )

    household_summary["Estimated_Annual_Savings"] = (
        household_summary["Annual_Income"]
        - household_summary["Annual_Expenses"]
    )

    household_summary["Estimated_Savings_Rate_Percent"] = (
        household_summary["Estimated_Annual_Savings"]
        / household_summary["Annual_Income"]
        * 100
    ).round(2)

    household_summary = household_summary.reset_index()

    household_summary.to_csv(
        os.path.join(OUTPUT_DIR, "household_summary.csv"),
        index=False,
    )

    household_summary.set_index("Household_ID")[
        ["Annual_Income", "Annual_Expenses"]
    ].plot(kind="bar", figsize=(10, 6))

    plt.title("Annual Income vs Annual Expenses")
    plt.xlabel("Household")
    plt.ylabel("Amount")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "household_income_expenses.png")
    )
    plt.close()

    print("\n========== HOUSEHOLD SUMMARY ==========")
    print(household_summary.round(2).to_string(index=False))

    return household_summary


# ==========================================
# 6. PAYMENT MODE ANALYSIS
# ==========================================

def analyze_payment_modes(df):
    payment_totals = (
        df.groupby("Payment_Mode")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    payment_totals.to_csv(
        os.path.join(OUTPUT_DIR, "payment_mode_spending.csv")
    )

    plt.figure(figsize=(8, 6))
    payment_totals.plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title("Spending by Payment Mode")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "payment_mode_spending.png")
    )
    plt.close()

    print("\n========== PAYMENT MODE ANALYSIS ==========")
    print(payment_totals.round(2))


# ==========================================
# 7. WEEKDAY AND WEEKEND ANALYSIS
# ==========================================

def analyze_weekday_patterns(df):
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    daily_totals = (
        df.groupby("Day_Name")["Amount"]
        .sum()
        .reindex(weekday_order, fill_value=0)
    )

    daily_totals.to_csv(
        os.path.join(OUTPUT_DIR, "weekday_spending.csv")
    )

    weekend_summary = (
        df.groupby("Is_Weekend")["Amount"]
        .agg(["count", "sum", "mean"])
        .rename(index={
            False: "Weekday",
            True: "Weekend",
        })
    )

    weekend_summary.to_csv(
        os.path.join(OUTPUT_DIR, "weekend_summary.csv")
    )

    plt.figure(figsize=(10, 6))
    daily_totals.plot(kind="bar")
    plt.title("Spending by Day of the Week")
    plt.xlabel("Day")
    plt.ylabel("Total Spending")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        os.path.join(CHART_DIR, "weekday_spending.png")
    )
    plt.close()

    print("\n========== WEEKDAY SPENDING ==========")
    print(daily_totals.round(2))

    print("\n========== WEEKDAY VS WEEKEND ==========")
    print(weekend_summary.round(2))


# ==========================================
# 8. ANOMALY DETECTION USING Z-SCORE
# ==========================================

def detect_anomalies(df):
    """Flag transactions with an absolute Z-score above 3."""

    mean_amount = df["Amount"].mean()
    std_amount = df["Amount"].std(ddof=0)

    result = df.copy()

    if std_amount == 0 or np.isnan(std_amount):
        result["Z_Score"] = 0.0
        result["Is_Anomaly"] = False
    else:
        result["Z_Score"] = (
            result["Amount"] - mean_amount
        ) / std_amount

        result["Is_Anomaly"] = (
            result["Z_Score"].abs() > 3
        )

    anomalies = result[result["Is_Anomaly"]].copy()

    anomalies.to_csv(
        os.path.join(OUTPUT_DIR, "anomalous_transactions.csv"),
        index=False,
    )

    print("\n========== ANOMALY DETECTION ==========")
    print("Transactions analyzed:", len(result))
    print("Potential anomalies found:", len(anomalies))

    if not anomalies.empty:
        print(
            anomalies[
                ["Transaction_ID", "Household_ID", "Category",
                 "Amount", "Z_Score"]
            ].head(10).round(2).to_string(index=False)
        )
    else:
        print("No transactions exceeded the Z-score threshold.")

    print(
        "\nNote: An anomaly is unusual spending, "
        "not necessarily an error or fraud."
    )

    return anomalies


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

def main():
    print("=" * 55)
    print(" HOUSEHOLD EXPENSE PATTERN ANALYZER")
    print("=" * 55)

    df = load_and_clean_data()

    analyze_categories(df)
    analyze_monthly_spending(df)
    analyze_households(df)
    analyze_payment_modes(df)
    analyze_weekday_patterns(df)
    detect_anomalies(df)

    print("\n" + "=" * 55)
    print(" PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 55)
    print("Dataset:", DATA_FILE)
    print("Analysis results:", OUTPUT_DIR)
    print("Charts:", CHART_DIR)


if __name__ == "__main__":
    main()