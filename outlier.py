import matplotlib.pyplot as plt
import seaborn as sns

num = df.select_dtypes('number')

for col in num:

    # ===== Outlier Calculations =====
    q1 = num[col].quantile(0.25)
    q3 = num[col].quantile(0.75)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    out = num[(num[col] < lb) | (num[col] > ub)][col]

    print(f"\nFeature: {col}")
    print(f"  Q1 = {q1}, Q3 = {q3}, IQR = {iqr}")
    print(f"  Lower Bound = {lb}, Upper Bound = {ub}")
    print(f"  Outliers:\n{out.values if not out.empty else 'No significant outliers'}")

    # ===== Skewness Check =====
    skew_value = num[col].skew()

    if skew_value > 0.5:
        dist = "Right Skewed"
    elif skew_value < -0.5:
        dist = "Left Skewed"
    else:
        dist = "Approximately Normal"

    print(f"  Skew = {skew_value:.3f} → {dist}")

    # ===== Plots in one figure =====
    plt.figure(figsize=(15,4))

    # Histogram
    plt.subplot(1, 3, 1)
    plt.hist(num[col], bins=10)
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

    # Boxplot
    plt.subplot(1, 3, 2)
    plt.boxplot(num[col])
    plt.title(f'Boxplot: {col}')

    # KDE Curve
    plt.subplot(1, 3, 3)
    sns.kdeplot(num[col], fill=True)
    plt.title(f'KDE Curve: {col}\n({dist})')
    plt.xlabel(col)
    plt.ylabel('Density')

    plt.tight_layout()
    plt.show()
