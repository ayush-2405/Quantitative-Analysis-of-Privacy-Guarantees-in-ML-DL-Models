import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("tabular-actgan-employee.csv")

# Generalization function for binning
def generalize_column(column, bin_size):
    return column.apply(lambda x: f"[{(x // bin_size) * bin_size}-{((x // bin_size) + 1) * bin_size - 1}]") 

# k-anonymity check
def is_k_anonymous(df, quasi_identifiers, k):
    return (df.groupby(quasi_identifiers).size() >= k).all()

# ℓ-diversity check
def is_l_diverse(df, quasi_identifiers, sensitive_attr, l):
    return (df.groupby(quasi_identifiers)[sensitive_attr].nunique() >= l).all()

# t-closeness check using Total Variation Distance (TVD)
def is_t_close(df, quasi_identifiers, sensitive_attr, t):
    global_dist = df[sensitive_attr].value_counts(normalize=True)

    for _, group in df.groupby(quasi_identifiers):
        group_dist = group[sensitive_attr].value_counts(normalize=True)
        group_dist = group_dist.reindex(global_dist.index, fill_value=0)
        tvd = 0.5 * np.abs(global_dist - group_dist).sum()
        if tvd > t:
            return False
    return True

# Combined function for all 3 privacy criteria
def apply_k_l_t_privacy(df, k=5, l=2, t=0.2, max_age_bin=10):
    for age_bin in range(1, max_age_bin + 1):
        for exp_bin in range(1, max_age_bin + 1):
            temp_df = df.copy()
            temp_df['Age'] = generalize_column(temp_df['Age'], age_bin)
            temp_df['ExperienceInCurrentDomain'] = generalize_column(temp_df['ExperienceInCurrentDomain'], exp_bin)
            
            if is_k_anonymous(temp_df, ['Age', 'ExperienceInCurrentDomain'], k) and \
               is_l_diverse(temp_df, ['Age', 'ExperienceInCurrentDomain'], 'LeaveOrNot', l) and \
               is_t_close(temp_df, ['Age', 'ExperienceInCurrentDomain'], 'LeaveOrNot', t):
                return temp_df, age_bin, exp_bin
    return None, None, None

# Apply the transformation
k = 5
l = 2
t = 0.2
result_df, age_bin, exp_bin = apply_k_l_t_privacy(df, k=k, l=l, t=t)

# Output result
if result_df is not None:
    print(f"✅ Satisfied k={k} anonymity, ℓ={l} diversity, and t={t} closeness")
    print(f"   Age bin size: {age_bin}, Experience bin size: {exp_bin}")
    result_df.to_csv("employee_k5_l2_t02.csv", index=False)
else:
    print("❌ Could not satisfy all three privacy constraints under tested bin sizes.")