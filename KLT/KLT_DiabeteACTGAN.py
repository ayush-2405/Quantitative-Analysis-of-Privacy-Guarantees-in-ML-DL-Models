import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Synthetic_generated_ACTGAN_Daibetes.csv")

# Function to generalize a numeric column using binning
def generalize_column(column, bin_size):
    return column.apply(lambda x: f"[{(x // bin_size) * bin_size}-{((x // bin_size) + 1) * bin_size - 1}]")

# Check k-anonymity
def is_k_anonymous(df, quasi_identifiers, k):
    group_sizes = df.groupby(quasi_identifiers).size()
    return (group_sizes >= k).all()

# Check l-diversity
def is_l_diverse(df, quasi_identifiers, sensitive_attr, l):
    diversity = df.groupby(quasi_identifiers)[sensitive_attr].nunique()
    return (diversity >= l).all()

# Check t-closeness using Total Variation Distance
def is_t_close(df, quasi_identifiers, sensitive_attr, t):
    global_dist = df[sensitive_attr].value_counts(normalize=True)
    for _, group in df.groupby(quasi_identifiers):
        group_dist = group[sensitive_attr].value_counts(normalize=True)
        group_dist = group_dist.reindex(global_dist.index, fill_value=0)
        tvd = 0.5 * np.abs(global_dist - group_dist).sum()
        if tvd > t:
            return False
    return True

# Find the most granular (smallest bin sizes) transformation that satisfies all constraints
def find_most_granular_k_l_t(df, k=5, l=2, t=0.2, max_bin_size=20):
    best_result = None
    best_bin_sum = float('inf')  # Track sum of bin sizes (smaller = more granular)

    for age_bin in range(1, max_bin_size):
        for preg_bin in range(1, max_bin_size):
            temp_df = df.copy()
            temp_df['Age'] = generalize_column(temp_df['Age'], age_bin)
            temp_df['Pregnancies'] = generalize_column(temp_df['Pregnancies'], preg_bin)

            if is_k_anonymous(temp_df, ['Pregnancies', 'Age'], k) and \
               is_l_diverse(temp_df, ['Pregnancies', 'Age'], 'Outcome', l) and \
               is_t_close(temp_df, ['Pregnancies', 'Age'], 'Outcome', t):
                
                current_bin_sum = age_bin + preg_bin
                if current_bin_sum < best_bin_sum:
                    best_result = (temp_df, age_bin, preg_bin)
                    best_bin_sum = current_bin_sum

    if best_result:
        return best_result
    else:
        return None, None, None

# Parameters
k = 5
l = 2
t = 0.2

# Apply the transformation
result_df, age_bin, preg_bin = find_most_granular_k_l_t(df, k=k, l=l, t=t)

# Output result
if result_df is not None:
    print(f"✅ Most granular binning that satisfies k={k}, l={l}, t={t}:")
    print(f"   Age bin size: {age_bin}, Pregnancies bin size: {preg_bin}")
    result_df.to_csv("most_granular_k5_l2_t02.csv", index=False)
else:
    print("❌ No configuration found that satisfies all constraints.")