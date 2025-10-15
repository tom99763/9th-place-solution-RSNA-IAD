import pandas as pd


# Load the three metric datasets
map_50 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/wandb_map_50.csv')
map_50_95 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/wandb_map_50_95.csv')
mauc = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/wand_auc.csv')
#mauc = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/wandb_export_2025-09-29T14_43_48.536-03_00.csv')
# Display structure of each dataset
print("MAP50 Dataset:")
print(map_50.head())
print("\nMAP50-95 Dataset:")
print(map_50_95.head())
print("\nMAUC Dataset:")
print(mauc.head())

# Display column names to understand the metric columns
print("\nMAP50 columns:", map_50.columns.tolist())
print("MAP50-95 columns:", map_50_95.columns.tolist())
print("MAUC columns:", mauc.columns.tolist())

# Extract the metric values (using MAX values for fitness calculation)
map50_values = map_50['yolo_11_m_one_loss_fold0 - metrics/mAP50(B)__MAX']
map50_95_values = map_50_95['yolo_11_m_one_loss_fold0 - metrics/mAP50-95(B)__MAX']
mauc_values = mauc['yolo_11_m_one_loss_fold0 - metrics/mauc(B)__MAX']
#mauc_values = mauc['cv_y11_yolo11m_new_fit_fold02 - metrics/mauc(B)__MAX']

# Calculate fitness: 0.1 * map50 + 0.8 * map_50_95 + 0.1 * mauc
# Ensure all datasets have the same length
min_length = min(len(map50_values), len(map50_95_values), len(mauc_values))
print(f"\nDataset lengths - MAP50: {len(map50_values)}, MAP50-95: {len(map50_95_values)}, MAUC: {len(mauc_values)}")
print(f"Using {min_length} steps for fitness calculation")

# Calculate fitness for each step
fitness_scores = []
for i in range(min_length):
    fitness = 0.5 * map50_values.iloc[i] + 0.25 * map50_95_values.iloc[i] + 0.25 * mauc_values.iloc[i]
    fitness_scores.append(fitness)  

# Create a results dataframe
results_df = pd.DataFrame({
    'Step': range(1, min_length + 1),
    'MAP50': map50_values.iloc[:min_length],
    'MAP50-95': map50_95_values.iloc[:min_length],
    'MAUC': mauc_values.iloc[:min_length],
    'Fitness': fitness_scores
})

print("\n" + "="*80)
print("FITNESS CALCULATION RESULTS")
print("="*80)
print("Formula: Fitness = 0.15 * MAP50 + 0.7 * MAP50-95 + 0.15 * MAUC")
print("\nFirst 10 steps:")
print(results_df.head(10).to_string(index=False, float_format='%.6f'))

print("\nLast 10 steps:")
print(results_df.tail(10).to_string(index=False, float_format='%.6f'))

print(f"\nSUMMARY STATISTICS:")
print(f"Best Fitness Score: {max(fitness_scores):.6f} (Step {fitness_scores.index(max(fitness_scores)) + 1})")
print(f"Final Fitness Score: {fitness_scores[-1]:.6f} (Step {min_length})")
print(f"Average Fitness Score: {sum(fitness_scores)/len(fitness_scores):.6f}")

# Print top 3 fitness scores
top_3_fitness = results_df.nlargest(10, 'Fitness')
print(f"\nTOP 3 FITNESS SCORES:")
print("="*50)
print(top_3_fitness.to_string(index=False, float_format='%.6f'))
# Step    MAP50  MAP50-95     MAUC  Fitness
#   68 0.461130  0.283530 0.923750 0.674680
#   84 0.458970  0.287790 0.921990 0.673362
#   77 0.465770  0.278830 0.917950 0.673166