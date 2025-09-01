import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

fold_0 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/fold_0.csv')
fold_1 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/fold_1.csv')
fold_2 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/fold_2.csv')
fold_3 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/fold_3.csv')
fold_4 = pd.read_csv('/home/sersasj/RSNA-IAD-Codebase/fold_4.csv')

folds = [fold_0, fold_1, fold_2, fold_3, fold_4]

def calculate_auc_safe(y_true, y_prob):
    if y_true.nunique() < 2:
        return None  # Cannot calculate AUC if only one class
    return roc_auc_score(y_true, y_prob)

# Individual folds AUC
print("Individual Folds AUC-ROC:")
for i, fold in enumerate(folds):
    print(f"\nFold {i}:")
    # Aneurysm present
    auc_aneurysm = calculate_auc_safe(fold['label_aneurysm'], fold['aneurysm_prob'])
    print(f"  Aneurysm Present: {auc_aneurysm}")
    
    # Location classes
    for loc in range(13):
        auc_loc = calculate_auc_safe(fold[f'loc_label_{loc}'], fold[f'loc_prob_{loc}'])
        print(f"  Location {loc}: {auc_loc}")

# Overall OOF AUC
all_data = pd.concat(folds, ignore_index=True)
print("\n\nOverall OOF AUC-ROC:")
auc_aneurysm_oof = calculate_auc_safe(all_data['label_aneurysm'], all_data['aneurysm_prob'])
print(f"Aneurysm Present: {auc_aneurysm_oof}")

for loc in range(13):
    auc_loc_oof = calculate_auc_safe(all_data[f'loc_label_{loc}'], all_data[f'loc_prob_{loc}'])
    print(f"Location {loc}: {auc_loc_oof}")