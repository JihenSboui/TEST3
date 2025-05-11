import joblib
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score, make_scorer, f1_score
from sklearn.inspection import permutation_importance

def multioutput_f1(y_true, y_pred):
    scores = []
    for col in range(y_true.shape[1]):
        scores.append(
            f1_score(
                y_true.iloc[:, col],
                y_pred[:, col],
                average='macro',
                zero_division=0
            )
        )
    return np.mean(scores)

custom_scorer = make_scorer(multioutput_f1)
data = "BaseExcel"
fil = [os.path.join(data, f) for f in os.listdir(data) if f.endswith('.xlsx')]
df = pd.concat([pd.read_excel(file) for file in fil], ignore_index=True)
df.columns = df.columns.str.strip()
df = df.drop(columns=['Session_ID', 'Timestamp', 'Frame'], errors='ignore')

angle_cols = [
    "Right_Shoulder_Angle",      
    "Left_Shoulder_Angle",       
    "Right_Elbow_Angle",        
    "Left_Elbow_Angle",          
    "Right_Wrist_Rotation",     
    "Left_Wrist_Rotation",       
    "Torso_Angle",              
    "Neck_Angle"                
]

risk_cols = [
    'Right_Shoulder_RULA_Risk',
    'Left_Shoulder_RULA_Risk',
    'Right_Elbow_RULA_Risk',
    'Left_Elbow_RULA_Risk',
    'Right_Wrist_RULA_Risk',
    'Left_Wrist_RULA_Risk',
    'Torso_RULA_Risk',
    'Neck _RULA_Risk'
]
map_risque = {
    "Risque Faible": 1,
    "Risque Modéré": 3,
    "Risque Élevé": 5,
}

for col in risk_cols:
    df[col + "_Encoded"] = df[col].map(map_risque)

encoded_risk_cols = [col + "_Encoded" for col in risk_cols]
X = df[angle_cols]
y = df[encoded_risk_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
base_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = MultiOutputClassifier(base_model)
param_grid = {
    'estimator__max_features': ['sqrt', 'log2'],
    'estimator__min_samples_split': [2, 5]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring=custom_scorer,
    n_jobs=-1,
    error_score='raise'
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

labels = [1, 3, 5]
target_names = ["Risque Faible", "Risque Modéré", "Risque Élevé"]

y_pred = best_model.predict(X_test)

for idx, col in enumerate(encoded_risk_cols):
    print(f"\n Rapport pour {risk_cols[idx]} :")
    print(classification_report(
        y_test.iloc[:, idx],
        y_pred[:, idx],
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test.iloc[:, idx], y_pred[:, idx]):.2f}")
    print("-" * 60)
    # Sauvegarde du modèle

joblib.dump(best_model, 'modele_risque_rula_compat.pkl')
print("\n✅ Modèle enregistré dans 'modele_risque_rula_compat.pkl'")