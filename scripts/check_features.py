import joblib
m = joblib.load('properly_trained_model.joblib')
print('Total features:', len(m.feature_names))
print('\nAll features:')
for i, f in enumerate(m.feature_names):
    print(f"{i+1}. {f}")
