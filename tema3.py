import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# ==========================================
# 1. ÎNCĂRCARE ȘI CURĂȚARE DATE
# ==========================================
print("--- Pasul 1: Încărcare date ---")
try:
    # 'latin1' rezolvă eroarea de UnicodeDecodeError
    df_cons = pd.read_csv('consumption_user.csv', encoding='latin1')
    df_subj = pd.read_csv('subject_user.csv', encoding='latin1')
    print("Succes: Datele au fost încărcate.")
except Exception as e:
    print(f"Eroare la încărcare: {e}")
    exit()

# Curățare: eliminăm spațiile albe din numele alimentelor
df_cons['INGREDIENT_ENG'] = df_cons['INGREDIENT_ENG'].str.strip()

# ==========================================
# 2. ANALIZĂ DESCRIPTIVĂ (Vizualizări pentru Raport)
# ==========================================
print("\n--- Pasul 2: Analiză descriptivă ---")

# A. Top 15 Alimente Consumate
top_foods = df_cons['INGREDIENT_ENG'].value_counts().head(15)
plt.figure(figsize=(12, 7))
top_foods.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('Top 15 Ingrediente Consumate (Frecvență)', fontsize=14)
plt.xlabel('Număr de apariții')
plt.ylabel('Ingredient')
plt.tight_layout()
plt.savefig('fig1_top_alimente.png')
print("Salvat: fig1_top_alimente.png")

# B. Distribuția pe Sexe
gender_counts = df_subj['SEX'].replace({1: 'Masculin', 2: 'Feminin'}).value_counts()
plt.figure(figsize=(7, 7))
gender_counts.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=90)
plt.title('Distribuția Subiecților pe Sexe', fontsize=14)
plt.ylabel('')
plt.savefig('fig2_distributie_sexe.png')
print("Salvat: fig2_distributie_sexe.png")

# ==========================================
# 3. PREGĂTIRE TRANZACȚII (Modelare pentru Algoritm)
# ==========================================
print("\n--- Pasul 3: Pregătire tranzacții ---")
# Grupăm alimentele după Subiect, Zi și Masă pentru a defini o "tranzacție"
transactions = df_cons.groupby(['SUBJECT', 'SURVEY_DAY', 'MEAL_NAME'])['INGREDIENT_ENG'].apply(list).tolist()
print(f"Total tranzacții generate: {len(transactions)}")

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# ==========================================
# 4. ALGORITMUL FP-GROWTH
# ==========================================
print("\n--- Pasul 4: Execuție FP-Growth ---")
# Support 0.02 = alimente care apar în cel puțin 2% din mese
frequent_itemsets = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)

# Generăm regulile (Lift > 1.2 pentru a elimina asocierile triviale)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules = rules.sort_values('lift', ascending=False)

# ==========================================
# 5. VIZUALIZARE REZULTATE ALGORITM
# ==========================================
print("\n--- Pasul 5: Generare grafic reguli ---")
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', s=100, alpha=0.7)
plt.colorbar(label='Lift')
plt.title('Harta Regulilor de Asociere (Support vs Confidence)', fontsize=14)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('fig3_analiza_reguli.png')
print("Salvat: fig3_analiza_reguli.png")

# Exportăm rezultatele brute pentru tabelele din raport
rules.to_csv('reguli_asociere_finale.csv', index=False)
print("Salvat: reguli_asociere_finale.csv")

print("\n--- ANALIZĂ COMPLETĂ FINALIZATĂ ---")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))