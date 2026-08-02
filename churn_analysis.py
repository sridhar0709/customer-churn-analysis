import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

np.random.seed(42)
n = 3000

contract = np.random.choice(['Month-to-month','One year','Two year'], n, p=[0.55,0.25,0.20])
tenure = np.random.exponential(20, n).clip(0,72).round().astype(int)
internet = np.random.choice(['Fiber optic','DSL','No internet'], n, p=[0.44,0.34,0.22])
tech_support = np.random.choice(['Yes','No'], n, p=[0.35,0.65])
monthly_charges = np.round(np.random.normal(65, 25, n).clip(18,140),2)
payment = np.random.choice(['Electronic check','Mailed check','Bank transfer','Credit card'], n, p=[0.34,0.19,0.24,0.23])
senior = np.random.choice([0,1], n, p=[0.84,0.16])
partner = np.random.choice(['Yes','No'], n, p=[0.48,0.52])
paperless = np.random.choice(['Yes','No'], n, p=[0.59,0.41])

# latent churn probability driven by known real-world churn drivers
logit = (
    -1.8
    + 1.35*(contract=='Month-to-month')
    - 0.55*(contract=='One year')
    - 1.10*(contract=='Two year')
    - 0.028*tenure
    + 0.65*(internet=='Fiber optic')
    - 0.45*(tech_support=='Yes')
    + 0.012*(monthly_charges-65)
    + 0.55*(payment=='Electronic check')
    + 0.25*(paperless=='Yes')
    + 0.30*senior
    - 0.20*(partner=='Yes')
)
prob = 1/(1+np.exp(-logit))
churn = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    'tenure':tenure,'contract':contract,'internet':internet,'tech_support':tech_support,
    'monthly_charges':monthly_charges,'payment':payment,'senior':senior,'partner':partner,
    'paperless':paperless,'churn':churn
})

overall_rate = df.churn.mean()

def rate_by(col):
    g = df.groupby(col)['churn'].agg(['mean','count']).reset_index()
    g.columns=[col,'churn_rate','n']
    return g.sort_values('churn_rate', ascending=False).to_dict(orient='records')

# tenure buckets
bins=[0,6,12,24,36,48,72]
labels=['0-6','7-12','13-24','25-36','37-48','49-72']
df['tenure_bucket']=pd.cut(df.tenure,bins=bins,labels=labels,include_lowest=True)

tenure_curve = df.groupby('tenure_bucket')['churn'].mean().reindex(labels).round(4).to_dict()

# monthly charges buckets
mc_bins=[0,40,60,80,100,140]
mc_labels=['<40','40-60','60-80','80-100','100+']
df['mc_bucket']=pd.cut(df.monthly_charges,bins=mc_bins,labels=mc_labels,include_lowest=True)
mc_curve = df.groupby('mc_bucket')['churn'].mean().reindex(mc_labels).round(4).to_dict()

# Feature importance via logistic regression on encoded features
X = pd.get_dummies(df[['tenure','contract','internet','tech_support','monthly_charges','payment','senior','partner','paperless']], drop_first=True)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
clf = LogisticRegression(max_iter=2000)
clf.fit(Xs, df.churn)
coefs = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
top_drivers = [{'feature':k, 'coef':round(float(v),3)} for k,v in coefs.head(10).items()]

# Segmentation: simple rule-based risk segments
def segment(row):
    risk=0
    if row.contract=='Month-to-month': risk+=1
    if row.tenure<12: risk+=1
    if row.internet=='Fiber optic': risk+=1
    if row.tech_support=='No': risk+=1
    if row.payment=='Electronic check': risk+=1
    if risk>=4: return 'High Risk'
    if risk>=2: return 'Medium Risk'
    return 'Low Risk'
df['segment']=df.apply(segment,axis=1)
seg_summary = df.groupby('segment').agg(n=('churn','size'), churn_rate=('churn','mean'), avg_monthly=('monthly_charges','mean'), avg_tenure=('tenure','mean')).round(3).reset_index().to_dict(orient='records')

output = {
    'overall_rate': round(float(overall_rate),4),
    'n_customers': int(n),
    'contract_rate': rate_by('contract'),
    'internet_rate': rate_by('internet'),
    'payment_rate': rate_by('payment'),
    'tech_support_rate': rate_by('tech_support'),
    'tenure_curve': tenure_curve,
    'mc_curve': mc_curve,
    'top_drivers': top_drivers,
    'seg_summary': seg_summary
}

with open('churn_data.json','w') as f:
    json.dump(output, f, indent=2, default=str)

print(json.dumps(output, indent=2, default=str)[:2000])
