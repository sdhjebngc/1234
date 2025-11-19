# 1. 库导入
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, classification_report

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 2. 数据导入与初步查看
script_dir = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
data_path = os.path.join(script_dir, 'Telco-Customer-Churn.csv')
df = pd.read_csv(data_path)

print("数据集形状：", df.shape)
print("\n前5行数据：")
print(df.head())
print("\n数据类型与缺失值：")
print(df.info())


# 3. 数据清洗
repl_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in repl_columns:
    df[col] = df[col].replace({'No internet service': 'No'})

df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df = df.dropna(subset=['TotalCharges'])
df.reset_index(drop=True, inplace=True)
df['TotalCharges'] = df['TotalCharges'].astype('float')

def transform_tenure(x):
    if x <= 12:
        return 'Tenure_1'
    elif x <= 24:
        return 'Tenure_2'
    elif x <= 36:
        return 'Tenure_3'
    elif x <= 48:
        return 'Tenure_4'
    elif x <= 60:
        return 'Tenure_5'
    else:
        return 'Tenure_over_5'
df['tenure_group'] = df['tenure'].apply(transform_tenure)

Id_col = ['customerID']
target_col = ['Churn']
cat_cols = df.nunique()[df.nunique() < 10].index.tolist()
num_cols = [col for col in df.columns if col not in cat_cols + Id_col + target_col]

print("\n清洗后数据集形状：", df.shape)


# 4. 探索性分析（保存为文件）
# 流失分布饼图
churn_counts = df['Churn'].value_counts()
trace0 = go.Pie(
    labels=['未流失客户', '流失客户'],
    values=churn_counts.values,
    hole=.5, rotation=90,
    marker=dict(colors=['rgb(154,203,228)', 'rgb(191,76,81)'])
)
fig = go.Figure(data=[trace0], layout=go.Layout(title='客户流失分布'))
py.offline.plot(fig, filename=os.path.join(script_dir, 'churn_distribution.html'))

# 在网时长与流失关系
def plot_bar(input_col, title, save_path):
    cross_table = pd.crosstab(df[input_col], df['Churn'], normalize='index')*100
    trace0 = go.Bar(x=cross_table.index, y=cross_table['No'], name='未流失')
    trace1 = go.Bar(x=cross_table.index, y=cross_table['Yes'], name='流失')
    fig = go.Figure(data=[trace0, trace1], layout=go.Layout(title=title, barmode='stack'))
    py.offline.plot(fig, filename=save_path)

plot_bar(
    'tenure_group', 
    '在网时长与流失关系', 
    os.path.join(script_dir, 'tenure_churn.html')
)

# 数值特征相关性热力图
plt.figure(figsize=(8,6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
plt.title('数值特征相关性热力图')
plt.savefig(os.path.join(script_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()


# 5. 数据转换
df_model = df.copy()

binary_cols = df_model.nunique()[df_model.nunique() == 2].index.tolist()
le = LabelEncoder()
for col in binary_cols:
    df_model[col] = le.fit_transform(df_model[col])

multi_cols = [col for col in cat_cols if col not in binary_cols + target_col]
df_model = pd.get_dummies(df_model, columns=multi_cols, drop_first=False)

X = df_model.drop(Id_col + target_col, axis=1)
y = df_model[target_col].values.ravel()


# 6. 特征筛选
fs = SelectKBest(score_func=f_classif, k=20)
X_fs = fs.fit_transform(X, y)

def get_selected_features(feature_data, fs_model):
    scores = fs_model.scores_
    indices = np.argsort(scores)[::-1][:fs_model.k]
    return feature_data.columns[indices].tolist()

selected_features = get_selected_features(X, fs)
print("筛选后的Top20特征：", selected_features)

X_train = pd.DataFrame(X_fs, columns=selected_features)


# 7. 划分训练集与测试集
X_train_split, X_test_split, y_train, y_test = train_test_split(
    X_train, y, test_size=0.2, random_state=0, stratify=y
)

num_cols_selected = [col for col in selected_features if col in num_cols]
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_split[num_cols_selected])
X_test_num_scaled = scaler.transform(X_test_split[num_cols_selected])

X_train_scaled = pd.concat([
    X_train_split.drop(num_cols_selected, axis=1).reset_index(drop=True),
    pd.DataFrame(X_train_num_scaled, columns=num_cols_selected)
], axis=1)

X_test_scaled = pd.concat([
    X_test_split.drop(num_cols_selected, axis=1).reset_index(drop=True),
    pd.DataFrame(X_test_num_scaled, columns=num_cols_selected)
], axis=1)

print("\n训练集形状：", X_train_scaled.shape)
print("测试集形状：", X_test_scaled.shape)


# 8. 模型训练与评估
def model_report(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return pd.DataFrame([{
        "模型": model_name,
        "准确率(Accuracy)": round(accuracy_score(y_test, y_pred), 2),
        "召回率(Recall)": round(recall_score(y_test, y_pred), 2),
        "精确率(Precision)": round(precision_score(y_test, y_pred), 2),
        "F1分数": round(f1_score(y_test, y_pred), 2),
        "AUC": round(roc_auc_score(y_test, y_score), 2)
    }])

models = [
    (LogisticRegression(random_state=0), "逻辑回归"),
    (KNeighborsClassifier(n_neighbors=5), "KNN"),
    (SVC(kernel='linear', probability=True, random_state=0), "线性SVM"),
    (SVC(kernel='rbf', probability=True, random_state=0), "RBF核SVM"),
    (MLPClassifier(hidden_layer_sizes=(8,), max_iter=50000, random_state=0), "多层感知机"),
    (GaussianNB(), "朴素贝叶斯"),
    (DecisionTreeClassifier(random_state=0), "决策树"),
    (RandomForestClassifier(n_estimators=100, random_state=0), "随机森林"),
    (LGBMClassifier(random_state=0), "LightGBM"),
    (XGBClassifier(random_state=0), "XGBoost")
]

results = []
for model, name in models:
    if name in ["朴素贝叶斯", "决策树", "随机森林", "LightGBM", "XGBoost"]:
        report = model_report(model, X_train_split, X_test_split, y_train, y_test, name)
    else:
        report = model_report(model, X_train_scaled, X_test_scaled, y_train, y_test, name)
    results.append(report)

model_results = pd.concat(results, ignore_index=True)
print("\n多模型评估结果：")
print(model_results.sort_values(by="AUC", ascending=False))

# 决策树调参
param_grid = {
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy'],
    'max_depth': range(3, 10)
}
dt = DecisionTreeClassifier(random_state=25)
grid_search = GridSearchCV(
    estimator=dt, param_grid=param_grid,
    scoring='f1', cv=10, n_jobs=-1
)
grid_search.fit(X_train_split, y_train)

print("\n决策树最优参数：", grid_search.best_params_)
best_dt = grid_search.best_estimator_
y_dt_pred = best_dt.predict(X_test_split)
print("\n调优后决策树测试集报告：")
print(classification_report(y_test, y_dt_pred))