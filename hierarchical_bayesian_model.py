import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

topic_num = 13
data = pd.read_csv(f"tokyo_data_with_topics_5_topic{topic_num}.csv")
data = data[(data['money_room'] > 0) & (data['unit_area'] > 0)].copy()

# 価格の計算と標準化
data['log_price'] = np.log(data['money_room'])
price_mean = data['log_price'].mean()
price_std = data['log_price'].std()
data['log_price_std'] = (data['log_price'] - price_mean) / price_std

# unit_areaの標準化
data['log_unit_area'] = np.log(data['unit_area'])
data['log_unit_area_std'] = (data['log_unit_area'] - data['log_unit_area'].mean()) / data['log_unit_area'].std()

# 地域コードの設定
data['addr1_2_code'] = data['addr1_2'].astype('category').cat.codes
num_regions = data['addr1_2_code'].nunique()

# トピック列の取得
topic_cols = [f'topic_{i}' for i in range(topic_num)]
topics = data[topic_cols].values

# データの分割
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# モデルに存在しないaddr1_2_codeを持つtest_dataの行を除外
train_unique_regions = train_data['addr1_2_code'].unique()
test_unique_regions = test_data['addr1_2_code'].unique()
invalid_regions = test_unique_regions[~np.isin(test_unique_regions, train_unique_regions)]
if len(invalid_regions) > 0:
    test_data = test_data[test_data['addr1_2_code'].isin(train_unique_regions)].copy()

with pm.Model() as hierarchical_model:
    mu_beta = pm.Normal('mu_beta', mu=0, sigma=1, shape=topic_num)
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=1, shape=topic_num)
    
    beta_topics_region = pm.Normal('beta_topics_region', mu=mu_beta, sigma=sigma_beta, shape=(num_regions, topic_num))
    beta_intercept_region = pm.Normal('beta_intercept_region', mu=0, sigma=1, shape=num_regions)
    beta_unit_area = pm.Normal('beta_unit_area', mu=0, sigma=1)

    region_indices = train_data['addr1_2_code'].values
    mu = (
        beta_intercept_region[region_indices] + 
        pm.math.sum(beta_topics_region[region_indices] * train_data[topic_cols].values, axis=1) +
        beta_unit_area * train_data['log_unit_area_std']
    ) 
    sigma = pm.Exponential('sigma', 1)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=train_data['log_price_std'].values)

    approx = pm.fit(n=30000, method='advi', random_seed=42)
    trace = approx.sample(3000)

# 結果の可視化
az.plot_trace(trace, var_names=['beta_topics_region', 'sigma'])
plt.tight_layout()
# plt.savefig("trace_plot_hierarchical_variational.png", dpi=300, bbox_inches="tight")

summary = az.summary(trace, var_names=['beta_topics_region', 'sigma'])
print(summary)

# テストデータへの予測と評価
beta_topics_region_samples = trace.posterior['beta_topics_region'].stack(samples=("chain", "draw")).values
beta_intercept_region_samples = trace.posterior['beta_intercept_region'].stack(samples=("chain", "draw")).values
beta_unit_area_samples = trace.posterior['beta_unit_area'].stack(samples=("chain", "draw")).values

test_unit_area = test_data['log_unit_area_std'].values
test_regions = test_data['addr1_2_code'].values
test_topics = test_data[topic_cols].values

# 予測値の計算
intercepts = beta_intercept_region_samples[test_regions, :]
unit_area_effect = test_unit_area[:, np.newaxis] * beta_unit_area_samples[np.newaxis, :]
region_effects = np.einsum('ijk,ij->ik', beta_topics_region_samples[test_regions, :, :], test_topics)
mu_test = intercepts + region_effects + unit_area_effect
predictions = mu_test.mean(axis=1)

# 予測精度の評価
log_price_std_test = test_data['log_price_std'].values
mse = mean_squared_error(log_price_std_test, predictions)
mae = mean_absolute_error(log_price_std_test, predictions)
r2 = r2_score(log_price_std_test, predictions)

print("Mean Square Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R² Score:", r2)

# 有意なトピック効果の抽出
def get_significant_region_topic_flags(samples, hdi_prob=0.94):
    significance_data = []
    for region in range(samples.shape[0]):
        flags = []
        for topic in range(samples.shape[1]):
            sample = samples[region, topic, :]
            hdi = az.hdi(sample, hdi_prob=hdi_prob)
            flag = 1 if (hdi[0] > 0 or hdi[1] < 0) else 0
            flags.append(flag)
        significance_data.append(flags)
    return pd.DataFrame(significance_data, columns=[f'{col}_significant' for col in topic_cols])

# トピック係数と有意性判定の保存
region_ids = pd.factorize(data['addr1_2'])[1]
topic_coefficients = beta_topics_region_samples.mean(axis=2)
topic_coefficients_df = pd.DataFrame(topic_coefficients, columns=topic_cols, index=region_ids)
topic_coefficients_df.reset_index(inplace=True)
topic_coefficients_df.rename(columns={'index': 'region_code'}, inplace=True)

significant_flags_df = get_significant_region_topic_flags(beta_topics_region_samples)
output_df = pd.concat([topic_coefficients_df, significant_flags_df], axis=1)
# output_df.to_csv('topic_coefficients_by_region.csv', index=False)
print("Topic coefficients by region saved to 'topic_coefficients_by_region.csv'")

# 予測結果の可視化
plt.figure(figsize=(10, 6))
plt.scatter(log_price_std_test, predictions, alpha=0.5, label='Predictions')
plt.plot([log_price_std_test.min(), log_price_std_test.max()],
         [log_price_std_test.min(), log_price_std_test.max()],
         'r--', label='Perfect Prediction')
plt.xlabel('Actual log_price_std')
plt.ylabel('Predicted log_price_std')
plt.legend()
plt.tight_layout()
# plt.savefig("actual_vs_predicted_hierarchical_variational.png", dpi=300, bbox_inches="tight")
plt.show()

# 残差の分布をプロット
residuals = log_price_std_test - predictions
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True, color='skyblue')
plt.title('Residuals Distribution (Hierarchical Model)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.tight_layout()
# plt.savefig("residuals_distribution_hierarchical_variational.png", dpi=300, bbox_inches="tight")
plt.show()
