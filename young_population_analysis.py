import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

coefficients_file_path = 'topic_coefficients_by_region.csv'
mesh_shapefile_path = './500m_mesh_suikei_2018_shape_13/500m_mesh_2018_13.shp'
tokyo_boundary_path = './N03-20240101_13_GML/N03-20240101_13.geojson'

coefficients_df = pd.read_csv(coefficients_file_path)
mesh_gdf = gpd.read_file(mesh_shapefile_path)
tokyo_gdf = gpd.read_file(tokyo_boundary_path)

# addr1_2と市区町村コードの対応設定
coefficients_df['N03_007'] = '13' + coefficients_df['region_code'].astype(str).str.zfill(3)

# topic_2の有意な地域を抽出
significant_topic_2 = coefficients_df[(coefficients_df['topic_2_significant'] == 1)]

# 市区町村境界データと有意なtopic_2の地域をマージ
tokyo_gdf = tokyo_gdf.rename(columns={'N03_007': '市区町村コード'})
merged_gdf = tokyo_gdf.merge(significant_topic_2[['N03_007', 'topic_2']], left_on='市区町村コード', right_on='N03_007', how='left')

# 存在しない市区町村も含めるため、`topic_2`の欠損値に中間色を適用
merged_gdf['topic_2'] = merged_gdf['topic_2'].fillna(0)

# 若者の人口増減率を計算
mesh_gdf['PTN_2020_0_18'] = (
    mesh_gdf['PT1_2020'] +
    mesh_gdf['PT2_2020'] +
    mesh_gdf['PT3_2020'] +
    mesh_gdf['PT4_2020']
)
mesh_gdf['PTN_2050_0_18'] = (
    mesh_gdf['PT1_2050'] +
    mesh_gdf['PT2_2050'] +
    mesh_gdf['PT3_2050'] +
    mesh_gdf['PT4_2050']
)
mesh_gdf['population_change_rate'] = (
    (mesh_gdf['PTN_2050_0_18'] - mesh_gdf['PTN_2020_0_18']) / mesh_gdf['PTN_2020_0_18']
) * 100

# メッシュの中心点を計算してプロット用に準備
mesh_gdf['centroid'] = mesh_gdf.geometry.centroid
point_gdf = mesh_gdf.set_geometry('centroid')

# 若者人口の増減カテゴリーの作成
def categorize_population_change(row):
    if row['population_change_rate'] > 0:
        return 'Increase'
    elif 0 >= row['population_change_rate'] > -50:
        return 'Decrease (0%~50%)'
    elif row['population_change_rate'] <= -50:
        return 'Significant Decrease (50%~)'

point_gdf['population_category'] = point_gdf.apply(categorize_population_change, axis=1)

# 色設定
population_color_dict = {
    'Increase': 'darkgreen',         # 増加: 超濃い緑
    'Decrease (0%~50%)': 'lightgreen',             # 少し減少: 緑
    # 'Moderate Decrease (30%~50%)': 'lightgreen',    # 30%以上50%未満減少: 薄い緑
    'Significant Decrease (50%~)': '#e6ffe6'     # 50%以上減少: ほぼ白緑
}

# プロット
fig, ax = plt.subplots(1, 1, figsize=(15, 15))

# 東京都市区町村境界データとtopic_2の係数に応じたヒートマップをプロット
cmap = mcolors.LinearSegmentedColormap.from_list("topic2_cmap", ["#7f7fff", "#c0c0c0", "#ff7f7f"], N=256)
norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
merged_gdf.plot(column='topic_2', ax=ax, cmap=cmap, edgecolor='black', legend=True, norm=norm,
                legend_kwds={'label': "Topic 3 Coefficient", 'orientation': "horizontal"})

# 各メッシュの中心点を若者人口の増減カテゴリーごとに色分けしてプロット
for category, color in population_color_dict.items():
    subset = point_gdf[point_gdf['population_category'] == category]
    subset.plot(ax=ax, color=color, markersize=5, label=category)

# 緯度経度の範囲設定（東京都の範囲にズーム）
ax.set_xlim([138.9, 139.95])  # 経度の範囲
ax.set_ylim([35.47, 35.93])   # 緯度の範囲

plt.legend(title="Population Change Categories")
plt.title("Topic 3 Coefficient by Municipality and Youth Population Change in Tokyo")
plt.tight_layout()
plt.savefig("topic3_population_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
