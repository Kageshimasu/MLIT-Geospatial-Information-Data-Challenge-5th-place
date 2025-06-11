import pandas as pd

tokyo_data = pd.read_csv('./data/train.csv')
tokyo_data = tokyo_data[tokyo_data['addr1_1'] == 13].copy()
tokyo_data = tokyo_data[(tokyo_data['money_room'] > 0) & (tokyo_data['unit_area'] > 0)].copy()
tokyo_data = tokyo_data.dropna(subset=['walk_distance1'])

# 必要なカラムのみ選択
tokyo_data = tokyo_data[['target_ym', 'money_room', 'lat', 'lon', 'addr1_1', 'addr1_2', 'unit_area', 
                         'madori_number_all', 'madori_kind_all', 'building_type', 
                         'building_structure', 'building_tag_id', 'statuses', 'year_built', 'walk_distance1']]

# 築浅を作る
tokyo_data['target_ym'] = pd.to_datetime(tokyo_data['target_ym'], format='%Y%m', errors='coerce')
tokyo_data['year_built'] = pd.to_datetime(tokyo_data['year_built'], format='%Y%m', errors='coerce')
tokyo_data = tokyo_data.dropna(subset=['target_ym', 'year_built']).copy()
tokyo_data['築年数'] = (tokyo_data['target_ym'].dt.year - tokyo_data['year_built'].dt.year) - \
                       ((tokyo_data['target_ym'].dt.month < tokyo_data['year_built'].dt.month).astype(int))
tokyo_data['築浅'] = (tokyo_data['築年数'] <= 5).astype(int)

# "駅近" カテゴリ変数の作成
tokyo_data['駅近'] = tokyo_data['walk_distance1'].apply(lambda x: 1 if x <= 800 else 0)

# 各種マッピング
building_type_map = {1: 'マンション', 3: 'アパート'}
building_structure_map = {
    1: '木造', 2: 'ブロック', 3: '鉄骨造', 4: 'RC', 5: 'SRC', 6: 'PC', 7: 'HPC', 
    9: 'その他', 10: '軽量鉄骨', 11: 'ALC', 12: '鉄筋ブロック', 13: 'CFT(コンクリート充填鋼管)'
}
madori_kind_map = {
    10: 'R', 20: 'K', 25: 'SK', 30: 'DK', 35: 'SDK', 40: 'LK', 45: 'SLK', 50: 'LDK', 55: 'SLDK'
}

# マッピングによるカテゴリ名の変換とOne-Hotエンコーディング
tokyo_data['建物の種類'] = tokyo_data['building_type'].map(building_type_map).fillna('欠損')
tokyo_data = pd.get_dummies(tokyo_data, columns=['建物の種類'])

tokyo_data['建物構造'] = tokyo_data['building_structure'].map(building_structure_map)
tokyo_data = pd.get_dummies(tokyo_data, columns=['建物構造'])

tokyo_data['間取り種類'] = tokyo_data['madori_kind_all'].map(madori_kind_map)
tokyo_data = pd.get_dummies(tokyo_data, columns=['間取り種類'])

# 間取りの全ての種類の列があるか確認し、ない場合は0を追加
for kind in ['R', 'K', 'SK', 'DK', 'SDK', 'LK', 'SLK', 'LDK', 'SLDK']:
    col_name = f'間取り種類_{kind}'
    if col_name not in tokyo_data.columns:
        tokyo_data[col_name] = 0


# 部屋数の最大値を確認
max_rooms = tokyo_data['madori_number_all'].max()
room_numbers = range(1, int(max_rooms)+1)

# One-Hotエンコーディング
tokyo_data = pd.get_dummies(tokyo_data, columns=['madori_number_all'], prefix='部屋数_', prefix_sep='')

# 部屋数の全ての種類の列があるか確認し、ない場合は0を追加
room_number_cols = []
for room in room_numbers:
    col_name = f'部屋数_{room}'
    room_number_cols.append(col_name)
    if col_name not in tokyo_data.columns:
        tokyo_data[col_name] = 0

# building_tag_id と statuses の処理（マルチラベルのOne-Hotエンコーディング）
building_tag = pd.read_csv('./building_tag.csv')
statuses_tag = pd.read_csv('./statuses.csv')

# building_tag_id のOne-Hotエンコーディング（b_ prefix）
for idx, row in tokyo_data.iterrows():
    tags = str(row['building_tag_id']).split('/')
    for tag in tags:
        if tag.isdigit():
            tag_name = building_tag.loc[building_tag['tag_id'] == int(tag), 'tag_name'].values[0]
            tokyo_data.loc[idx, f'b_{tag_name}'] = 1

# statuses のOne-Hotエンコーディング（s_ prefix）
for idx, row in tokyo_data.iterrows():
    statuses = str(row['statuses']).split('/')
    for status in statuses:
        if status.isdigit():
            status_name = statuses_tag.loc[statuses_tag['tag_id'] == int(status), 'tag_name'].values[0]
            tokyo_data.loc[idx, f's_{status_name}'] = 1

# NaNを0に変換し、全てのフラグ列をint型に統一
tokyo_data.fillna(0, inplace=True)
for col in tokyo_data.columns:
    # latとlonはfloatのままにするため、それ以外のfloat64やbool型のみをintに変換
    if col not in ['lat', 'lon', 'unit_area'] and (tokyo_data[col].dtype == 'float64' or tokyo_data[col].dtype == 'bool'):
        tokyo_data[col] = tokyo_data[col].astype(int)

# 必要なカラムのリストを用意
basic_columns = ['target_ym', 'money_room', 'lat', 'lon', 'unit_area', 'addr1_1', 'addr1_2']

# 間取り種類のOne-Hotエンコーディング用に作成したカラムのリスト
madori_kind_columns = [f'間取り種類_{kind}' for kind in ['R', 'K', 'SK', 'DK', 'SDK', 'LK', 'SLK', 'LDK', 'SLDK']]

# b_ と s_ prefixで生成されたOne-Hotカラムの取得
building_tag_columns = [col for col in tokyo_data.columns if 'b_' in col]
statuses_columns = [col for col in tokyo_data.columns if 's_' in col]

# 距離のカラム
distance_columns = ['駅近']

# 必要なカラムのリストを統合
required_columns = basic_columns  + building_tag_columns + statuses_columns + madori_kind_columns + room_number_cols + ['築浅'] + distance_columns

# 必要なカラムのみを保持し、不要なカラムを削除
tokyo_data = tokyo_data[required_columns]

# 削除するカラムを指定
columns_to_drop = [
    'b_エレベーター',
    'b_オートロック',
    'b_オール電化',
    'b_プロパンガス',
    'b_防犯カメラ',
    'b_タイル貼り'
]

# 指定したカラムを削除
tokyo_data_dropped = tokyo_data.drop(columns=columns_to_drop, errors='ignore')

# 最終結果を確認
tokyo_data_dropped.to_csv("./lda_feature_5.csv", index=False)
