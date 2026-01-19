import pickle

# 1. 加载
with open('col_stats.pkl', 'rb') as f:
    stats = pickle.load(f)

# 2. 获取 file_id 的统计字典
f_stats = stats['file_id']

# 3. 动态寻找现有的键（比如 COUNT），利用它的类型来构造新键
existing_key = list(f_stats.keys())[0]
StatTypeClass = type(existing_key)

# 尝试手动构造一个 CATEGORIES 成员
# 大多数枚举类支持通过名称获取成员
try:
    cat_key = StatTypeClass['CATEGORIES']
except:
    # 如果还是不行，直接用字符串 'CATEGORIES'
    # 在很多版本的 torch_frame 中，字符串键也是兼容的
    cat_key = 'CATEGORIES'

# 4. 注入 46 个类别 (确保与模型训练时的数量一致)
f_stats[cat_key] = list(range(46))

# 5. 保存
with open('col_stats.pkl', 'wb') as f:
    pickle.dump(stats, f)

print('✅ 已物理修复 col_stats.pkl')
print('现在的键为:', list(f_stats.keys()))