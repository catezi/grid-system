import pymysql
# 连接数据库
db = pymysql.connect(
  host='127.0.0.1',
  port=3306, 
  user='lc', 
  passwd='liuchang0810', 
  db='grid', 
  charset='utf8'
)
# 创建数据表
cursor = db.cursor()
# 插入数据
a = [17.4, 2.5, 21.9, 79.7, 1.23]
sql = "INSERT INTO  grid_powergrid (timestep, vTime) VALUES (%s, %s)"
val = ("001", str(a))

cursor.execute(sql, val)
db.commit()
  
# 关闭数据库连接
db.close()