from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
import matplotlib.pyplot as plt1
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# اين كلاس براي مدل گروهي تعريف ده كه يك فايل جداگانه دارد كه لازم است به پروژه اضافه شود
class ensmodel:
    def __init__(self, models, weights=None):
        self.n = len(models)
        self.models = models
        self.weights = weights
      
        # If weights is empty, we'll simply use the arithmetic mean
        if self.weights is None: self.weights = [(1/self.n)]*self.n
    def predict(self, x):
        pred = self.weights[0]*self.models[0].predict(x)
        # Calculate the weighted average
        for i in range(1, self.n): pred +=       self.weights[i]*self.models[i].predict(x)
        return pred

# فايل ورودي
#df = pd.read_csv (r'.\DCORE.cSV')
df = pd.read_csv (r'.\DCORE.csv')
# چاپ دو خط اول فايل جهت تست
print(df.head(2))

# تعيين مقدار X,Y
y = df.iloc[:, [11]].values
scaler = MinMaxScaler()
x = df.iloc[:, [2,3,4,5,6,7,8,10]].values

# استانداردسازي داده
scaled = scaler.fit_transform(x)
#print(scaled)
x=scaled


# define the model  20 درصد را براي مجموعه آزمايش  تعيين مي كنيم

validation_size=0.20
best_acc=0
num_fold=10
    
 


print("-------------//////////////////////////////---------------")
# ماتريسهاي آموزش و آزمايش را تعيين مي كنيم
X_train,X_validation,Y_train,Y_validation=train_test_split(x,y,test_size=validation_size,random_state=8)

# الگوريتم درخت تصميم را بر روي مجموعه آموزش اعمال مي كنيم
xg=DecisionTreeRegressor(criterion='squared_error',max_depth=8, random_state=4)
xg.fit(np.real(X_train),np.ravel(Y_train))
y_p=xg.predict(X_validation)
print("DTR:", r2_score(Y_validation,y_p)*100)
   
# (كار اضافه انجام داديم مي تواند نباشد براي اينكه نتايج تك تك را ببينيم) الگوريتم درخت تصميم افزوده  را بر روي مجموعه آموزش اعمال مي كنيم
X_train,X_validation,Y_train,Y_validation=train_test_split(x,y,test_size=validation_size,random_state=4)
model = ExtraTreesRegressor(n_estimators=400, random_state=4,criterion='absolute_error')
model.fit(np.real(X_train),np.ravel(Y_train)) 
y_pred = model.predict(X_validation)
print("ETR",r2_score(Y_validation,y_pred)*100)


#  الگوريتم گراديان تقويت شده را بر روي مجموعه آموزش اعمال مي كنيم(كار اضافه انجام داديم مي تواند نباشد براي اينكه نتايج تك تك را ببينيم)
X_train,X_validation,Y_train,Y_validation=train_test_split(x,y,test_size=validation_size,random_state=21)   
gb=GradientBoostingRegressor(n_estimators=450, random_state=21,loss='lad',max_depth=8) 
gb.fit(np.real(X_train),np.ravel(Y_train))
ygb = gb.predict(X_validation)
print("GBR",r2_score(Y_validation,ygb)*100)

   
# الگوريتم درخت تصميم  را با درخت تصميم افزوده  و الگوريتم گراديان تقويت شده تركيب كرده  و تركيب آنها را بر روي مجموعه آموزش اعمال مي كنيم
green = ensmodel([model, xg], [0.9, 0.1])
m=ensmodel([green, gb], [0.9, 0.1])
yens=m.predict(np.real(X_validation))
print("yens",r2_score(Y_validation,yens)*100)
print("MSE:", mean_squared_error(Y_validation,yens))
print("MAE:", mean_absolute_error(Y_validation,yens))


# نتايج تخمين را در فايل اكسل ذخيره مي كنيم
result = np.column_stack((Y_validation,yens))
import xlsxwriter

workbook = xlsxwriter.Workbook('arrays.xlsx')
worksheet = workbook.add_worksheet()
row = 0

for col, data in enumerate(result):
    worksheet.write_column(row,col,data)

workbook.close()