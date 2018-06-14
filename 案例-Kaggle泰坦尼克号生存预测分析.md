
## 数据采集和理解


```python
#设置ast_node_interactivity = "all"使得可以同时输出多条语句
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#导入包
import pandas as pd
import numpy as np
#导入数据
train=pd.read_csv(r'E:\python\data\titanic\train.csv')
test=pd.read_csv(r'E:\python\data\titanic\test.csv')
print('训练集数据规模:{}'.format(train.shape))
print('测试集数据规模:{}'.format(test.shape))
```

    训练集数据规模:(891, 12)
    测试集数据规模:(418, 11)
    

训练数据集比测试数据集的列多一个，即Survived值。由于它是预测的生存值，所以，在测试数据集中没有。


```python
#查看训练集信息
train.head()
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



为了方便对训练数据和测试数据进行清洗，将训练数据和测试数据进行合并


```python
#通过设置ignore_index=True参数，合并后的数据集会重新生成一个index
full=pd.concat([train,test],ignore_index=True)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>



针对每一个字段做一个简单的解释：
PassengerId: 乘客ID；

Survived: 生存情况，0代表不幸遇难，1代表存活；

Pclass: 仓位等级，1为一等舱，2为二等舱，3为三等舱；

Name: 乘客姓名；

Sex: 性别；

Age: 年龄；

SibSp: 乘客在船上的兄妹姐妹数/配偶数（即同代直系亲属数）；

Parch: 乘客在船上的父母数/子女数（即不同代直系亲属数）；

Ticket: 船票编号；

Fare: 船票价格；

Cabin: 客舱号；

Embarked: 登船港口（S: Southampton; C: Cherbourg Q: Queenstown）


```python
#查看数据描述性统计
full.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1046.000000</td>
      <td>1308.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>33.295479</td>
      <td>0.385027</td>
      <td>655.000000</td>
      <td>2.294882</td>
      <td>0.498854</td>
      <td>0.383838</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.413493</td>
      <td>51.758668</td>
      <td>0.865560</td>
      <td>378.020061</td>
      <td>0.837836</td>
      <td>1.041658</td>
      <td>0.486592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>7.895800</td>
      <td>0.000000</td>
      <td>328.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>14.454200</td>
      <td>0.000000</td>
      <td>655.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>39.000000</td>
      <td>31.275000</td>
      <td>0.000000</td>
      <td>982.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>9.000000</td>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#因为，describe()函数只能查看数据类型的描述统计信息，无法查看类似字符类型的信息。故，需用info()函数进一步查看每一列的数据信息。
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1046 non-null float64
    Cabin          295 non-null object
    Embarked       1307 non-null object
    Fare           1308 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

数据的总行数为1309行，其中，Age一栏中263列有缺失项；Fare一栏中1列有缺失项；Survived一栏只有891列，刚好对应训练数据集的行数。除了Age和Fare以外，Cabin/Embarked也有缺失项。
也可以用另一个命令，查看缺失项信息


```python
full.isnull().sum()
```




    Age             263
    Cabin          1014
    Embarked          2
    Fare              1
    Name              0
    Parch             0
    PassengerId       0
    Pclass            0
    Sex               0
    SibSp             0
    Survived        418
    Ticket            0
    dtype: int64



## 数据清洗

如果是数值类型，使用平均值或者中位数进行填充

年龄(Age) 最小值为0.17，不存在0值，其数据缺失率为263/1309=20.09%，由于Age的平均数与中位数接近，故选择平均值作为缺失项的填充值。


```python
full['Age']=full['Age'].fillna(full['Age'].mean())
```

船票价格(Fare)一栏数据缺失项仅为一行，且存在票价为0的记录，如下：


```python
full.loc[full['Fare']==0,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>179</th>
      <td>36.000000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Leonard, Mr. Lionel</td>
      <td>0</td>
      <td>180</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>LINE</td>
    </tr>
    <tr>
      <th>263</th>
      <td>40.000000</td>
      <td>B94</td>
      <td>S</td>
      <td>0.0</td>
      <td>Harrison, Mr. William</td>
      <td>0</td>
      <td>264</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>112059</td>
    </tr>
    <tr>
      <th>271</th>
      <td>25.000000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>0</td>
      <td>272</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>1.0</td>
      <td>LINE</td>
    </tr>
    <tr>
      <th>277</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>0</td>
      <td>278</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239853</td>
    </tr>
    <tr>
      <th>302</th>
      <td>19.000000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>0</td>
      <td>303</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>LINE</td>
    </tr>
    <tr>
      <th>413</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>0</td>
      <td>414</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239853</td>
    </tr>
    <tr>
      <th>466</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Campbell, Mr. William</td>
      <td>0</td>
      <td>467</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239853</td>
    </tr>
    <tr>
      <th>481</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>0</td>
      <td>482</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239854</td>
    </tr>
    <tr>
      <th>597</th>
      <td>49.000000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Johnson, Mr. Alfred</td>
      <td>0</td>
      <td>598</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>LINE</td>
    </tr>
    <tr>
      <th>633</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>0</td>
      <td>634</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>112052</td>
    </tr>
    <tr>
      <th>674</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>0</td>
      <td>675</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239856</td>
    </tr>
    <tr>
      <th>732</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Knight, Mr. Robert J</td>
      <td>0</td>
      <td>733</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>239855</td>
    </tr>
    <tr>
      <th>806</th>
      <td>39.000000</td>
      <td>A36</td>
      <td>S</td>
      <td>0.0</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>0</td>
      <td>807</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>112050</td>
    </tr>
    <tr>
      <th>815</th>
      <td>29.881138</td>
      <td>B102</td>
      <td>S</td>
      <td>0.0</td>
      <td>Fry, Mr. Richard</td>
      <td>0</td>
      <td>816</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>112058</td>
    </tr>
    <tr>
      <th>822</th>
      <td>38.000000</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>0</td>
      <td>823</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0.0</td>
      <td>19972</td>
    </tr>
    <tr>
      <th>1157</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
      <td>Chisholm, Mr. Roderick Robert Crispin</td>
      <td>0</td>
      <td>1158</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>NaN</td>
      <td>112051</td>
    </tr>
    <tr>
      <th>1263</th>
      <td>49.000000</td>
      <td>B52 B54 B56</td>
      <td>S</td>
      <td>0.0</td>
      <td>Ismay, Mr. Joseph Bruce</td>
      <td>0</td>
      <td>1264</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>NaN</td>
      <td>112058</td>
    </tr>
  </tbody>
</table>
</div>



让我们先看下那些票价不为0的数据，其不同仓位等级的票均价


```python
full.loc[full['Fare']!=0,:].groupby('Pclass')['Fare'].mean()
```




    Pclass
    1    89.447482
    2    21.648108
    3    13.378473
    Name: Fare, dtype: float64



我们可以用这三个均值分别填充不同仓位其票价为0的记录,并用所有记录的均值填充na


```python
full.loc[(full['Fare']==0)&(full['Pclass']==1),'Fare']=89.4
full.loc[(full['Fare']==0)&(full['Pclass']==2),'Fare']=21.6
full.loc[(full['Fare']==0)&(full['Pclass']==3),'Fare']=13.4
full['Fare']=full['Fare'].fillna(full['Fare'].mean())
full.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>33.913981</td>
      <td>0.385027</td>
      <td>655.000000</td>
      <td>2.294882</td>
      <td>0.498854</td>
      <td>0.383838</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.883193</td>
      <td>51.776484</td>
      <td>0.865560</td>
      <td>378.020061</td>
      <td>0.837836</td>
      <td>1.041658</td>
      <td>0.486592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>3.170800</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.000000</td>
      <td>7.925000</td>
      <td>0.000000</td>
      <td>328.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.881138</td>
      <td>14.500000</td>
      <td>0.000000</td>
      <td>655.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35.000000</td>
      <td>31.387500</td>
      <td>0.000000</td>
      <td>982.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>9.000000</td>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



如果是分类数据，使用最常见的类别取代


```python
#查看Embarked列中各value的数目
full['Embarked'].value_counts()
```




    S    914
    C    270
    Q    123
    Name: Embarked, dtype: int64



可以看到登船港口Embarked最常见的类别是"S"，故，使用其填充缺失项。


```python
full['Embarked']=full['Embarked'].fillna('S')
```

如果是字符串类型，按照实际情况填写，无法追踪的信息，用"Unknow"填充。处理Cabin缺失值 U代表Unknow


```python
full['Cabin']=full['Cabin'].fillna('U')
```


```python
full.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>33.913981</td>
      <td>0.385027</td>
      <td>655.000000</td>
      <td>2.294882</td>
      <td>0.498854</td>
      <td>0.383838</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.883193</td>
      <td>51.776484</td>
      <td>0.865560</td>
      <td>378.020061</td>
      <td>0.837836</td>
      <td>1.041658</td>
      <td>0.486592</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>3.170800</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>22.000000</td>
      <td>7.925000</td>
      <td>0.000000</td>
      <td>328.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29.881138</td>
      <td>14.500000</td>
      <td>0.000000</td>
      <td>655.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>35.000000</td>
      <td>31.387500</td>
      <td>0.000000</td>
      <td>982.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>9.000000</td>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
    Age            1309 non-null float64
    Cabin          1309 non-null object
    Embarked       1309 non-null object
    Fare           1309 non-null float64
    Name           1309 non-null object
    Parch          1309 non-null int64
    PassengerId    1309 non-null int64
    Pclass         1309 non-null int64
    Sex            1309 non-null object
    SibSp          1309 non-null int64
    Survived       891 non-null float64
    Ticket         1309 non-null object
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB
    

## 特征提取
如何知道哪些特征比较重要呢？通常需要与熟悉业务逻辑的人进行沟通，将业务人员说的特征反映到代码中，并通过实验和经验不断尝试，产生新的特征。

### Sex（性别）：


```python
#将性别的值映射为数值
#male对应数值1，female对应数值0
sex_dict={'male':1,'female':0}
full['Sex']=full['Sex'].map(sex_dict)
full['Sex'].head()
```




    0    1
    1    0
    2    0
    3    0
    4    1
    Name: Sex, dtype: int64



### Embarked（登船港口）：
使用get_dummies进行one-hot编码


```python
#使用get_dummies进行one-hot编码，产生虚拟变量（dummy variables），列名前缀(prefix)是Embarked
EmbarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked')
EmbarkedDf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 将EmbarkedDf的特征添加至full数据集
full=pd.concat([full,EmbarkedDf],axis=1)#axis=1表示按列插入数据
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



因为已经使用登船港口(Embarked)进行了one-hot编码产生了它的虚拟变量（dummy variables）,
所以这里把登船港口(Embarked)删掉


```python
full=full.drop('Embarked',axis=1)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Pclass(客舱等级)
方法同上


```python
PclassDf=pd.get_dummies(full['Pclass'],prefix='Pclass')
PclassDf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
full=pd.concat([full,PclassDf],axis=1)
full=full.drop('Pclass',axis=1)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Name(乘客姓名)：


```python
full['Name'].head()
```




    0                              Braund, Mr. Owen Harris
    1    Cumings, Mrs. John Bradley (Florence Briggs Th...
    2                               Heikkinen, Miss. Laina
    3         Futrelle, Mrs. Jacques Heath (Lily May Peel)
    4                             Allen, Mr. William Henry
    Name: Name, dtype: object



从上述Name字符串中发现每个名字里面都包含了头衔，我们可以获取到每个乘客的头衔，它可以帮助我们分析到更多有用的信息。


```python
def getTitle(name):
    s1=name.split(',')[1]
    s2=s1.split('.')[0]
    return s2.strip()#移除字符串头尾空格
full['Title']=full['Name'].map(getTitle)
full['Title'].value_counts()
```




    Mr              757
    Miss            260
    Mrs             197
    Master           61
    Rev               8
    Dr                8
    Col               4
    Ms                2
    Major             2
    Mlle              2
    Capt              1
    Don               1
    Mme               1
    Jonkheer          1
    Dona              1
    Sir               1
    the Countess      1
    Lady              1
    Name: Title, dtype: int64



将上述头衔对应到下面的几种类别中:

Officer政府官员；
Royalty王室（皇室）；
Mr已婚男士；
Mrs已婚妇女；
Miss年轻未婚女子；
Master有技能的人/教师


```python
title_dict={"Capt":"Officer","Col":"Officer","Major":"Officer","Jonkheer":"Royalty","Don":"Royalty","Sir":"Royalty","Dr":"Officer","Rev":"Officer"
            ,"the Countess":"Royalty","Dona":"Royalty","Mme":"Mrs","Mlle":"Miss","Ms":"Mrs","Mr" :"Mr","Mrs" :"Mrs","Miss" :"Miss"
            ,"Master" :"Master", "Lady" : "Royalty"}
full['Title']=full['Title'].map(title_dict)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
full['Title'].value_counts()
```




    Mr         757
    Miss       262
    Mrs        200
    Master      61
    Officer     23
    Royalty      6
    Name: Title, dtype: int64



利用上述头衔数据框进行One-hot编码


```python
TitleDf=pd.get_dummies(full['Title'])#One-hot编码
full=pd.concat([full,TitleDf],axis=1)#将特征添加至源数据集
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>...</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Title</th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Mr</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mrs</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Miss</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Mrs</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Mr</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
full=full.drop(['Name','Title'],axis=1)#删掉不需要的列
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>...</th>
      <th>Embarked_S</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>U</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>C85</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>U</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>C123</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>U</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



### Cabin(客舱号)：
客场号的类别值是首字母，因此我们提取客舱号的首字母为特征。


```python
full['Cabin']=full['Cabin'].map(lambda x:x[0])
full['Cabin'].value_counts()
```




    U    1014
    C      94
    B      65
    D      46
    E      41
    A      22
    F      21
    G       5
    T       1
    Name: Cabin, dtype: int64




```python
CabinDf=pd.get_dummies(full['Cabin'],prefix='Cabin')
full=pd.concat([full,CabinDf],axis=1)
full=full.drop('Cabin',axis=1)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>
      <th>Royalty</th>
      <th>Cabin_A</th>
      <th>Cabin_B</th>
      <th>Cabin_C</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



### 建立家庭人数和家庭类别：
家庭人数=同代直系亲属数（Parch）+不同代直系亲属数（SibSp）+乘客自己（因为乘客自己也是家庭成员的一个，所以这里加1）

小家庭Family_Single： 家庭人数=1

中等家庭Family_Small: 2<=家庭人数<=4

大家庭Family_Large: 家庭人数>=5


```python
full['familysize']=full['Parch']+full['SibSp']+1
full['family_singel']=np.where(full['familysize']==1,1,0)
full['family_small']=np.where((full['familysize']>=2)&(full['familysize']<=4),1,0)
full['family_large']=np.where(full['familysize']>=5,1,0)
full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>familysize</th>
      <th>family_singel</th>
      <th>family_small</th>
      <th>family_large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>53.1000</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 33 columns</p>
</div>




```python
full.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 33 columns):
    Age              1309 non-null float64
    Fare             1309 non-null float64
    Parch            1309 non-null int64
    PassengerId      1309 non-null int64
    Sex              1309 non-null int64
    SibSp            1309 non-null int64
    Survived         891 non-null float64
    Ticket           1309 non-null object
    Embarked_C       1309 non-null uint8
    Embarked_Q       1309 non-null uint8
    Embarked_S       1309 non-null uint8
    Pclass_1         1309 non-null uint8
    Pclass_2         1309 non-null uint8
    Pclass_3         1309 non-null uint8
    Master           1309 non-null uint8
    Miss             1309 non-null uint8
    Mr               1309 non-null uint8
    Mrs              1309 non-null uint8
    Officer          1309 non-null uint8
    Royalty          1309 non-null uint8
    Cabin_A          1309 non-null uint8
    Cabin_B          1309 non-null uint8
    Cabin_C          1309 non-null uint8
    Cabin_D          1309 non-null uint8
    Cabin_E          1309 non-null uint8
    Cabin_F          1309 non-null uint8
    Cabin_G          1309 non-null uint8
    Cabin_T          1309 non-null uint8
    Cabin_U          1309 non-null uint8
    familysize       1309 non-null int64
    family_singel    1309 non-null int32
    family_small     1309 non-null int32
    family_large     1309 non-null int32
    dtypes: float64(3), int32(3), int64(5), object(1), uint8(21)
    memory usage: 134.3+ KB
    


```python
full.loc[889:898,:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>familysize</th>
      <th>family_singel</th>
      <th>family_small</th>
      <th>family_large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>889</th>
      <td>26.0</td>
      <td>30.0000</td>
      <td>0</td>
      <td>890</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>111369</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>32.0</td>
      <td>7.7500</td>
      <td>0</td>
      <td>891</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>370376</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>891</th>
      <td>34.5</td>
      <td>7.8292</td>
      <td>0</td>
      <td>892</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>330911</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>892</th>
      <td>47.0</td>
      <td>7.0000</td>
      <td>0</td>
      <td>893</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>363272</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>893</th>
      <td>62.0</td>
      <td>9.6875</td>
      <td>0</td>
      <td>894</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>240276</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>894</th>
      <td>27.0</td>
      <td>8.6625</td>
      <td>0</td>
      <td>895</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>315154</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>895</th>
      <td>22.0</td>
      <td>12.2875</td>
      <td>1</td>
      <td>896</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>3101298</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>896</th>
      <td>14.0</td>
      <td>9.2250</td>
      <td>0</td>
      <td>897</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>7538</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>897</th>
      <td>30.0</td>
      <td>7.6292</td>
      <td>0</td>
      <td>898</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>330972</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>898</th>
      <td>26.0</td>
      <td>29.0000</td>
      <td>1</td>
      <td>899</td>
      <td>1</td>
      <td>1</td>
      <td>NaN</td>
      <td>248738</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 33 columns</p>
</div>



## 特征选择和特征降维
通过前面的特征选取，得到32个特征，下面使用相关系数法选取特征


```python
#计算相关性矩阵
corr_df=full.corr()
corr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>PassengerId</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>...</th>
      <th>Cabin_D</th>
      <th>Cabin_E</th>
      <th>Cabin_F</th>
      <th>Cabin_G</th>
      <th>Cabin_T</th>
      <th>Cabin_U</th>
      <th>familysize</th>
      <th>family_singel</th>
      <th>family_small</th>
      <th>family_large</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>0.176326</td>
      <td>-0.130872</td>
      <td>0.025731</td>
      <td>0.057397</td>
      <td>-0.190747</td>
      <td>-0.070323</td>
      <td>0.076179</td>
      <td>-0.012718</td>
      <td>-0.059153</td>
      <td>...</td>
      <td>0.132886</td>
      <td>0.106600</td>
      <td>-0.072644</td>
      <td>-0.085977</td>
      <td>0.032461</td>
      <td>-0.271918</td>
      <td>-0.196996</td>
      <td>0.116675</td>
      <td>-0.038189</td>
      <td>-0.161210</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.176326</td>
      <td>1.000000</td>
      <td>0.216043</td>
      <td>0.034120</td>
      <td>-0.176464</td>
      <td>0.154383</td>
      <td>0.246552</td>
      <td>0.279941</td>
      <td>-0.133808</td>
      <td>-0.161943</td>
      <td>...</td>
      <td>0.070403</td>
      <td>0.071746</td>
      <td>-0.039065</td>
      <td>-0.023580</td>
      <td>0.000847</td>
      <td>-0.513016</td>
      <td>0.219629</td>
      <td>-0.264940</td>
      <td>0.188678</td>
      <td>0.167639</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.130872</td>
      <td>0.216043</td>
      <td>1.000000</td>
      <td>0.008942</td>
      <td>-0.213125</td>
      <td>0.373587</td>
      <td>0.081629</td>
      <td>-0.008635</td>
      <td>-0.100943</td>
      <td>0.071881</td>
      <td>...</td>
      <td>-0.027385</td>
      <td>0.001084</td>
      <td>0.020481</td>
      <td>0.058325</td>
      <td>-0.012304</td>
      <td>-0.036806</td>
      <td>0.792296</td>
      <td>-0.549022</td>
      <td>0.248532</td>
      <td>0.624627</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>0.025731</td>
      <td>0.034120</td>
      <td>0.008942</td>
      <td>1.000000</td>
      <td>0.013406</td>
      <td>-0.055224</td>
      <td>-0.005007</td>
      <td>0.048101</td>
      <td>0.011585</td>
      <td>-0.049836</td>
      <td>...</td>
      <td>0.000549</td>
      <td>-0.008136</td>
      <td>0.000306</td>
      <td>-0.045949</td>
      <td>-0.023049</td>
      <td>0.000208</td>
      <td>-0.031437</td>
      <td>0.028546</td>
      <td>0.002975</td>
      <td>-0.063415</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.057397</td>
      <td>-0.176464</td>
      <td>-0.213125</td>
      <td>0.013406</td>
      <td>1.000000</td>
      <td>-0.109609</td>
      <td>-0.543351</td>
      <td>-0.066564</td>
      <td>-0.088651</td>
      <td>0.115193</td>
      <td>...</td>
      <td>-0.057396</td>
      <td>-0.040340</td>
      <td>-0.006655</td>
      <td>-0.083285</td>
      <td>0.020558</td>
      <td>0.137396</td>
      <td>-0.188583</td>
      <td>0.284537</td>
      <td>-0.255196</td>
      <td>-0.077748</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.190747</td>
      <td>0.154383</td>
      <td>0.373587</td>
      <td>-0.055224</td>
      <td>-0.109609</td>
      <td>1.000000</td>
      <td>-0.035322</td>
      <td>-0.048396</td>
      <td>-0.048678</td>
      <td>0.073709</td>
      <td>...</td>
      <td>-0.015727</td>
      <td>-0.027180</td>
      <td>-0.008619</td>
      <td>0.006015</td>
      <td>-0.013247</td>
      <td>0.009064</td>
      <td>0.861952</td>
      <td>-0.591077</td>
      <td>0.253590</td>
      <td>0.699681</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.070323</td>
      <td>0.246552</td>
      <td>0.081629</td>
      <td>-0.005007</td>
      <td>-0.543351</td>
      <td>-0.035322</td>
      <td>1.000000</td>
      <td>0.168240</td>
      <td>0.003650</td>
      <td>-0.149683</td>
      <td>...</td>
      <td>0.150716</td>
      <td>0.145321</td>
      <td>0.057935</td>
      <td>0.016040</td>
      <td>-0.026456</td>
      <td>-0.316912</td>
      <td>0.016639</td>
      <td>-0.203367</td>
      <td>0.279855</td>
      <td>-0.125147</td>
    </tr>
    <tr>
      <th>Embarked_C</th>
      <td>0.076179</td>
      <td>0.279941</td>
      <td>-0.008635</td>
      <td>0.048101</td>
      <td>-0.066564</td>
      <td>-0.048396</td>
      <td>0.168240</td>
      <td>1.000000</td>
      <td>-0.164166</td>
      <td>-0.778262</td>
      <td>...</td>
      <td>0.107782</td>
      <td>0.027566</td>
      <td>-0.020010</td>
      <td>-0.031566</td>
      <td>-0.014095</td>
      <td>-0.258257</td>
      <td>-0.036553</td>
      <td>-0.107874</td>
      <td>0.159594</td>
      <td>-0.092825</td>
    </tr>
    <tr>
      <th>Embarked_Q</th>
      <td>-0.012718</td>
      <td>-0.133808</td>
      <td>-0.100943</td>
      <td>0.011585</td>
      <td>-0.088651</td>
      <td>-0.048678</td>
      <td>0.003650</td>
      <td>-0.164166</td>
      <td>1.000000</td>
      <td>-0.491656</td>
      <td>...</td>
      <td>-0.061459</td>
      <td>-0.042877</td>
      <td>-0.020282</td>
      <td>-0.019941</td>
      <td>-0.008904</td>
      <td>0.142369</td>
      <td>-0.087190</td>
      <td>0.127214</td>
      <td>-0.122491</td>
      <td>-0.018423</td>
    </tr>
    <tr>
      <th>Embarked_S</th>
      <td>-0.059153</td>
      <td>-0.161943</td>
      <td>0.071881</td>
      <td>-0.049836</td>
      <td>0.115193</td>
      <td>0.073709</td>
      <td>-0.149683</td>
      <td>-0.778262</td>
      <td>-0.491656</td>
      <td>1.000000</td>
      <td>...</td>
      <td>-0.056023</td>
      <td>0.002960</td>
      <td>0.030575</td>
      <td>0.040560</td>
      <td>0.018111</td>
      <td>0.137351</td>
      <td>0.087771</td>
      <td>0.014246</td>
      <td>-0.062909</td>
      <td>0.093671</td>
    </tr>
    <tr>
      <th>Pclass_1</th>
      <td>0.362587</td>
      <td>0.614106</td>
      <td>-0.013033</td>
      <td>0.026495</td>
      <td>-0.107371</td>
      <td>-0.034256</td>
      <td>0.285904</td>
      <td>0.325722</td>
      <td>-0.166101</td>
      <td>-0.181800</td>
      <td>...</td>
      <td>0.275698</td>
      <td>0.242963</td>
      <td>-0.073083</td>
      <td>-0.035441</td>
      <td>0.048310</td>
      <td>-0.776987</td>
      <td>-0.029656</td>
      <td>-0.126551</td>
      <td>0.165965</td>
      <td>-0.067523</td>
    </tr>
    <tr>
      <th>Pclass_2</th>
      <td>-0.014193</td>
      <td>-0.122792</td>
      <td>-0.010057</td>
      <td>0.022714</td>
      <td>-0.028862</td>
      <td>-0.052419</td>
      <td>0.093349</td>
      <td>-0.134675</td>
      <td>-0.121973</td>
      <td>0.196532</td>
      <td>...</td>
      <td>-0.037929</td>
      <td>-0.050210</td>
      <td>0.127371</td>
      <td>-0.032081</td>
      <td>-0.014325</td>
      <td>0.176485</td>
      <td>-0.039976</td>
      <td>-0.035075</td>
      <td>0.097270</td>
      <td>-0.118495</td>
    </tr>
    <tr>
      <th>Pclass_3</th>
      <td>-0.302093</td>
      <td>-0.430696</td>
      <td>0.019521</td>
      <td>-0.041544</td>
      <td>0.116562</td>
      <td>0.072610</td>
      <td>-0.322308</td>
      <td>-0.171430</td>
      <td>0.243706</td>
      <td>-0.003805</td>
      <td>...</td>
      <td>-0.207455</td>
      <td>-0.169063</td>
      <td>-0.041178</td>
      <td>0.056964</td>
      <td>-0.030057</td>
      <td>0.527614</td>
      <td>0.058430</td>
      <td>0.138250</td>
      <td>-0.223338</td>
      <td>0.155560</td>
    </tr>
    <tr>
      <th>Master</th>
      <td>-0.363923</td>
      <td>0.008946</td>
      <td>0.253482</td>
      <td>0.002254</td>
      <td>0.164375</td>
      <td>0.329171</td>
      <td>0.085221</td>
      <td>-0.014172</td>
      <td>-0.009091</td>
      <td>0.018297</td>
      <td>...</td>
      <td>-0.042192</td>
      <td>0.001860</td>
      <td>0.058311</td>
      <td>-0.013690</td>
      <td>-0.006113</td>
      <td>0.041178</td>
      <td>0.355061</td>
      <td>-0.265355</td>
      <td>0.120166</td>
      <td>0.301809</td>
    </tr>
    <tr>
      <th>Miss</th>
      <td>-0.254146</td>
      <td>0.086006</td>
      <td>0.066473</td>
      <td>-0.050027</td>
      <td>-0.672819</td>
      <td>0.077564</td>
      <td>0.332795</td>
      <td>-0.014351</td>
      <td>0.198804</td>
      <td>-0.113886</td>
      <td>...</td>
      <td>-0.012516</td>
      <td>0.008700</td>
      <td>-0.003088</td>
      <td>0.061881</td>
      <td>-0.013832</td>
      <td>-0.004364</td>
      <td>0.087350</td>
      <td>-0.023890</td>
      <td>-0.018085</td>
      <td>0.083422</td>
    </tr>
    <tr>
      <th>Mr</th>
      <td>0.165476</td>
      <td>-0.184520</td>
      <td>-0.304780</td>
      <td>0.014116</td>
      <td>0.870678</td>
      <td>-0.243104</td>
      <td>-0.549199</td>
      <td>-0.065538</td>
      <td>-0.080224</td>
      <td>0.108924</td>
      <td>...</td>
      <td>-0.030261</td>
      <td>-0.032953</td>
      <td>-0.026403</td>
      <td>-0.072514</td>
      <td>0.023611</td>
      <td>0.131807</td>
      <td>-0.326487</td>
      <td>0.386262</td>
      <td>-0.300872</td>
      <td>-0.194207</td>
    </tr>
    <tr>
      <th>Mrs</th>
      <td>0.198091</td>
      <td>0.134059</td>
      <td>0.213491</td>
      <td>0.033299</td>
      <td>-0.571176</td>
      <td>0.061643</td>
      <td>0.344935</td>
      <td>0.098379</td>
      <td>-0.100374</td>
      <td>-0.022950</td>
      <td>...</td>
      <td>0.080393</td>
      <td>0.045538</td>
      <td>0.013376</td>
      <td>0.042547</td>
      <td>-0.011742</td>
      <td>-0.162253</td>
      <td>0.157233</td>
      <td>-0.354649</td>
      <td>0.361247</td>
      <td>0.012893</td>
    </tr>
    <tr>
      <th>Officer</th>
      <td>0.162818</td>
      <td>0.027077</td>
      <td>-0.032631</td>
      <td>0.002231</td>
      <td>0.087288</td>
      <td>-0.013813</td>
      <td>-0.031316</td>
      <td>0.003678</td>
      <td>-0.003212</td>
      <td>-0.001202</td>
      <td>...</td>
      <td>0.006055</td>
      <td>-0.024048</td>
      <td>-0.017076</td>
      <td>-0.008281</td>
      <td>-0.003698</td>
      <td>-0.067030</td>
      <td>-0.026921</td>
      <td>0.013303</td>
      <td>0.003966</td>
      <td>-0.034572</td>
    </tr>
    <tr>
      <th>Royalty</th>
      <td>0.059466</td>
      <td>0.044920</td>
      <td>-0.030197</td>
      <td>0.004400</td>
      <td>-0.020408</td>
      <td>-0.010787</td>
      <td>0.033391</td>
      <td>0.077213</td>
      <td>-0.021853</td>
      <td>-0.054250</td>
      <td>...</td>
      <td>-0.012950</td>
      <td>-0.012202</td>
      <td>-0.008665</td>
      <td>-0.004202</td>
      <td>-0.001876</td>
      <td>-0.071672</td>
      <td>-0.023600</td>
      <td>0.008761</td>
      <td>-0.000073</td>
      <td>-0.017542</td>
    </tr>
    <tr>
      <th>Cabin_A</th>
      <td>0.125177</td>
      <td>0.028783</td>
      <td>-0.030707</td>
      <td>-0.002831</td>
      <td>0.047561</td>
      <td>-0.039808</td>
      <td>0.022287</td>
      <td>0.094914</td>
      <td>-0.042105</td>
      <td>-0.056984</td>
      <td>...</td>
      <td>-0.024952</td>
      <td>-0.023510</td>
      <td>-0.016695</td>
      <td>-0.008096</td>
      <td>-0.003615</td>
      <td>-0.242399</td>
      <td>-0.042967</td>
      <td>0.045227</td>
      <td>-0.029546</td>
      <td>-0.033799</td>
    </tr>
    <tr>
      <th>Cabin_B</th>
      <td>0.113458</td>
      <td>0.408948</td>
      <td>0.073051</td>
      <td>0.015895</td>
      <td>-0.094453</td>
      <td>-0.011569</td>
      <td>0.175095</td>
      <td>0.161595</td>
      <td>-0.073613</td>
      <td>-0.095790</td>
      <td>...</td>
      <td>-0.043624</td>
      <td>-0.041103</td>
      <td>-0.029188</td>
      <td>-0.014154</td>
      <td>-0.006320</td>
      <td>-0.423794</td>
      <td>0.032318</td>
      <td>-0.087912</td>
      <td>0.084268</td>
      <td>0.013470</td>
    </tr>
    <tr>
      <th>Cabin_C</th>
      <td>0.167993</td>
      <td>0.397754</td>
      <td>0.009601</td>
      <td>0.006092</td>
      <td>-0.077473</td>
      <td>0.048616</td>
      <td>0.114652</td>
      <td>0.158043</td>
      <td>-0.059151</td>
      <td>-0.101861</td>
      <td>...</td>
      <td>-0.053083</td>
      <td>-0.050016</td>
      <td>-0.035516</td>
      <td>-0.017224</td>
      <td>-0.007691</td>
      <td>-0.515684</td>
      <td>0.037226</td>
      <td>-0.137498</td>
      <td>0.141925</td>
      <td>0.001362</td>
    </tr>
    <tr>
      <th>Cabin_D</th>
      <td>0.132886</td>
      <td>0.070403</td>
      <td>-0.027385</td>
      <td>0.000549</td>
      <td>-0.057396</td>
      <td>-0.015727</td>
      <td>0.150716</td>
      <td>0.107782</td>
      <td>-0.061459</td>
      <td>-0.056023</td>
      <td>...</td>
      <td>1.000000</td>
      <td>-0.034317</td>
      <td>-0.024369</td>
      <td>-0.011817</td>
      <td>-0.005277</td>
      <td>-0.353822</td>
      <td>-0.025313</td>
      <td>-0.074310</td>
      <td>0.102432</td>
      <td>-0.049336</td>
    </tr>
    <tr>
      <th>Cabin_E</th>
      <td>0.106600</td>
      <td>0.071746</td>
      <td>0.001084</td>
      <td>-0.008136</td>
      <td>-0.040340</td>
      <td>-0.027180</td>
      <td>0.145321</td>
      <td>0.027566</td>
      <td>-0.042877</td>
      <td>0.002960</td>
      <td>...</td>
      <td>-0.034317</td>
      <td>1.000000</td>
      <td>-0.022961</td>
      <td>-0.011135</td>
      <td>-0.004972</td>
      <td>-0.333381</td>
      <td>-0.017285</td>
      <td>-0.042535</td>
      <td>0.068007</td>
      <td>-0.046485</td>
    </tr>
    <tr>
      <th>Cabin_F</th>
      <td>-0.072644</td>
      <td>-0.039065</td>
      <td>0.020481</td>
      <td>0.000306</td>
      <td>-0.006655</td>
      <td>-0.008619</td>
      <td>0.057935</td>
      <td>-0.020010</td>
      <td>-0.020282</td>
      <td>0.030575</td>
      <td>...</td>
      <td>-0.024369</td>
      <td>-0.022961</td>
      <td>1.000000</td>
      <td>-0.007907</td>
      <td>-0.003531</td>
      <td>-0.236733</td>
      <td>0.005525</td>
      <td>0.004055</td>
      <td>0.012756</td>
      <td>-0.033009</td>
    </tr>
    <tr>
      <th>Cabin_G</th>
      <td>-0.085977</td>
      <td>-0.023580</td>
      <td>0.058325</td>
      <td>-0.045949</td>
      <td>-0.083285</td>
      <td>0.006015</td>
      <td>0.016040</td>
      <td>-0.031566</td>
      <td>-0.019941</td>
      <td>0.040560</td>
      <td>...</td>
      <td>-0.011817</td>
      <td>-0.011135</td>
      <td>-0.007907</td>
      <td>1.000000</td>
      <td>-0.001712</td>
      <td>-0.114803</td>
      <td>0.035835</td>
      <td>-0.076397</td>
      <td>0.087471</td>
      <td>-0.016008</td>
    </tr>
    <tr>
      <th>Cabin_T</th>
      <td>0.032461</td>
      <td>0.000847</td>
      <td>-0.012304</td>
      <td>-0.023049</td>
      <td>0.020558</td>
      <td>-0.013247</td>
      <td>-0.026456</td>
      <td>-0.014095</td>
      <td>-0.008904</td>
      <td>0.018111</td>
      <td>...</td>
      <td>-0.005277</td>
      <td>-0.004972</td>
      <td>-0.003531</td>
      <td>-0.001712</td>
      <td>1.000000</td>
      <td>-0.051263</td>
      <td>-0.015438</td>
      <td>0.022411</td>
      <td>-0.019574</td>
      <td>-0.007148</td>
    </tr>
    <tr>
      <th>Cabin_U</th>
      <td>-0.271918</td>
      <td>-0.513016</td>
      <td>-0.036806</td>
      <td>0.000208</td>
      <td>0.137396</td>
      <td>0.009064</td>
      <td>-0.316912</td>
      <td>-0.258257</td>
      <td>0.142369</td>
      <td>0.137351</td>
      <td>...</td>
      <td>-0.353822</td>
      <td>-0.333381</td>
      <td>-0.236733</td>
      <td>-0.114803</td>
      <td>-0.051263</td>
      <td>1.000000</td>
      <td>-0.014155</td>
      <td>0.175812</td>
      <td>-0.211367</td>
      <td>0.056438</td>
    </tr>
    <tr>
      <th>familysize</th>
      <td>-0.196996</td>
      <td>0.219629</td>
      <td>0.792296</td>
      <td>-0.031437</td>
      <td>-0.188583</td>
      <td>0.861952</td>
      <td>0.016639</td>
      <td>-0.036553</td>
      <td>-0.087190</td>
      <td>0.087771</td>
      <td>...</td>
      <td>-0.025313</td>
      <td>-0.017285</td>
      <td>0.005525</td>
      <td>0.035835</td>
      <td>-0.015438</td>
      <td>-0.014155</td>
      <td>1.000000</td>
      <td>-0.688864</td>
      <td>0.302640</td>
      <td>0.801623</td>
    </tr>
    <tr>
      <th>family_singel</th>
      <td>0.116675</td>
      <td>-0.264940</td>
      <td>-0.549022</td>
      <td>0.028546</td>
      <td>0.284537</td>
      <td>-0.591077</td>
      <td>-0.203367</td>
      <td>-0.107874</td>
      <td>0.127214</td>
      <td>0.014246</td>
      <td>...</td>
      <td>-0.074310</td>
      <td>-0.042535</td>
      <td>0.004055</td>
      <td>-0.076397</td>
      <td>0.022411</td>
      <td>0.175812</td>
      <td>-0.688864</td>
      <td>1.000000</td>
      <td>-0.873398</td>
      <td>-0.318944</td>
    </tr>
    <tr>
      <th>family_small</th>
      <td>-0.038189</td>
      <td>0.188678</td>
      <td>0.248532</td>
      <td>0.002975</td>
      <td>-0.255196</td>
      <td>0.253590</td>
      <td>0.279855</td>
      <td>0.159594</td>
      <td>-0.122491</td>
      <td>-0.062909</td>
      <td>...</td>
      <td>0.102432</td>
      <td>0.068007</td>
      <td>0.012756</td>
      <td>0.087471</td>
      <td>-0.019574</td>
      <td>-0.211367</td>
      <td>0.302640</td>
      <td>-0.873398</td>
      <td>1.000000</td>
      <td>-0.183007</td>
    </tr>
    <tr>
      <th>family_large</th>
      <td>-0.161210</td>
      <td>0.167639</td>
      <td>0.624627</td>
      <td>-0.063415</td>
      <td>-0.077748</td>
      <td>0.699681</td>
      <td>-0.125147</td>
      <td>-0.092825</td>
      <td>-0.018423</td>
      <td>0.093671</td>
      <td>...</td>
      <td>-0.049336</td>
      <td>-0.046485</td>
      <td>-0.033009</td>
      <td>-0.016008</td>
      <td>-0.007148</td>
      <td>0.056438</td>
      <td>0.801623</td>
      <td>-0.318944</td>
      <td>-0.183007</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>32 rows × 32 columns</p>
</div>




```python
#提取各特征与生存情况（Survived）的相关系数，并降序排列
corr_df['Survived'].sort_values(ascending=False)
```




    Survived         1.000000
    Mrs              0.344935
    Miss             0.332795
    Pclass_1         0.285904
    family_small     0.279855
    Fare             0.246552
    Cabin_B          0.175095
    Embarked_C       0.168240
    Cabin_D          0.150716
    Cabin_E          0.145321
    Cabin_C          0.114652
    Pclass_2         0.093349
    Master           0.085221
    Parch            0.081629
    Cabin_F          0.057935
    Royalty          0.033391
    Cabin_A          0.022287
    familysize       0.016639
    Cabin_G          0.016040
    Embarked_Q       0.003650
    PassengerId     -0.005007
    Cabin_T         -0.026456
    Officer         -0.031316
    SibSp           -0.035322
    Age             -0.070323
    family_large    -0.125147
    Embarked_S      -0.149683
    family_singel   -0.203367
    Cabin_U         -0.316912
    Pclass_3        -0.322308
    Sex             -0.543351
    Mr              -0.549199
    Name: Survived, dtype: float64



根据各特征与生存情况（Survived）的相关系数大小，选取以下特征进行建模：头衔（前面所在的数据集TitleDf）、客舱等级（PclassDf）、船票价格（Fare）、船舱号（CabinDf）、登船港口（EmbarkedDf）、性别（Sex）、家庭大小及类别（familysize,family_small,family_large,family_singel）


```python
full_x=pd.concat([TitleDf,PclassDf,CabinDf,EmbarkedDf,full['Fare'],full['Sex'],full['familysize'],full['family_small']
                  ,full['family_large'],full['family_singel']],axis=1)
full_x.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Officer</th>
      <th>Royalty</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Pclass_3</th>
      <th>Cabin_A</th>
      <th>...</th>
      <th>Cabin_U</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Fare</th>
      <th>Sex</th>
      <th>familysize</th>
      <th>family_small</th>
      <th>family_large</th>
      <th>family_singel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7.2500</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7.9250</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>53.1000</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8.0500</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



## 构建模型
### 建立训练数据集和测试数据集
根据前面的数据我们知道，train.csv里包含Survived标签，因此用来作为模型训练的数据，并需要将其分为训练数据集和测试数据集，test.csv无Survived标签，用来作为预测数据集


```python
#前891行为原始训练数据，我们将其提取出来
source_x=full_x.loc[0:890,:]#提取特征
source_y=full.loc[0:890,'Survived']#提取标签
#后418行为预测数据
pred_x=full_x.loc[891:,:]
source_x.shape
source_y.shape
pred_x.shape
```




    (891, 27)






    (891,)






    (418, 27)




```python
#建立模型用的训练数据集和测试数据集，按照二八原则分为训练数据和测试数据，其中80%为训练数据
from sklearn.cross_validation import train_test_split
train_x,test_x,train_y,test_y=train_test_split(source_x,source_y,train_size=0.8)
print('训练数据集特征:{0},训练数据集标签:{1}'.format(train_x.shape,train_y.shape))
print('测试数据集特征:{0},测试数据集标签:{1}'.format(test_x.shape,test_y.shape))
```

    训练数据集特征:(712, 27),训练数据集标签:(712,)
    测试数据集特征:(179, 27),测试数据集标签:(179,)
    


```python
#对train_x,test_x进行标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x_std=sc.fit_transform(train_x)
test_x_std=sc.transform(test_x)
```

### 选择算法训练模型
这里我们选择逻辑回归


```python
#第一步：选择算法，并导入相应算发包
from sklearn.linear_model import LogisticRegression
#第二步：创建模型
model=LogisticRegression()
#第三步：训练模型
model.fit(train_x_std,train_y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



### 评估模型


```python
#得出模型正确率
model.score(test_x_std,test_y)
```




    0.8212290502793296



### 方案实施


```python
#使用训练得到的模型对pred_x的生存情况进行预测
pred_x_std=sc.fit_transform(pred_x)
pred_y=model.predict(pred_x_std)
pred_y
```




    array([0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
           0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0.,
           0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 1., 1., 1.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1.,
           0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           1., 1., 1., 1., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 1., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 1.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0.,
           1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0.,
           1., 1., 1., 1., 1., 0., 1., 0., 0., 1.])




```python
pred_df=pd.DataFrame({'PassengerId':test.PassengerId,'Survived':pred_y})
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred_df.shape
```




    (418, 2)




```python
pred_df['Survived']=pred_df['Survived'].astype('int')
pred_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 2 columns):
    PassengerId    418 non-null int64
    Survived       418 non-null int32
    dtypes: int32(1), int64(1)
    memory usage: 5.0 KB
    


```python
#保存结果
pred_df.to_csv(r'E:\python\data\titanic\predict.csv',index=False)
```
