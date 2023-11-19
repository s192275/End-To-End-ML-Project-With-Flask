"""
@author : Serhat KILIÇ
@date : 4.11.2023 / 14.11
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cnf
from itertools import combinations
from sklearn.model_selection import GridSearchCV, cross_validate


def first_edit(data):
    """
        Verisetindeki sütunları değiştirme işlemi yapar. Bunları belli standartlar çerçevesinde icra eder.
        Dili İngilizce formatına uygun dönüştürür.
    Args:
        data : İşlem yapılacak veriseti
    """
    data.columns = [col.upper() for col in data.columns]
    data.columns = data.columns.str.replace(' ', '_')
    data.columns = data.columns.str.replace('İ', 'I')
    data.columns = data.columns.str.replace('Ğ', 'G')
    data.columns = data.columns.str.replace('Ö', 'O')
    data.columns = data.columns.str.replace('Ü', 'U')
    data.columns = data.columns.str.replace('Ş', 'S')
    data.columns = data.columns.str.replace('Ç', 'C')

def analyse_dataset(data):
    """
    Veriseti hakkında analiz yapmayı ve verisetinin şekli, tipi, baş-sonundaki veriler, null veriler
    hakkında bilgi vererek verisetinin sayısal değişkenlerinde quantile verilerinin incelenmesini sağlar.
    """
    print("########## SHAPE ##########")
    print(data.shape)
    print("########## DTYPES ##########")
    print(data.dtypes)
    print("########## HEAD ##########")
    print(data.head())
    print("########## TAIL ##########")
    print(data.tail())
    print("########## ISNA ##########")
    print(data.isna().sum())
    print("########## DESCRIBE ##########")
    print(data.describe())
    
def grab_dataset(data, cat_th = 10, car_th = 20):
    """
    Verisetindeki kategorik gibi görünen nümerik değişkeni nümerik gibi görünen kategorik değişkeni kategorik ve nümerik değişkenleri tespit etmeye yarar.
    Args:
        data (_type_): _description_
        cat_th (int, optional): _description_. Defaults to 10.
        car_th (int, optional): _description_. Defaults to 20.
    Returns:
        cat_cols, num_cols    
    """
    #cat_cols + cat_but_car
    cat_cols = [col for col in data.columns if data[col].dtypes == 'O']
    num_but_cat = [col for col in data.columns if data[col].dtypes != 'O' and 
                   data[col].nunique() < cat_th]
    cat_but_car = [col for col in data.columns if data[col].dtypes == 'O' and 
                   data[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    #num_cols
    num_cols = [col for col in data.columns if data[col].dtypes != 'O']
    num_cols = [col for col in num_cols if col not in num_but_cat]
    
    print(f"Observations: {data.shape[0]}")
    print(f"Variables: {data.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    return cat_cols, num_cols
    
def missing_values_table(data, na_name=False):
    """
    Null satırları tespit eder. Bunları sıralayarak verisetine oranlar ve null değerlerin her bir satırdaki oranını tespit eder.
    Args:
        data (_type_): Satırlarındaki null değerleri tespit edilecek veriseti
        na_name (bool, optional): Null değerleri dönmek istiyorsak True yoksa False olur.

    Returns:
        na_columns : Opsiyonel
    """
    na_columns = [col for col in data.columns if data[col].isnull().sum() > 0]
    n_miss = data[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (data[na_columns].isnull().sum() / data.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
    
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
    Verisetindeki nümerik değişkenler üzerinden alt ve üst sınırlar belirler(interquantile range) 
    ve onların arasındaki değerleri aykırı olarak tespit eder.
    
    Args:
        dataframe (_type_): Aykırı değerlerin tepit edileceği veriseti
        col_name (_type_): Verisetindeki nümerik satır.
        q1 (float, optional): _description_. Defaults to 0.05.
        q3 (float, optional): _description_. Defaults to 0.95.

    Returns:
        low_limit : Alt sınır
        up_limit : Üst sınır
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
    Yukarıdaki fonksiyondan gelen alt ve üst limitlere göre aykırı değişken var mı yok mu onu tespit eder.
    Args:
        dataframe (_type_): Aykırı değişken analizi yapılacak veriseti
        col_name (_type_): Nümerik satır

    Returns:
        True : Aykırı değişken varsa
        False : Aykırı değişken yoksa
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    """
    Yukarıdaki fonksiyonda True dönen değişkenler için aykırı değerleri gösterir.

    Args:
        dataframe (_type_): Aykırı değerleri gösterilecek veriseti
        col_name (_type_): Nümerik satır
        index (bool, optional): Default olarak False tur. Eğer indexleri dönmek istiyorsak True olur.

    Returns:
        outlier_index : Opsiyonel
    """
    low, up = outlier_thresholds(dataframe, col_name,q1=0.01,q3 = 0.99)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def replace_outliers(dataframe,col_name):
    """
    Yukarıdaki fonksiyonda True dönen değişkenler için aykırı değerleri baskılar.

    Args:
        dataframe (_type_): Aykırı değerleri bastırılacak veriseti
        col_name (_type_): Nümerik satır
    """
    down, up = outlier_thresholds(dataframe,col_name)
    if dataframe.loc[(dataframe[col_name] > up) | (dataframe[col_name] < down)].any(None):
        dataframe.loc[(dataframe[col_name] < down) , col_name] = down
        dataframe.loc[(dataframe[col_name] > up) , col_name] = up
        return dataframe    
    
def num_summary(dataframe, numerical_col, plot=False):
    """
        Numerik kolonlar input olarak verilmelidir.
        Sadece ekrana cikti veren herhangi bir degeri return etmeyen bir fonksiyondur.
        For dongusuyle calistiginda grafiklerde bozulma olmamaktadir.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    
def numcols_target_corr(dataframe, num_cols,target = cnf.target):
    """
    Bir hedef seçerek nümerik satırların onunla olan ilişkisi görselleştirilir.
    Args:
        dataframe (_type_): Veriseti
        num_cols (_type_): Nümerik satırlar
        target (_type_, optional): Bizden tahmin edilmesi istenilen değişken
    """
    numvar_combinations = list(combinations(num_cols, 2))
    
    for item in numvar_combinations:
        
        plt.subplots(figsize=(8,4))
        sns.scatterplot(x=dataframe[item[0]], 
                        y=dataframe[item[1]],
                        hue=dataframe[target],
                        palette=cnf.bright_palette
                       ).set_title(f'{item[0]}   &   {item[1]}')
        plt.grid(True)
        plt.show()            
            
def numeric_variables_boxplot(df, num_cols, target=None, alert=False):
    """
    Hedef değişken ile nümerik değişkenlerin arasındaki ilişki incelenerek box plot çizer.
    
    Args:
        df (_type_): Veriseti
        num_cols (_type_): Nümerik satırlar
        target (_type_, optional): Hedef değişken
        alert (bool, optional): Opsiyonel
    """
    
    if alert == True:
        palette = cnf.alert_palette
    else:
        palette = cnf.bright_palette
        
    if target == None:
        
        fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4, figsize=(7,3))

        for col, ax, i in zip(num_cols, [ax1,ax2,ax3,ax4], range(4)):
            sns.boxplot(df[col], 
                        color=palette[i], 
                        ax=ax
                        ).set_title(col)
            
        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xticklabels([])
    else:
        for col in num_cols:
            plt.subplots(figsize=(7,3))
            sns.boxplot(x=df[target], 
                                y=df[col],
                                hue=df[target],
                                dodge=False, 
                                fliersize=3,
                                linewidth=0.7,
                                palette=palette)
            plt.title(col)
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks(rotation=45)
            plt.legend('',frameon=False)

    plt.tight_layout()
    plt.show()
    
def plot_categorical_data(dataframe, x, hue, title='', label_angle=0):
    """
    Kategorik veri görselleştirmesi için alt grafikleri çizen bir fonksiyon. 
    """
    # Alt grafikleri yan yana düzenleme
    fig, ax = plt.subplots(1, figsize=(8, 3))

    # Grafik 1
    sns.countplot(data=dataframe, x=x, hue=hue, ax=ax, palette='husl')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    ax.legend(prop={'size': 10})

    # Grafikleri göster
    plt.tight_layout()
    plt.xticks(rotation=label_angle)
    plt.show(block=True)
    
def sample_sub(ypred):
    """
    Kaggle için tahmini .csv uzantılı dosyaya çeviren fonksiyon
    Args:
        ypred (_type_): Model tahmini
    """
    sample = pd.read_csv("sample_submission.csv")

    submission = pd.DataFrame({"PassengerId": sample["PassengerId"],
                             "Transported": ypred})
    submission["Transported"] = submission["Transported"].astype(bool)                          
    submission.to_csv("tekrar23.csv",index=False)
  
def plot_avg_numvars_by_target(dataframe, num_cols,agg='mean', round_ndigits=1, alert=False):
    
    if alert == True:
        palette = cnf.alert_palette
    else:
        palette = cnf.sequential_palette
        
    for col in num_cols:
        
        if agg == 'max':
            col_grouped = dataframe.groupby(cnf.target)[col].max().reset_index().sort_values(ascending=False,by=col)
        elif agg == 'min':
            col_grouped = dataframe.groupby(cnf.target)[col].min().reset_index().sort_values(ascending=True,by=col)
        elif agg == 'sum':
            col_grouped = dataframe.groupby(cnf.target)[col].sum().reset_index().sort_values(ascending=False,by=col)
        elif agg == 'std':
            col_grouped = dataframe.groupby(cnf.target)[col].std().reset_index().sort_values(ascending=False,by=col)
        else:
            col_grouped = dataframe.groupby(cnf.target)[col].mean().reset_index().sort_values(ascending=False,by=col)
        
        plt.subplots(figsize=(6,3))
        ax = sns.barplot(x=col_grouped[cnf.target], 
                     y=col_grouped[col], 
                     width=0.8,
                     palette=palette,
                     errorbar=None)
        ax.set_yticklabels([])  
    
        for p in ax.patches:
            ax.text(p.get_x() + p.get_width() / 2., 
                    p.get_height(), 
                    round(p.get_height(),ndigits=round_ndigits), 
                    fontsize=10, color='black', ha='center', va='top')
            
        plt.xlabel('')  
        #plt.ylabel('')
        plt.xticks(rotation=45)
        plt.title(f'{agg} {col} - {cnf.target}')
        plt.show()   
        
        
def correlation_matrix(df, cols):
    """
    Korelasyon matrisi çizen fonksiyon
    Args:
        df (_type_): Veriseti
        cols (_type_): Sütunlar
        """
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    """
    Grid search ile daha önce belirlenen parametreler ile hiperparametre optimizasyonu yapılıyor.
    """
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in cnf.classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models        
