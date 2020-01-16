import numpy as np
import string
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# read csv
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
pd.set_option('mode.chained_assignment', None)

# data feature
# create function that takes ticket feature and returns list of ticket_types
def ticket_sep(data_ticket):
    ticket_type = []
    for i in range(len(data_ticket)):

        ticket =data_ticket.iloc[i]

        for c in string.punctuation:
            ticket = ticket.replace(c,"")
            splited_ticket = ticket.split(" ")
        if len(splited_ticket) == 1:
            ticket_type.append('NO')
        else:
            ticket_type.append(splited_ticket[0])
    return ticket_type


# create function that takes cabin from dataset and extracts cabin type for each cabin that is not missing.
# If cabin is missing, leaves missing value:
def cabin_sep(data_cabin):
    cabin_type = []

    for i in range(len(data_cabin)):

        if data_cabin.isnull()[i] == True:
            cabin_type.append('NaN')
        else:
            cabin = data_cabin[i]
            cabin_type.append(cabin[:1])

    return cabin_type


# Create function that take name and separates it into title, family name and deletes all puntuation from name column:
def name_sep(data):
    families = []
    titles = []
    new_name = []
    # for each row in dataset:
    for i in range(len(data)):
        name = data.iloc[i]
        # extract name inside brakets into name_bracket:
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name

        family = name_no_bracket.split(",")[0]
        title = name_no_bracket.split(",")[1].strip().split(" ")[0]

        # remove punctuations accept brackets:
        for c in string.punctuation:
            name = name.replace(c, "").strip()
            family = family.replace(c, "").strip()
            title = title.replace(c, "").strip()

        families.append(family)
        titles.append(title)
        new_name.append(name)

    return families, titles, new_name


# ticket_type box
train['Ticket_type'] = ticket_sep(train.Ticket)
test['Ticket_type'] = ticket_sep(test.Ticket)

for t in train['Ticket_type'].unique():
    if len(train[train['Ticket_type'] == t]) < 15:
        train.loc[train.Ticket_type == t, 'Ticket_type'] = 'Other_T'
train['Ticket_type'] = np.where(train['Ticket_type'] == 'SOTONOQ', 'A5', train['Ticket_type'])
for t in test['Ticket_type'].unique():
    if len(test[test['Ticket_type'] == t]) < 7:
        test.loc[test.Ticket_type == t, 'Ticket_type'] = 'Other_T'
test['Ticket_type'] = np.where(test['Ticket_type'] == 'SCPARIS', 'A5', test['Ticket_type'])
test['Ticket_type'] = np.where(test['Ticket_type'] == 'SOTONOQ', 'A5', test['Ticket_type'])

# Cabin_type box
train['Cabin_type'] = cabin_sep(train.Cabin)
test['Cabin_type'] = cabin_sep(test.Cabin)
for t in train['Cabin_type'].unique():
    if len(train[train['Cabin_type'] == t]) <= 15:
        train.loc[train.Cabin_type == t, 'Cabin_type'] = 'OTHER_C'
train['Has_cabin'] = np.where(train['Cabin'].isna(), 0, 1)
for t in test['Cabin_type'].unique():
    if len(test[test['Cabin_type'] == t]) <= 8:
        test.loc[test.Cabin_type == t, 'Cabin_type'] = 'OTHER_C'
test['Has_cabin'] = np.where(test['Cabin'].isna(), 0, 1)

# Name & Title box
train['Family'], train['Title'], train['Name'] = name_sep(train.Name)
test['Family'], test['Title'], test['Name'] = name_sep(test.Name)
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
train.Title = train.Title.map(normalized_titles)
test.Title = test.Title.map(normalized_titles)

# create a list with all overlapping families
overlap = [x for x in train.Family.unique() if x in test.Family.unique()]
train['Family_size'] = train['SibSp'] + train['Parch'] + 1
test['Family_size'] = test['SibSp'] + test['Parch'] + 1
all = pd.concat([train, test], sort=True).reset_index(drop=True)
rate_family = all.groupby('Family')['Survived', 'Family', 'Family_size'].median()

# if family size is more than 1 and family name is in overlap list
overlap_family = {}

for i in range(len(rate_family)):
    if rate_family.index[i] in overlap and rate_family.iloc[i, 1] > 1:
        overlap_family[rate_family.index[i]] = rate_family.iloc[i, 0]

# train data
mean_survival_rate = np.mean(train.Survived)
family_survival_rate = []
family_survival_rate_NA = []

for i in range(len(train)):
    if train.Family[i] in overlap_family:
        family_survival_rate.append(overlap_family[train.Family[i]])
        family_survival_rate_NA.append(1)
    else:
        family_survival_rate.append(mean_survival_rate)
        family_survival_rate_NA.append(0)

train['Family_survival_rate'] = family_survival_rate
train['Family_survival_rate_NA'] = family_survival_rate_NA
# test data
family_survival_rate = []
family_survival_rate_NA = []

for i in range(len(test)):
    if test.Family[i] in overlap_family:
        family_survival_rate.append(overlap_family[test.Family[i]])
        family_survival_rate_NA.append(1)
    else:
        family_survival_rate.append(mean_survival_rate)
        family_survival_rate_NA.append(0)

test['Family_survival_rate'] = family_survival_rate
test['Family_survival_rate_NA'] = family_survival_rate_NA

family_map = {2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
train['Family_size_grouped'] = train['Family_size'].map(family_map)
test['Family_size_grouped'] = test['Family_size'].map(family_map)
train['isAlone'] = np.where(train['Family_size'] == 1, 1, 0)
test['isAlone'] = np.where(test['Family_size'] == 1, 1, 0)

# fill na
train['Embarked'] = train['Embarked'].fillna('S')
train['Fare'] = np.where(np.logical_and(train['Fare'].isna(), train['Pclass'] == 1), 84.155, train['Fare'])
train['Fare'] = np.where(np.logical_and(train['Fare'].isna(), train['Pclass'] == 2), 20.662, train['Fare'])
train['Fare'] = np.where(np.logical_and(train['Fare'].isna(), train['Pclass'] == 3), 13.676, train['Fare'])

test['Embarked'] = test['Embarked'].fillna('S')
test['Fare'] = np.where(np.logical_and(test['Fare'].isna(), test['Pclass'] == 1), 84.155, test['Fare'])
test['Fare'] = np.where(np.logical_and(test['Fare'].isna(), test['Pclass'] == 2), 20.662, test['Fare'])
test['Fare'] = np.where(np.logical_and(test['Fare'].isna(), test['Pclass'] == 3), 13.676, test['Fare'])

# age feature for null column
train['Age'] = np.where(np.logical_and(train['Age'].isna(), train['Title'] == 'Master'), 5, train['Age'])
train['Age'] = np.where(np.logical_and(train['Age'].isna(), train['Title'] == 'Miss'), 22, train['Age'])
train['Age'] = np.where(np.logical_and(train['Age'].isna(), train['Title'] == 'Mrs'), 36, train['Age'])
train['Age'] = np.where(np.logical_and(train['Age'].isna(), train['Title'] == 'Mr'), 33, train['Age'])
train['Age'] = np.where(np.logical_and(train['Age'].isna(), train['Title'] == 'OTHER'), 46, train['Age'])

test['Age'] = np.where(np.logical_and(test['Age'].isna(), test['Title'] == 'Master'), 5, test['Age'])
test['Age'] = np.where(np.logical_and(test['Age'].isna(), test['Title'] == 'Miss'), 22, test['Age'])
test['Age'] = np.where(np.logical_and(test['Age'].isna(), test['Title'] == 'Mrs'), 36, test['Age'])
test['Age'] = np.where(np.logical_and(test['Age'].isna(), test['Title'] == 'Mr'), 33, test['Age'])
test['Age'] = np.where(np.logical_and(test['Age'].isna(), test['Title'] == 'OTHER'), 46, test['Age'])
# print(train.groupby('Title')['Age'].mean())
# print(train.Age.isna().any())
# print(pd.crosstab(train.Ticket_type, train.Pclass, margins=True))

# mapping data to numeric
sex_dummy = pd.get_dummies(train['Sex'])
# embarked_dummy = pd.get_dummies(train['Embarked'], prefix='Embarked')
title_dummy = pd.get_dummies(train['Title'], prefix='Title')
pclass_dummy = pd.get_dummies(train['Pclass'], prefix='Pclass')
fsg_dummy = pd.get_dummies(train['Family_size_grouped'], prefix='FSG')
ticket_dummy = pd.get_dummies(train['Ticket_type'], prefix='Ticket')
cabin_dummy = pd.get_dummies(train['Cabin_type'], prefix='Cabin')
train = pd.concat([train, sex_dummy, title_dummy, pclass_dummy, fsg_dummy,
                   ticket_dummy, cabin_dummy], axis=1)

sex_dummy = pd.get_dummies(test['Sex'])
# embarked_dummy = pd.get_dummies(test['Embarked'], prefix='Embarked')
title_dummy = pd.get_dummies(test['Title'], prefix='Title')
pclass_dummy = pd.get_dummies(test['Pclass'], prefix='Pclass')
fsg_dummy = pd.get_dummies(test['Family_size_grouped'], prefix='FSG')
ticket_dummy = pd.get_dummies(test['Ticket_type'], prefix='Ticket')
cabin_dummy = pd.get_dummies(test['Cabin_type'], prefix='Cabin')
test = pd.concat([test, sex_dummy, title_dummy, pclass_dummy, fsg_dummy,
                   ticket_dummy, cabin_dummy], axis=1)

# fare category
bin = [-1, 7.91, 14.454, 26.55, 31, 55.5, 83.475, np.inf]
name = ['Very_Low', 'Low', 'Mid', 'Mid_High', 'High', 'VIP', 'VVIP']
train['Fare'] = pd.cut(train['Fare'], bin, labels=name)
dummy = pd.get_dummies((train['Fare']))
train = pd.concat([train, dummy], axis=1)

test['Fare'] = pd.cut(test['Fare'], bin, labels=name)
dummy = pd.get_dummies((test['Fare']))
test = pd.concat([test, dummy], axis=1)

# age category
bins = [-1, 16, 25, 30, 48, 65, np.inf]
names = ['Child', 'Young', 'Young_Adult', 'Adult', 'Parent', 'Old']
train['Age'] = pd.cut(train['Age'], bins, labels=names)
dummy = pd.get_dummies((train['Age']))
train = pd.concat([train, dummy], axis=1)

test['Age'] = pd.cut(test['Age'], bins, labels=names)
dummy = pd.get_dummies((test['Age']))
test = pd.concat([test, dummy], axis=1)

# polynomial feature for family size
'''
poly = PolynomialFeatures(degree=2, interaction_only=True)
family = train[['SibSp', 'Parch', 'Family_size']]
family = poly.fit_transform(family)
index = [i for i, _ in enumerate(family)]
columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11']
family = pd.DataFrame(family, index, columns)
train = pd.concat([train, family], axis=1)
'''

# remove not relevant columns
drop_element = ['Name', 'Sex', 'Age', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', 'Pclass',
                'Family_size_grouped', 'Ticket_type', 'Family', 'Cabin_type', 'Cabin_NaN']
train = train.drop(drop_element, axis=1)
test = test.drop(drop_element, axis=1)

colormap = plt.cm.viridis
plt.figure(figsize=(12, 12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()

train.to_csv('train_final.csv')
test.to_csv('test_final.csv')
