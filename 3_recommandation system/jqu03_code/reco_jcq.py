from collections import defaultdict
from collections import OrderedDict
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import operator
import matplotlib.pyplot as plt
import warnings; warnings.simplefilter('ignore')
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import GridSearchCV

# from surprise doc:
#   get the top-N recommendations for each user
#   example from surprise doc page
def get_top_n(predictions, n):
    '''
    Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# # # # # # # # # # # #
# Part 1: load the data,
# train a learning model
# using svd() with k and lambda,
# make prediction export to the a new file
df = pd.read_csv('u.data', delim_whitespace=True, header=None, names=["user", "item", "rating", "timestamp"])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# from surprise doc:
#   pick a set of parameter
#   by using GridSearchCV

param_grid = {'n_epochs':[50],'biased': [False], 'reg_all': [0.11, 0.12, 0.13, 0.14], 'lr_all': [0.01],
              'n_factors': [10, 20, 25, 30]}
# param_grid = {'n_epochs':[50],'biased': [False], 'reg_all': [0.11], 'lr_all': [0.01],
#               'n_factors': [25, 40, 50]}
# param_grid_step3 = {'n_epochs':[20],'biased': [False], 'reg_all': [0.11, 0.12], 'lr_all': [0.05,0.01],
#               'n_factors': [20, 25, 30]}
# param_grid_step2 = {'n_epochs':[50],'biased': [False], 'reg_all': [0.11, 0.12, 0.13, 0.14], 'lr_all': [0.01],
#               'n_factors': [10, 20, 25, 30]}
# param_grid_step1 = {'biased': [False], 'reg_all': [0.005,0.008, 0.01, 0.03,0.07,0.1, 0.15], 'lr_all': [0.005],
#               'n_factors': [5, 8,10, 15, 20]}
# param_grid_step0 = {'biased':[False], 'reg_all': [0.01, 0.03, 0.05, 0.1, 0.2], 'lr_all': [0.005, 0.01, 0.02, 0.05],
#               'n_factors': [8, 12, 15, 35, 50]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=5)
gs.fit(data)

# print(pd.DataFrame.from_dict(gs.cv_results))

# find the best RMSE score
print("best rmse is: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print("best setting is: ", gs.best_params['rmse'])

# find and store the best set
best_set = gs.best_params['rmse']
biased, reg_all, n_factors, lr_all, n_epochs = best_set["biased"], best_set["reg_all"], best_set["n_factors"], best_set["lr_all"], best_set['n_epochs']

# print("the best set is: ",biased, reg_all, n_factors )

# load the data and train a svd model using best_set{}
df = pd.read_csv('u.data', delim_whitespace=True, header=None, names=["user", "item", "rating", "timestamp"])
reader = Reader(rating_scale=(1, 5))
data_set = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
svd_algo = SVD(biased=biased, reg_all=reg_all, n_factors=n_factors, lr_all=lr_all, n_epochs=n_epochs)
trainset = data.build_full_trainset()

svd_algo.fit(trainset)

# save the U matrix for section 3
# pu user vectors: shape of n_users * n_factors
# qi item vectors: shape of n_items * n_factors
U_vectors, V_vectors = svd_algo.pu, svd_algo.qi
U_vectors, V_vectors = U_vectors.transpose(), V_vectors.transpose()

# predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = svd_algo.test(testset)

# store the top 5 predictions for each user
#   by calling the method get_top_n
top_n = get_top_n(predictions, n=5)
#   then sort the dict based on keys (uid)
top_n = OrderedDict(sorted(top_n.items(), key=lambda t: t[0]))

# store the recommendation as ['user_id', 'rec1', 'rec2', 'rec3', 'rec4', 'rec5']
new_df = pd.DataFrame(columns=['user_id', 'rec1', 'rec2', 'rec3', 'rec4', 'rec5'])
for uid, user_ratings in top_n.items():
    new_df.loc[uid-1] = [uid] + [iid for (iid, _) in user_ratings]

# check the dataframe by print(new_df)
# and export the recommendation data to a file named 'jcq03_result.data'
export_name = 'jcq03_result.data'
new_df.to_csv(export_name, index = False,sep=',', encoding='utf-8')


# # # # # # # # # # # #
# Part 2:
# calculate and
# plot the largest correlation coefficients
# corresponding to Uik which is a column of data in U or V

# load user_info and item_info
user_info_data = pd.read_csv('u.user', sep="|", header=None)
item_info_data = pd.read_csv('u.item', sep="|", header=None, encoding = "ISO-8859-1")

# user_info_data:
#   second column -> ages
#   third column  -> gender ('M' & 'F')
#       Gender remapping: 0 for Female and 1 for Male
ages=user_info_data.iloc[:,1]
ages.astype('int64')
ages = ages.values
gender = user_info_data.iloc[:,2].values
gender[gender == 'M'] = 1
gender[gender == 'F'] = 0
gender = np.int_(gender)

# item_info_data:
#   second column -> title (release year)
#   select the release year and put it into an numpy int array
#   by removing NON-digit chars and append each year_info to [] list
release = item_info_data.iloc[:,1]
release_years = []
i = 1
for each_line in release:
    # get rid of the items that didnt show up in the rating_data file
    if i in [1235, 1310, 1461, 1494, 1580, 1583, 1618, 1653, 1654, 1671]:
        pass
    else:
        each_line = re.sub("\D", "", each_line)
        each_line = each_line[-4:]
        release_years.append(each_line)
    i += 1
release_years = np.int_(release_years)

# find out the largest correlation coefficient for ages, genders and release_year
a, b, c, ka, kb, kc = 0, 0, 0, 0, 0, 0
for i in range(n_factors):
    temp = np.corrcoef(U_vectors[i], ages )[0, 1]
    # print('corrcoef of ages against big U: k = {} is equal to {} '.format(i, temp))
    if abs(temp) > a:
        U_vector_age = U_vectors[i]
        a, ka = abs(temp), i+1
for i in range(n_factors):
    temp = np.corrcoef(U_vectors[i], gender )[0, 1]
    if abs(temp) > b:
        U_vector_gender = U_vectors[i]
        b, kb = abs(temp), i+1

    # print('corrcoef of gender against big U: k = {} is equal to {} '.format(i, np.corrcoef(U_vectors[i], gender )[0, 1]))
for i in range(n_factors):
    temp = np.corrcoef(V_vectors[i], release_years )[0, 1]
    if abs(temp) > c:
        V_vector_releaseYear = V_vectors[i]
        c, kc = abs(temp), i+1
    # print('corrcoef of release_years  against big V: k = {} is equal to {} '.format(i, np.corrcoef(V_vectors[i], release_years )[0, 1]))
print("corrcoef of U vector vs Age \t", a," \tat k = \t", ka)
print("corrcoef of U vector vs Gender \t", b," at k = \t", kb)
print("V vector vs Release Year \t", c," \tat k = \t", kc)

plt.figure(1)
plt.title('U vector k={} vs Ages'.format(ka))
plt.plot(ages, U_vector_age, '+', color='r')
plt.savefig('jqu03_u_ages.png')
plt.figure(2)
plt.title('U vector k={} vs Genders'.format(kb))
plt.plot(gender, U_vector_gender, '*', color='g')
plt.savefig('jqu03_u_gender.png')
plt.figure(3)
plt.title('V vector k={} vs Release Year'.format(kc))
plt.plot(release_years, V_vector_releaseYear, '.', color='b')
plt.savefig('jqu03_v_yr.png')
# plt.show()
