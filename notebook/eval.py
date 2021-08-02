
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
#from datetime import timedelta
#from functools import reduce
#from tqdm import tqdm
import lightgbm as lgbm
import os
import gc
import joblib
#import shap
#from shapash.explainer.smart_explainer import SmartExplainer
#import missingno as msno
from sklearn.preprocessing import OrdinalEncoder
#,LabelEncoder,OneHotEncoder

#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sns

#from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error,mean_absolute_error
#from sklearn.metrics import make_scorer
#from sklearn.model_selection import cross_val_score,cross_validate

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
#np.set_printoptions(threshold=1000)
pd.options.display.max_info_columns=1000

BASE_DIR = Path('../data')
TRAIN_DIR = Path('../data')

from typing import Optional, Tuple

class Environment:
    def __init__(self,
                 data_dir: str,
                 eval_start_day: int,
                 eval_end_day: Optional[int],
                 use_updated: bool,
                 multiple_days_per_iter: bool):
        warnings.warn('this is mock module for mlb')

        postfix = '_updated' if use_updated else ''
        
        # recommend to replace this with pickle, feather etc to speedup preparing data
        df_train = pd.read_csv(os.path.join(data_dir, f'train{postfix}.csv'))

        players = pd.read_csv(os.path.join(data_dir, 'players.csv'))

        self.players = players[players['playerForTestSetAndFuturePreds'] == True]['playerId'].astype(str)
        if eval_end_day is not None:
            self.df_train = df_train.set_index('date').loc[eval_start_day:eval_end_day]
        else:
            self.df_train = df_train.set_index('date').loc[eval_start_day:]
        self.date = self.df_train.index.values
        self.n_rows = len(self.df_train)
        self.multiple_days_per_iter = multiple_days_per_iter

        assert self.n_rows > 0, 'no data to emulate'

    def predict(self, df: pd.DataFrame) -> float:
        # if you want to emulate public LB, store your prediction here and calculate MAE
        pass

    def iter_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.multiple_days_per_iter:
            for i in range(self.n_rows // 2):
                date1 = self.date[2 * i]
                date2 = self.date[2 * i + 1]
                sample_sub1 = self._make_sample_sub(date1)
                sample_sub2 = self._make_sample_sub(date2)
                sample_sub = pd.concat([sample_sub1, sample_sub2]).reset_index(drop=True)
                df = self.df_train.loc[date1:date2]

                yield df, sample_sub.set_index('date')
        else:
            for i in range(self.n_rows):
                date = self.date[i]
                sample_sub = self._make_sample_sub(date)
                df = self.df_train.loc[date:date]

                yield df, sample_sub.set_index('date')

    def _make_sample_sub(self, date: int) -> pd.DataFrame:
        next_day = (pd.to_datetime(date, format='%Y%m%d') + pd.to_timedelta(1, 'd')).strftime('%Y%m%d')
        sample_sub = pd.DataFrame()
        sample_sub['date_playerId'] = next_day + '_' + self.players
        sample_sub['target1'] = 0
        sample_sub['target2'] = 0
        sample_sub['target3'] = 0
        sample_sub['target4'] = 0
        sample_sub['date'] = date
        return sample_sub


class MLBEmulator:
    def __init__(self,
                 data_dir: str = '../data',
                 eval_start_day: int = 20210401,
                 eval_end_day: Optional[int] = 20210430,
                 use_updated: bool = True,
                 multiple_days_per_iter: bool = False):
        self.data_dir = data_dir
        self.eval_start_day = eval_start_day
        self.eval_end_day = eval_end_day
        self.use_updated = use_updated
        self.multiple_days_per_iter = multiple_days_per_iter

    def make_env(self) -> Environment:
        return Environment(self.data_dir,
                           self.eval_start_day,
                           self.eval_end_day,
                           self.use_updated,
                           self.multiple_days_per_iter)

# split data
col_rosters = ['playerId','date','teamId','status']
col_awards = ['playerId','date','awardPlayerTeamId','awardName']
col_twP = ['playerId','date','numberOfFollowers']
col_twT = ['teamId','date','numberOfFollowersT']
col_stand = [
    'date','teamId','divisionId','divisionRank','leagueRank',
    'leagueGamesBack','sportGamesBack','divisionGamesBack','wins',
    'losses','pct','runsAllowed','runsScored','divisionChamp',
    'divisionLeader','eliminationNumber','homeWins','homeLosses',
    'awayWins','awayLosses','lastTenWins','lastTenLosses',
    'extraInningWins','extraInningLosses','oneRunWins',
    'oneRunLosses','dayWins','dayLosses','nightWins',
    'nightLosses','grassWins','grassLosses','turfWins',
    'turfLosses','divWins','divLosses'
]
col_tran = ['playerId','date','typeCode']
col_scoreP = [
    'playerId','date','gamePk','home','teamId_score',
    'jerseyNum','battingOrder','gamesPlayedBatting','flyOuts',
    'groundOuts','runsScored','doubles','triples','homeRuns',
    'strikeOuts','baseOnBalls','intentionalWalks','hits','hitByPitch',
    'atBats','caughtStealing','stolenBases','groundIntoDoublePlay',
    'groundIntoTriplePlay','plateAppearances','totalBases','rbi',
    'leftOnBase','sacBunts','sacFlies','catchersInterference',
    'pickoffs','gamesPlayedPitching','gamesStartedPitching',
    'completeGamesPitching','shutoutsPitching','winsPitching',
    'lossesPitching','flyOutsPitching','airOutsPitching',
    'groundOutsPitching','runsPitching','doublesPitching',
    'triplesPitching','homeRunsPitching','strikeOutsPitching',
    'baseOnBallsPitching','intentionalWalksPitching','hitsPitching',
    'hitByPitchPitching','atBatsPitching','caughtStealingPitching',
    'stolenBasesPitching','inningsPitched','saveOpportunities',
    'earnedRuns','battersFaced','outsPitching','pitchesThrown',
    'balls','strikes','hitBatsmen','balks','wildPitches',
    'pickoffsPitching','rbiPitching','gamesFinishedPitching',
    'inheritedRunners','inheritedRunnersScored','catchersInterferencePitching',
    'sacBuntsPitching','sacFliesPitching','saves','holds','blownSaves',
    'assists','putOuts','errors','chances'
]
col_game = [
    'date','gamePk','detailedGameState','isTie','gameNumber',
    'doubleHeader','dayNight','scheduledInnings','gamesInSeries',
    'seriesDescription','homeId','homeWins','homeLosses','homeWinPct',
    'homeWinner','awayId','awayWins','awayLosses','awayWinPct',
    'awayWinner'
]

col_label = ['target1','target2','target3','target4']
col_sub = ['date','date_playerId','target1','target2','target3','target4']





def decodeFunc(json_str,date)->pd.DataFrame:
    if json_str!=json_str:
        out = pd.DataFrame()
    else: 
        out = pd.read_json(json_str)
        out['date'] = date
    return out
def splitdata(sub_df,test_df,col_rosters,col_awards,col_twP,col_twT,col_stand,col_tran,col_scoreP,col_game):
       
    date = test_df.iloc[0,0]
    
    #====== Games
    games = decodeFunc(test_df.loc[0,'games'],date)
    if games.empty:
            games['gamePk'] = 0
            games['date'] = sub_df['date']
            col_game2 = [x for x in col_game if x not in ['gamePk','date']]
            for col in col_game2:
                games[col] = np.nan
    else:
        games = games[col_game]
        games = games[games.detailedGameState!='Postponed']
        games = games[~games['gamePk'].duplicated(keep='last')]
    
    #====== Rosters
    rosters = decodeFunc(test_df.loc[0,'rosters'],date)
    if rosters.empty:
        rosters['playerId'] = sub_df['playerId']
        rosters['date'] = sub_df['date']
        rosters['date_playerId'] = sub_df['date_playerId']
        rosters['teamId'] = 0
        rosters['status'] = 'unknown'
    else:
        rosters = rosters[col_rosters]
        rosters['teamId'].fillna(0,inplace=True)
        rosters['status'].fillna('unknown',inplace=True)
        rosters['date_playerId'] = rosters['date'].astype(str)+'_'+rosters['playerId'].astype(str)
        rosters = rosters[~rosters['date_playerId'].duplicated(keep='last')]

    #====== Player Scores
    scores = decodeFunc(test_df.loc[0,'playerBoxScores'],date)
    if scores.empty:
        scores['playerId'] = sub_df['playerId']
        scores['date'] = sub_df['date']
        col_score2 = [x for x in col_scoreP if x not in ['playerId','date']]
        for col in col_score2:
            scores[col] = np.nan
        scores['date_playerId'] = sub_df['date_playerId']
    else:
        scores.rename(columns={'teamId':'teamId_score'},inplace=True)
        scores = scores[col_scoreP]
        scores['date_playerId'] = scores['date'].astype(str)+'_'+scores['playerId'].astype(str)
        scores = scores[~scores['date_playerId'].duplicated(keep='last')]

    #====== Transaction
    trans = decodeFunc(test_df.loc[0,'transactions'],date) 
    if trans.empty:
        trans['playerId'] = sub_df['playerId']
        trans['date'] = sub_df['date']
        trans['bTrans'] = 0
        trans['date_playerId'] = sub_df['date_playerId']
    else:
        trans = trans[col_tran]
        trans['bTrans'] = trans['typeCode'].apply(lambda x: 1 if x==x else 0)
        trans['date_playerId'] = trans['date'].astype(str)+'_'+trans['playerId'].astype(str)
        trans = trans[~trans['date_playerId'].duplicated(keep='last')]
        #trans['typeCode'].fillna('unknown',inplace=True)
    

    #====== Awards
    awards = decodeFunc(test_df.loc[0,'awards'],date)
    if awards.empty:
        awards['playerId'] = sub_df['playerId']
        awards['date'] = sub_df['date']
        awards['awardPlayerTeamId'] = 0
        awards['awardName'] = 'noaward'
        awards['bAward'] = 0
        awards['date_playerId'] = sub_df['date_playerId']
    else:
        awards = awards[col_awards]
        #awards = awards.groupby(['playerId','date','awardPlayerTeamId'])['awardName'].sum('_')
        #awards = awards.reset_index()
        awards['awardPlayerTeamId'].fillna(0,inplace=True)
        awards['awardName'].fillna('noward',inplace=True)
        awards['bAward'] = awards['awardName'].apply(lambda x: 1 if x==x else 0)
        awards['date_playerId'] = awards['date'].astype(str)+'_'+awards['playerId'].astype(str)
        awards = awards[~awards['date_playerId'].duplicated(keep='last')]
    

    #====== Twitter player
    twitP = decodeFunc(test_df.loc[0,'playerTwitterFollowers'],date)
    if twitP.empty:
        twitP['playerId'] = sub_df['playerId']
        twitP['date'] = sub_df['date']
        twitP['numberOfFollowers'] = np.nan
        twitP['date_playerId'] = sub_df['date_playerId']
    else:
        twitP = twitP[col_twP]
        twitP['date_playerId'] = twitP['date'].astype(str)+'_'+twitP['playerId'].astype(str)
        twitP = twitP[~twitP['date_playerId'].duplicated(keep='last')]


    #====== Team standing
    stand = decodeFunc(test_df.loc[0,'standings'],date)
    if stand.empty:
        stand['teamId'] = 0
        stand['date'] = sub_df['date']
        col_stand2 = [x for x in col_stand if x not in ['teamId','date']]
        for col in col_stand2:
            stand[col] = np.nan
        stand.rename(columns=lambda x: x+'T', inplace=True)
    else:
        stand = stand[col_stand]
        stand.rename(columns=lambda x: x+'T', inplace=True)

    #====== Twitter Team
    twitT = decodeFunc(test_df.loc[0,'teamTwitterFollowers'],date)
    if twitT.empty:
        twitT['teamId'] = 0
        twitT['date'] = sub_df['date']
        twitT['numberOfFollowersT'] = np.nan
    else:
        twitT.rename(columns={'numberOfFollowers':'numberOfFollowersT'},inplace=True)
        twitT = twitT[col_twT]
        
    return games,rosters,awards,twitP,stand,twitT,trans,scores


def merge(targets,players,rosters,awards,twP,stand,twT,trans,scoreP)->pd.DataFrame():
    
    targets2 = targets.merge(players,how='left',on=['playerId'])
    targets2 = targets2[targets2.age.notnull()]
    
    targets3 = targets2.merge(rosters[['teamId','status','date_playerId']],how='left',on=['date_playerId',])
    targets3['teamId'].fillna(0,inplace=True)
    targets3['status'].fillna('unknown',inplace=True)
    
    targets4 = targets3.merge(awards[['bAward','date_playerId']],how='left',on=['date_playerId'])
    
    targets5 = targets4.merge(twP[['date_playerId','numberOfFollowers']],how='left',on=['date_playerId'])
    targets5 = targets5.sort_values(by='date')
    targets5['numberOfFollowers'] = targets5.groupby('playerId')['numberOfFollowers'].fillna(method='bfill')
    targets5['numberOfFollowers'] = targets5.groupby('playerId')['numberOfFollowers'].fillna(method='ffill')
    targets5['numberOfFollowers'].fillna(0,inplace=True)
    
    # team stand
    targets6 = targets5.merge(stand,how='left',left_on=['teamId','date'],right_on=['teamIdT','dateT'])
    #targets6 = targets6.fillna(0)
    targets6[
        ['leagueGamesBackT','sportGamesBackT','divisionGamesBackT','eliminationNumberT']
    ] = targets6[
        ['leagueGamesBackT','sportGamesBackT','divisionGamesBackT','eliminationNumberT']
    ].replace('-',0)
    targets6.drop(columns=['teamIdT','dateT'],inplace=True)
    
    targets7 = targets6.merge(twT,how='left',on=['teamId','date'])
    targets7 = targets7.sort_values(by='date')
    targets7['numberOfFollowersT'] = targets7.groupby('teamId')['numberOfFollowersT'].fillna(method='bfill')
    targets7['numberOfFollowersT'] = targets7.groupby('teamId')['numberOfFollowersT'].fillna(method='ffill')
    targets7['numberOfFollowersT'].fillna(0,inplace=True)
    
    targets8 = targets7.merge(trans[['date_playerId','bTrans']],how='left',on=['date_playerId'])
    
    targets9 = targets8.merge(scoreP.drop(columns=['playerId','date']),how='left',on=['date_playerId'])
    
    #targets9['teamId'] = targets9[['teamId','teamId_score']].apply(lambda x:x['teamId_score'] if x['teamId_score']==x['teamId_score'] and x['teamId_score']!=x['teamId'] else x['teamId'],axis=1)
    targets9 = targets9.drop(columns=['teamId_score'])

    return targets9


# %%
def clean(targets, games,cg_key,cg_int,cg_cat,cg_flag,
          ct_key,ct_int,ct_cat,ct_flag,col_float,encoder)->pd.DataFrame():
    
    games.rename(columns={
        'homeWins':'homeWinsG','homeLosses':'homeLossesG','awayWins':'awayWinsG',
        'awayLosses':'awayLossesG'},inplace=True)
    
    #
    targets['eliminationNumberT'] = targets['eliminationNumberT'].apply(lambda x:0 if str(x)=='E' else x)
    targets['gamePk'] = targets['gamePk'].fillna(0)
    targets[ct_cat] = targets[ct_cat].fillna('unknown')
    targets[ct_int] = targets[ct_int].fillna(0)
    targets[ct_flag] = targets[ct_flag].fillna(0)
    
    targets[ct_int] = targets[ct_int].apply(pd.to_numeric)
    targets[ct_int] = targets[ct_int].fillna(0)
    targets[ct_cat] = targets[ct_cat].astype('str')
    targets[ct_int] = targets[ct_int].astype(int)
    targets[ct_flag] = targets[ct_flag].astype(int)
    
    #for col in ct_flag:
    #    targets[col] = targets[col].apply(lambda x:'True' if str(x)=='1' else x)
    #    targets[col] = targets[col].apply(lambda x:'False' if str(x)=='0' else x)
    #    targets[col] = targets[col].apply(lambda x:'True' if str(x)=='1.0' else x)
    #    targets[col] = targets[col].apply(lambda x:'False' if str(x)=='0.0' else x)

    targets = targets.sort_values(by='date')
    targets2 = targets.merge(games,how='left',on=['gamePk','date'])
    # correct columns
    #targets2['divisionId'] = targets2['divisionId'].astype(float).astype(int)
    
    #game columns
    targets2[cg_cat] = targets2[cg_cat].fillna('unknown')
    targets2[cg_int] = targets2[cg_int].fillna(0)
    targets2[cg_flag] = targets2[cg_flag].fillna(0)
    
    targets2[cg_cat] = targets2[cg_cat].astype('str')
    targets2[cg_int] = targets2[cg_int].astype(float).astype(int)
    targets2[cg_flag] = targets2[cg_flag].astype(int)
   
    #for col in cg_flag:
    #    targets2[col] = targets2[col].apply(lambda x:'True' if x=='1' else x)
    #    targets2[col] = targets2[col].apply(lambda x:'True' if x=='1.0' else x)
    #    targets2[col] = targets2[col].apply(lambda x:'False' if x=='0' else x)
    #    targets2[col] = targets2[col].apply(lambda x:'False' if x=='0.0' else x)
    
    targets2[col_float] = targets2[col_float].fillna(0.0)    
    targets2[col_float] = targets2[col_float].astype(float)
    
    #
    col_encode = ct_cat+cg_cat
    targets2[col_encode] = encoder.transform(targets2[col_encode])
    #
    df_game = targets2[targets2['gamePk']!=0]
    df_game.drop(columns=['gamePk','teamId','homeId','awayId'],inplace=True)
    
    df_nogame = targets2[targets2['gamePk']==0]
    
    col_dropnogame = [
        'gamePk','scheduledInnings','gameNumber','gamesInSeries','homeWinsG',
        'homeLossesG','awayWinsG','awayLossesG','homeId','awayId',
        'detailedGameState','doubleHeader','dayNight','seriesDescription',
        'isTie','homeWinner','awayWinner','homeWinPct','awayWinPct'
    ] 
    df_nogame.drop(columns=col_dropnogame,inplace=True)
    
    return df_game, df_nogame


# %%
def add_stat(df_game,df_nogame,labelStat):
    # dateDate, year, month
    labelStat.rename(columns={'playerId':'playerIdKey'},inplace=True)
    df_game['dateDate'] = pd.to_datetime(df_game['date'],format='%Y%m%d')
    df_game['month'] = df_game['dateDate'].dt.month
    df_game['playerIdKey'] = df_game['date_playerId'].map(lambda x: int(x.split('_')[1]))
    
    df_nogame['dateDate'] = pd.to_datetime(df_nogame['date'],format='%Y%m%d')
    df_nogame['month'] = df_nogame['dateDate'].dt.month
    df_nogame['playerIdKey'] = df_nogame['date_playerId'].map(lambda x: int(x.split('_')[1]))
    #label['day'] = label['date'].dt.day
    
    df_game = df_game.merge(labelStat,how='left',on=['playerIdKey','month'])
    df_nogame = df_nogame.merge(labelStat,how='left',on=['playerIdKey','month'])
    
    df_game.fillna(0.0,inplace=True)
    df_nogame.fillna(0.0,inplace=True)
    
    df_game = df_game.drop(columns=['dateDate','month','playerIdKey'])
    df_nogame = df_nogame.drop(columns=['dateDate','month','playerIdKey'])
    
    return df_game, df_nogame


def add_lag(df_game,df_nogame,label,LAG=1,WIN_SIZE=[3]):
    col_label = ['target1','target2','target3','target4']
    col_mean = ['target1Mean','target2Mean','target3Mean','target4Mean']
    col_max = ['target1Max','target2Max','target3Max','target4Max']
    col_min = ['target1Min','target2Min','target3Min','target4Min']
    col_std = ['target1Std','target2Std','target3Std','target4Std']
    col_median = ['target1Med','target2Med','target3Med','target4Med']
    
    label_t = label.sort_values(by='date').reset_index(drop=True).copy(deep=True)
    col_lag = ['date_playerId']
    
    # lag targets
    for i in range(1,LAG+1):
        col_new = [x+'_Lag'+str(i) for x in col_label]
        label_t[col_new] = label_t.groupby('playerId')[col_label].shift(i)
        col_lag = col_lag+col_new
    
    col_roll = col_lag
    #roll() mean, max, min, std, median: run every iter, 4*5*Nsize
    for i in WIN_SIZE:
        col_meanR = [x+'_Size'+str(i) for x in col_mean]
        label_t[col_meanR] = label_t.groupby('playerId')[col_label].rolling(window=i).mean().reset_index(0,drop=True)

        col_maxR = [x+'_Size'+str(i) for x in col_max]
        label_t[col_maxR] = label_t.groupby('playerId')[col_label].rolling(window=i).max().reset_index(0,drop=True)

        col_minR = [x+'_Size'+str(i) for x in col_min]
        label_t[col_minR] = label_t.groupby('playerId')[col_label].rolling(window=i).min().reset_index(0,drop=True)

        col_medianR = [x+'_Size'+str(i) for x in col_median]
        label_t[col_medianR] = label_t.groupby('playerId')[col_label].rolling(window=i).median().reset_index(0,drop=True)

        col_stdR = [x+'_Size'+str(i) for x in col_std]
        label_t[col_stdR] = label_t.groupby('playerId')[col_label].rolling(window=i).std().reset_index(0,drop=True)
        col_roll = col_roll+col_meanR+col_maxR+col_minR+col_medianR+col_stdR
    
    label_t = label_t[col_roll].dropna()
    df_game = df_game.merge(label_t,how='left',on=['date_playerId'])
    df_nogame = df_nogame.merge(label_t,how='left',on=['date_playerId'])
    
    df_game.dropna(inplace=True)
    df_nogame.dropna(inplace=True)
    
    return df_game, df_nogame,label_t.columns.tolist()[1:]

# clean data: key, int, cat, flag for tragets and games
ct_key = [
    'date_playerId','teamId','date','gamePk',
]
ct_int = [
    'age','ageStart','yearPlay','numberOfFollowers',
    # stand
    'divisionRankT','leagueRankT','leagueGamesBackT','sportGamesBackT',
    'divisionGamesBackT','winsT','lossesT','runsAllowedT','runsScoredT',
    'eliminationNumberT','homeWinsT','homeLossesT','awayWinsT','awayLossesT',
    'lastTenWinsT','lastTenLossesT','extraInningWinsT','extraInningLossesT',
    'oneRunWinsT','oneRunLossesT','dayWinsT','dayLossesT','nightWinsT',
    'nightLossesT','grassWinsT','grassLossesT','turfWinsT','turfLossesT','divWinsT','divLossesT',
    #
    'numberOfFollowersT',
    # score
    'jerseyNum','battingOrder','flyOuts','groundOuts','runsScored','doubles','triples',
    'homeRuns','strikeOuts','baseOnBalls','intentionalWalks','hits','hitByPitch',
    'atBats','caughtStealing','stolenBases','groundIntoDoublePlay','groundIntoTriplePlay',
    'plateAppearances','totalBases','rbi','leftOnBase','sacBunts','sacFlies',
    'catchersInterference','pickoffs',
    'flyOutsPitching','airOutsPitching','groundOutsPitching','runsPitching','doublesPitching',
    'triplesPitching','homeRunsPitching','strikeOutsPitching','baseOnBallsPitching',
    'intentionalWalksPitching','hitsPitching','hitByPitchPitching','atBatsPitching',
    'caughtStealingPitching','stolenBasesPitching','inningsPitched','earnedRuns',
    'battersFaced','outsPitching','pitchesThrown','balls','strikes','hitBatsmen',
    'balks','wildPitches','pickoffsPitching','rbiPitching','gamesFinishedPitching',
    'inheritedRunners','inheritedRunnersScored','catchersInterferencePitching',
    'sacBuntsPitching','sacFliesPitching','assists','putOuts','errors','chances',
    #
    
    
]
ct_cat = [
    'playerId','birthCity','birthStateProvince','birthCountry',
    'primaryPositionName','status',
    # stand
    'divisionIdT','divisionChampT','divisionLeaderT',
    
]
ct_flag = [
    'bAward','bTrans',
    # score        
    'home','gamesPlayedBatting','gamesPlayedPitching','gamesStartedPitching',
    'completeGamesPitching','shutoutsPitching','winsPitching','lossesPitching',
    'saveOpportunities','saves','holds','blownSaves',
]

## game
cg_key = ['gamePk','date']

cg_int = ['scheduledInnings','gameNumber','gamesInSeries','homeWinsG','homeLossesG',
           'awayWinsG','awayLossesG','homeId','awayId']

cg_cat = ['detailedGameState','doubleHeader','dayNight',
           'seriesDescription']

cg_flag = ['isTie','homeWinner','awayWinner'] 

col_float = [
    'homeWinPct','awayWinPct','BMI','pctT'
]

col_key = ['date','date_playerId']


# %%
# target1: player+team
col_t1_game = [
    'rbi','homeRuns','numberOfFollowers','numberOfFollowersT',
    'plateAppearances','totalBases','ageStart','primaryPositionName',
    'dayNight','pctT','inningsPitched','grassWinsT','grassLossesT','dayWinsT','age',
    'atBatsPitching','hitsPitching','gamesStartedPitching','winsPitching','leagueRankT',
    'strikeOutsPitching','divWinsT','playerId','hits','awayLossesG',
    'divisionGamesBackT','BMI','saves','homeLossesT','awayWinsG',
    'lastTenWinsT','leagueGamesBackT','yearPlay','runsPitching',
    'gamesInSeries','oneRunWinsT','airOutsPitching','winsT',
    'birthCity','divisionRankT','bAward','earnedRuns',
    'divLossesT','awayLossesT','extraInningLossesT','homeWinsG',
    'awayWinPct','turfWinsT','pitchesThrown','atBats','divisionIdT',
    'completeGamesPitching','strikes','nightWinsT','outsPitching','extraInningWinsT', 
    'homeWinsT', 'awayWinsT','bTrans'
]


col_t1_nogame = [
    'status','numberOfFollowers','primaryPositionName','age',
    'ageStart','yearPlay','numberOfFollowersT','playerId',
    'divLossesT','runsAllowedT','BMI','dayWinsT','pctT',
    'birthCity', 'birthCountry', 'grassWinsT','bTrans'
    
]


# %%
# target2: 
col_t2_game = [
    'numberOfFollowers','plateAppearances','numberOfFollowersT','ageStart','dayNight',
    'age','atBats','totalBases','primaryPositionName','leagueRankT','pctT','saves','yearPlay',
    'BMI','rbi','playerId','homeRuns','homeWinsT','gamesStartedPitching','runsScored',
    'birthCountry','lastTenLossesT','runsScoredT','grassWinsT','birthStateProvince','gamesInSeries',
    'birthCity','hits','winsPitching','sportGamesBackT','dayLossesT','awayWinPct','strikeOutsPitching',
    'divisionIdT','lastTenWinsT','grassLossesT','homeLossesG','awayWinsG','lossesPitching',
    'leagueGamesBackT','outsPitching','saveOpportunities','inningsPitched',
    'divisionGamesBackT','awayLossesT','leftOnBase','baseOnBalls','lossesT','turfWinsT',
    'homeLossesT', 'bAward','bTrans'
]

col_t2_nogame = [
    'status','numberOfFollowers','numberOfFollowersT','yearPlay','runsAllowedT','ageStart',
    'divisionRankT','pctT','lossesT','divLossesT','divWinsT','playerId','leagueGamesBackT','leagueRankT',
    'winsT','nightLossesT','dayLossesT','awayLossesT','bTrans'
]


# %%
# target3: player score
col_t3_game = [
    'numberOfFollowers','homeRuns','gamesStartedPitching','pitchesThrown','rbi','numberOfFollowersT',
    'ageStart','totalBases','inningsPitched','runsPitching','strikes','leagueRankT','playerId',
    'strikeOutsPitching','battersFaced','age','lossesPitching','yearPlay','pctT','extraInningLossesT',
    'plateAppearances','saves','balls','primaryPositionName','outsPitching','winsPitching',
    'leagueGamesBackT','dayWinsT','awayWinsG','earnedRuns','rbiPitching','atBatsPitching','BMI','divisionIdT',
    'birthStateProvince','divisionGamesBackT','airOutsPitching','birthCity','bAward','dayNight',
    'winsT','sportGamesBackT','homeLossesT','turfWinsT','divisionRankT','grassWinsT','birthCountry',
    'runsAllowedT','atBats','runsScored','gamesInSeries','lastTenLossesT','hits','awayWinPct',
    'homeWinsT','lastTenWinsT','dayLossesT','lossesT','runsScoredT','leftOnBase',
    'saveOpportunities','awayLossesT','baseOnBalls','grassLossesT','bTrans'
]

col_t3_nogame = [
    'numberOfFollowers','status','ageStart','playerId','age','numberOfFollowersT','yearPlay',
    'birthCity','birthStateProvince','BMI','pctT','runsAllowedT','primaryPositionName',
    'divLossesT','lossesT','divisionIdT','awayWinsT','birthCountry','leagueGamesBackT','bTrans'
]

# target4
col_t4_game = [
    'numberOfFollowers','ageStart','yearPlay','numberOfFollowersT','plateAppearances',
    'playerId','totalBases','homeWinsT','gamesStartedPitching','saves','pctT',
    'dayNight','BMI','age','birthCity','birthStateProvince','primaryPositionName','runsScoredT',
    'homeLossesT','birthCountry','leagueRankT','awayLossesT','winsPitching','homeRuns',
    'awayWinPct','rbi','gamesInSeries','awayWinsT','sportGamesBackT','leagueGamesBackT',
    'runsAllowedT','grassWinsT','dayLossesT','lossesT','atBats','nightWinsT','divisionIdT',
    'awayWinsG','divisionGamesBackT','hits','oneRunLossesT','lastTenLossesT','divLossesT',
    'divisionRankT','dayWinsT','runsScored','extraInningLossesT','grassLossesT','divWinsT',
    'awayLossesG', 'turfLossesT','bTrans'
]

col_t4_nogame = ['numberOfFollowers','yearPlay','status','numberOfFollowersT','ageStart',
                 'age','playerId','primaryPositionName','birthStateProvince','divLossesT',
                 'birthCity','BMI','pctT','runsAllowedT','birthCountry','divWinsT',
                 'runsScoredT','winsT','lossesT','dayLossesT','nightLossesT','awayLossesT',
                 'divisionIdT','nightWinsT','extraInningLossesT','leagueGamesBackT','awayWinsT',
                 'leagueRankT','divisionGamesBackT','lastTenWinsT','grassLossesT','bTrans'
]

#
players = pd.read_csv(TRAIN_DIR / 'player.csv')
labelStat = pd.read_csv(TRAIN_DIR / 'labelStat.csv')
labelAll = pd.read_csv(TRAIN_DIR / 'targetsAll.csv')

labelAll = labelAll[(labelAll.date>20210101)&(labelAll.date<20210501)]

# shift last day targ to today
labelAll['date'] = labelAll['date_playerId'].map(lambda x: int(x.split('_')[0]))
labelAll['date'] = labelAll['date'].astype(int)
labelAll['dateDate'] = pd.to_datetime(labelAll['date'],format='%Y%m%d')
labelAll['dateDate'] = labelAll['dateDate'] + pd.Timedelta(days=1)
labelAll['date_playerId'] = labelAll['dateDate'].dt.strftime('%Y%m%d')+'_'+labelAll['playerId'].astype(str)
labelAll.drop(columns='dateDate',inplace=True)

col_stat = labelStat.columns.tolist()[2:]

cat_encoder = OrdinalEncoder()
cat_encoder.categories_ = np.load(TRAIN_DIR / 'encoder_all.npy',allow_pickle=True)

if __name__ == '__main__':

    mlb = MLBEmulator(eval_start_day=20210430, eval_end_day=20210530)
    env = mlb.make_env()

    lag_name = 'lag10rol6315'

    model1 = joblib.load(TRAIN_DIR / Path('t1g_'+lag_name+'.pkl'))
    model2 = joblib.load(TRAIN_DIR / Path('t2g_'+lag_name+'.pkl'))
    model3 = joblib.load(TRAIN_DIR / Path('t3g_'+lag_name+'.pkl'))
    model4 = joblib.load(TRAIN_DIR / Path('t4g_'+lag_name+'.pkl'))

    model1_no = joblib.load(TRAIN_DIR / Path('t1n_'+lag_name+'.pkl'))
    model2_no = joblib.load(TRAIN_DIR / Path('t2n_'+lag_name+'.pkl'))
    model3_no = joblib.load(TRAIN_DIR / Path('t3n_'+lag_name+'.pkl'))
    model4_no = joblib.load(TRAIN_DIR / Path('t4n_'+lag_name+'.pkl'))


    yLabel = pd.DataFrame()
    xData = pd.DataFrame()
    yPred = pd.DataFrame()


    for n, (test_df, sub_df) in enumerate(env.iter_test()):
        #submisson
        sub_df = sub_df.reset_index()
        sub_df['playerId'] = sub_df['date_playerId'].map(lambda x: int(x.split('_')[1]))
        sub_df = sub_df[['date','date_playerId','playerId']]
        #display(sub_df)
        #testing data
        test_df = test_df.reset_index()
        
       
        date = test_df.loc[0,'date']
        
        # True label for testing
        label = decodeFunc(test_df.loc[0,'nextDayPlayerEngagement'],date)
        label['engagementMetricsDate'] = pd.to_datetime(label['engagementMetricsDate'],format='%Y-%m-%d')
        label['engagementMetricsDate'] = label['engagementMetricsDate'].dt.strftime('%Y%m%d')
        label['date_playerId'] = label['engagementMetricsDate']+'_'+label['playerId'].astype(str)
        label = label.drop(columns=['engagementMetricsDate'])
        yLabel = pd.concat([yLabel,label],axis=0)
        
        games,rosters,awards,twitP,stand,twitT,trans,scores = splitdata(sub_df,
        test_df,col_rosters,col_awards,col_twP,col_twT,col_stand,col_tran,col_scoreP,col_game)
        
        #
        targets = merge(sub_df,players,rosters,awards,twitP,stand,twitT,trans,scores)

        #
        df_game, df_nogame = clean(targets, games,cg_key,cg_int,cg_cat,cg_flag,
                            ct_key,ct_int,ct_cat,ct_flag,col_float,cat_encoder)
        
        df_game1, df_nogame1 = add_stat(df_game, df_nogame,labelStat)
        
        
        df_game2, df_nogame2,col_lag = add_lag(df_game1, df_nogame1,labelAll,10,[6,9,12,15])
        
        
        if not df_game2.empty:
            pred1_game = model1.predict(df_game2[col_t1_game+col_stat+col_lag].values)
            pred2_game = model2.predict(df_game2[col_t2_game+col_stat+col_lag].values)
            pred3_game = model3.predict(df_game2[col_t3_game+col_stat+col_lag].values)
            pred4_game = model4.predict(df_game2[col_t4_game+col_stat+col_lag].values)

            df_game2['target1'] = np.clip(pred1_game, 0, 100)
            df_game2['target2'] = np.clip(pred2_game, 0, 100)
            df_game2['target3'] = np.clip(pred3_game, 0, 100)
            df_game2['target4'] = np.clip(pred4_game, 0, 100)
        
        if not df_nogame2.empty:
            pred1_nogame = model1_no.predict(df_nogame2[col_t1_nogame+col_stat+col_lag].values)
            pred2_nogame = model2_no.predict(df_nogame2[col_t2_nogame+col_stat+col_lag].values)
            pred3_nogame = model3_no.predict(df_nogame2[col_t3_nogame+col_stat+col_lag].values)
            pred4_nogame = model4_no.predict(df_nogame2[col_t4_nogame+col_stat+col_lag].values)

            df_nogame2['target1'] = np.clip(pred1_nogame, 0, 100)
            df_nogame2['target2'] = np.clip(pred2_nogame, 0, 100)
            df_nogame2['target3'] = np.clip(pred3_nogame, 0, 100)
            df_nogame2['target4'] = np.clip(pred4_nogame, 0, 100)
        
        
        if df_game2.empty:
            df_all = df_nogame2[col_sub]
        elif  df_nogame2.empty:
            df_all = df_game2[col_sub]
        else:
            df_all = pd.concat([df_game2[col_sub],df_nogame2[col_sub]],axis=0)
        
        sub_df = sub_df[['date_playerId']].merge(df_all,how='left',on=['date_playerId'])
        sub_df.fillna(0,inplace=True)
        sub_df['playerId'] = sub_df['date_playerId'].map(lambda x: int(x.split('_')[1]))
        
        # targ for next day
        targ_next = sub_df.copy(deep=True)
        targ_next['date'] = targ_next['date_playerId'].map(lambda x: int(x.split('_')[0]))
        targ_next['dateDate'] = pd.to_datetime(targ_next['date'],format='%Y%m%d')
        targ_next['dateDate'] = targ_next['dateDate'] + pd.Timedelta(days=1)
        targ_next['date_playerId'] = targ_next['dateDate'].dt.strftime('%Y%m%d')+'_'+targ_next['playerId'].astype(str)
        targ_next.drop(columns='dateDate',inplace=True)
        labelAll = pd.concat([labelAll,targ_next],axis=0)
        
        sub_df = sub_df[col_sub]
        sub_df = sub_df.set_index('date')
        
        
        yPred = pd.concat([yPred,sub_df],axis=0)
        

    yPred = yPred.merge(yLabel[col_sub],how='left',suffixes=('_x', '_y'),on='date_playerId')
    yPred.head()
    s1 = mean_absolute_error(yPred['target1_x'],yPred['target1_y'])
    s2 = mean_absolute_error(yPred['target2_x'],yPred['target2_y'])
    s3 = mean_absolute_error(yPred['target3_x'],yPred['target3_y'])
    s4 = mean_absolute_error(yPred['target4_x'],yPred['target4_y'])
    print(s1,s2,s3,s4)
    print((s1+s2+s3+s4)/4)
