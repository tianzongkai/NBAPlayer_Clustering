import pandas as pd
import urllib2
from bs4 import BeautifulSoup
import numpy as np


"""
#General#
basic_stats_pl
clutch_player_stats
advanced_player_stats

#Scoring#
player_scoring_runs
player_miscellaneous_scoring
player_shot_types
player-shot-distances
player-shot-zones


#Details#
four-point-plays-and-one
player_assist_details

Clutch Time:  During the 4th quarter or overtime, with less than five minutes remaining, 
and neither team ahead by more than five points
"""

categories =np.asarray(['basic_stats_pl','clutch_player_stats','advanced_player_stats',
              'player_miscellaneous_scoring','player_shot_types',
              'player_shot_distances','player_shot_zones',
              'four_point_and_one','player_assist_details'])

url_template = 'http://www.nbaminer.com/nbaminer_nbaminer/{category}.php' \
               '?partitionpage={season}&partition2page={type}&page={page}'
"""
partitionpage: 1=17-18 season; 8=10-11; 10=08-09 season
partition2page: 1=regular; 2=playoff
"""


for idx, category in enumerate(categories) :
    category_data_collection = set()
    for page_num in range(1, 24):
        url = url_template.format(category=category, season=8, type=1, page=page_num)
        print url
        page = urllib2.urlopen(url).read()
        soup = BeautifulSoup(page, "html.parser")
        if page_num == 1:
            column_headers = soup.find_all('tr', {'class': 'header'})[0].find_all('span')
            header_list = [head.getText() for head in column_headers]
        players_rows = soup.find_all('tr', {'class': 'pg-row'})
        row_length = len(players_rows)
        print 'row_length:', row_length
        if row_length <=3:
            print 'ERROR!, NO ROWS READ'
        for i in range(row_length):
            player_row_stat = []
            for td in players_rows[i].findAll('td'):
                player_row_stat.append(td.getText())
            category_data_collection.add(tuple(player_row_stat[2:]))
    category_dataframe = pd.DataFrame(list(category_data_collection), columns=header_list, dtype='float')
    # category_dataframe.to_csv(category+'.csv')
    if idx == 0:
        whole_dataframe = category_dataframe
    else:
        whole_dataframe = pd.merge(whole_dataframe, category_dataframe,
                  how='outer', on=['Player', 'Team', 'Games'])
whole_dataframe.to_csv('nba_player_whole_stats.csv')

