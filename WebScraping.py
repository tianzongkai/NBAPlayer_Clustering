import pandas as pd
import urllib2
from bs4 import BeautifulSoup


"""
player_shot_types
player_miscellaneous_scoring
player_scoring_runs

basic_stats_pl
clutch_player_stats
advanced_player_stats

player_tracking_player
player_assist_details
"""

# # url = 'http://www.nbaminer.com/player-shot-distances/'
# url = 'http://www.nbaminer.com/nbaminer_nbaminer/player_shot_distances.php?partitionpage=1&partition2page=&page=3'
# # url = 'http://www.espn.com/nba/salaries/_/year/2016'
# page = urllib2.urlopen(url).read()
# soup = BeautifulSoup(page, "html.parser")
# iframes = soup.find_all('iframe')
# embedded_frame = urllib2.urlopen(iframes[0].attrs['src'])
# iframe_soup = BeautifulSoup(embedded_frame, "html.parser")
#
# column_headers = iframe_soup.find_all('tr', {'class': 'header'})[0].find_all('span')
# header_list = [head.getText() for head in column_headers]
# # print len(header_list)
# # print header_list
#
# player_rows = iframe_soup.find_all('tr', {'class': 'pg-row'})

###############Shot Distance####################
url = 'http://www.nbaminer.com/nbaminer_nbaminer/player_shot_distances.php?partitionpage=1&partition2page=1&page=1'
page = urllib2.urlopen(url).read()
soup = BeautifulSoup(page, "html.parser")
column_headers = soup.find_all('tr', {'class': 'header'})[0].find_all('span')
header_list = [head.getText() for head in column_headers]

"""
partitionpage: 1=17-18 season; 10=08-09 season
partition2page: 1=regular; 2=playoff
"""
url_template = 'http://www.nbaminer.com/nbaminer_nbaminer/player_shot_distances.php' \
               '?partitionpage={season}&partition2page={type}&page={page}'

players_data_shot_distance = []
for page_num in range(1,3):
    url = url_template.format(season=1, type=1, page=page_num)
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page, "html.parser")
    players_rows = soup.find_all('tr', {'class': 'pg-row'})

    for i in range(len(players_rows)):
        player_row_stat = []
        # player_row_stat.append(player_rows[i].find('p'))
        for td in players_rows[i].findAll('td'):
            player_row_stat.append(td.getText())
        players_data_shot_distance.append(player_row_stat[2:])

df_shot_distance = pd.DataFrame(players_data_shot_distance, columns=header_list)
# df_shot_distance.set_index(['Player', 'Team'])

###############Shot Zone####################
url = 'http://www.nbaminer.com/nbaminer_nbaminer/player_shot_zones.php?' \
      'partitionpage=1&partition2page=1&page=1'
page = urllib2.urlopen(url).read()
soup = BeautifulSoup(page, "html.parser")
column_headers = soup.find_all('tr', {'class': 'header'})[0].find_all('span')
header_list_2 = [head.getText() for head in column_headers]
# print header_list_2
url_template = 'http://www.nbaminer.com/nbaminer_nbaminer/player_shot_zones.php' \
               '?partitionpage={season}&partition2page={type}&page={page}'

players_data_shot_zone = []
for page_num in range(1,3):
    url = url_template.format(season=1, type=1, page=page_num)
    page = urllib2.urlopen(url).read()
    soup = BeautifulSoup(page, "html.parser")
    players_rows = soup.find_all('tr', {'class': 'pg-row'})

    for i in range(len(players_rows)):
        player_row_stat = []
        # player_row_stat.append(player_rows[i].find('p'))
        for td in players_rows[i].findAll('td'):
            player_row_stat.append(td.getText())
        players_data_shot_zone.append(player_row_stat[2:])

# print players_data[0]
df_shot_zone = pd.DataFrame(players_data_shot_zone, columns=header_list_2)

merged = pd.merge(df_shot_zone, df_shot_distance,
                  how='outer', on=['Player', 'Team', 'Games'])
merged.set_index(['Player', 'Team', 'Games'])

merged.to_csv('nba_player_shotting.csv')
