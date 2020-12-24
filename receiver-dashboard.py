# IMPORTS ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
import seaborn as sns
import statsmodels
from statsmodels.nonparametric.smoothers_lowess import lowess
import streamlit as st


_lock = RendererAgg.lock
plt.style.use('default')

# SETUP ------------------------------------------------------------------------
st.set_page_config(page_title='Wide Receiver Dashboard',
                   page_icon='https://pbs.twimg.com/profile_images/'\
                             '1265092923588259841/LdwH0Ex1_400x400.jpg',
                   layout="wide")
# READ DATA --------------------------------------------------------------------
@st.cache(allow_output_mutation=True)
def get_pbp():

    #pbp data
    col_list1 = ['play_id','complete_pass','yards_gained','air_yards','touchdown',
                'epa','down','yardline_100','posteam','receiver','pass','play_type',
                'two_point_attempt','game_id','two_point_conv_result','fumble_lost',
                'week','rusher','receiver_id','success','punt_returner_player_name',
                'kickoff_returner_player_name','defteam']

    YEAR = 2020
    data_ = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/' \
                         'play_by_play_' + str(YEAR) + '.csv.gz?raw=True',
                         compression='gzip', low_memory=False, usecols=col_list1)
    return data_

data = get_pbp()
#--------
@st.cache(allow_output_mutation=True)
def get_pd():
    #plyer data
    col_list2 = ['headshot_url','full_name', 'birth_date','height','weight',
                 'college','position','team']

    player_data_ = pd.read_csv('https://github.com/mrcaseb/nflfastR-roster/blob/'\
                                'master/data/seasons/roster_2020.csv?raw=True',
                                low_memory=False, usecols=col_list2)

    return player_data_

player_data = get_pd()
#--------

COLORS = {'ARI':'#97233F','ATL':'#A71930','BAL':'#241773','BUF':'#00338D',
          'CAR':'#0085CA','CHI':'#00143F','CIN':'#FB4F14','CLE':'#FB4F14',
          'DAL':'#7F9695','DEN':'#FB4F14','DET':'#046EB4','GB':'#2D5039',
          'HOU':'#C9243F','IND':'#003D79','JAX':'#136677','KC':'#CA2430',
          'LA':'#003594','LAC':'#2072BA','LV':'#343434','MIA':'#0091A0',
          'MIN':'#4F2E84','NE':'#0A2342','NO':'#A08A58','NYG':'#192E6C',
          'NYJ':'#203731','PHI':'#014A53','PIT':'#FFC20E','SEA':'#7AC142',
          'SF':'#C9243F','TB':'#D40909','TEN':'#4095D1','WAS':'#FFC20F'}

# CLEAN DATA -------------------------------------------------------------------

#clean air_yards
data['air_yards'] = (
    np.where(
    data['air_yards'] < -10,
    data['air_yards'].median(),
    data['air_yards'])
    )
#---------
#getting rid of suffixes
names = ['receiver','rusher', 'punt_returner_player_name',
         'kickoff_returner_player_name']

for i in names:
    data[i] = data[i].str.replace(" Sr.","")
    data[i] = data[i].str.replace(" Jr.","")
#---------
#filter data
df = data[
          (data['pass']==1) &
          (data['play_type']=='pass') &
          (data.two_point_attempt==0) &
          (data['epa'].isna()==False)
          ]

#weird aj brown fumble pruitt td recovery
df.at[32403, 'touchdown'] = 0
#---------
#fpts data
fantasy = data[
          (data.play_type.isin(['no_play','pass','run','punt','kickoff'])) &
          (data['epa'].isna()==False)
          ]
fantasy  = pd.get_dummies(fantasy, columns=['two_point_conv_result'])

fantasy['fpts_skill'] = (
    fantasy['yards_gained'] * 0.1 +
    fantasy['complete_pass'] * 1 +
    fantasy['touchdown'] * 6 +
    fantasy['two_point_conv_result_success'] * 2 +
    fantasy['fumble_lost'] * -2
    )

receiving_fpts = (fantasy.groupby(
    ['receiver','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'receiver':'player'}))

rushing_fpts = (fantasy.groupby(
    ['rusher','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'rusher':'player'}))

kr_fpts = (fantasy.groupby(
    ['kickoff_returner_player_name','posteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'kickoff_returner_player_name':'player'}))

#return team is defteam pn punts
punt_fpts = (fantasy.groupby(
    ['punt_returner_player_name','defteam','week']
    )[['fpts_skill']]
    .sum()
    .reset_index()
    .sort_values(by='fpts_skill',ascending=False)
    .reset_index(drop=True)
    .rename(columns={'punt_returner_player_name':'player','defteam':'posteam'}))

fpts_skill = receiving_fpts.merge(
                rushing_fpts,on=['player','posteam','week'], how='outer'
                ).merge(
                    punt_fpts,on=['player','posteam','week'], how='outer'
                    ).merge(
                        kr_fpts,on=['player','posteam','week'], how='outer').fillna(0)

fpts_skill.columns = ['player','posteam','week','fpts_skill_x',
                      'fpts_skill_y','fpts_skill_z','fpts_skill_a']

fpts_skill['total_fpts'] = (
    fpts_skill['fpts_skill_x'] +
    fpts_skill['fpts_skill_y'] +
    fpts_skill['fpts_skill_z'] +
    fpts_skill['fpts_skill_a']
    )
#---------

# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.beta_columns(
    (.1, 2, 1.5, 1, .1)
    )

row1_1.title('NFL Receiver Dashboard')

with row1_2:
    st.write('')
    row1_2.subheader(
    'A Web App by [Max Bolger](https://twitter.com/mnpykings)')

# ROW 2 ------------------------------------------------------------------------

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.beta_columns(
    (.1, 1.6, .1, 1.6, .1)
    )

with row2_1:

    options_p = df.groupby(['receiver','posteam'])[['play_id']].count().reset_index()
    options_p = options_p.loc[options_p.play_id>29]
    player_list = options_p['receiver'].to_list()

    player_data['pbp_name'] = [item[0] + '.'+ ''.join(item.split()[1:]) for item in player_data['full_name']]
    pd_filt = player_data.loc[(player_data['pbp_name'].isin(player_list)) &
                          ((player_data['position'] == 'WR') |
                          (player_data['position'] == 'TE'))]
    #Hardcode common nflfastR names due to id issues
    remove = ['Josh Smith','Jerome Washington','D.J. Montgomery','Jaron Brown',
              'Jalen Williams','Jaeden Graham','Hunter Bryant','Connor Davis',
              'Mike Thomas','Maxx Williams','Joe Reed','Marvin Hall','Ito Smith']

    pd_filt = pd_filt.loc[~pd_filt.full_name.isin(remove)]
    player_list = pd_filt['pbp_name'].to_list()
    pd_filt = pd_filt.filter(
                ['full_name','pbp_name','team']
                    ).dropna().reset_index(drop=True).sort_values('full_name')
    records = pd_filt.to_dict('records')
    selected_data = st.selectbox('Select a Player', options=records,
        format_func=lambda record: f'{record["full_name"]}')

    player = selected_data.get('pbp_name')
    team = selected_data.get('team')



with row2_2:
    start_week, stop_week = st.select_slider(
    'Select A Range of Weeks',
    options=list(range(1,22)),
    value=(1,21))

# ROW 1 ------------------------------------------------------------------------

st.write('')
row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space4 = st.beta_columns(
    (.15, 1, .3, 1, .00000001, 3, 0.15))

with row1_1, _lock:

    player_filter = player_data.loc[(player_data['pbp_name'] == player) &
                                    (player_data['team'] == team)]
    url = player_filter['headshot_url'].dropna().iloc[-1]
    st.subheader('Player Info')
    st.image(url, width=300)



with row1_2, _lock:
    st.subheader(' ')
    st.text(' ')
    st.text(
        f"Name: {player_filter['full_name'].to_string(index=False).lstrip()}"
        )
    st.text(
        f"College: {player_filter['college'].to_string(index=False).lstrip()}"
        )
    st.text(
        f"Position: {player_filter['position'].to_string(index=False).lstrip()}"
        )
    st.text(
        f"Birthday: {player_filter['birth_date'].to_string(index=False).lstrip()}"
        )
    st.text(
        f"Height: {player_filter['height'].to_string(index=False).lstrip()}"
        )
    st.text(
        f"Weight: {player_filter['weight'].astype(int).to_string(index=False).lstrip()}"
        )

def game_logs(player,team):
  '''
  This function returns a gamelog for the desired wr
  '''
  receiver=df.loc[(df.receiver==player) & (df.posteam==team) &
                  (df.week>= start_week) & (df.week<= stop_week)]

  gamelog = receiver.groupby(['game_id']).agg(
      {
      'play_id':'count',
      'complete_pass':'sum',
      'yards_gained':'sum',
      'air_yards':'sum',
      'touchdown':'sum'
     }).rename(columns=
         {'play_id':'targets',
          'complete_pass':'receptions',
          'yards_gained':'rec. yards',
          'air_yards':'air yards',
          'touchdown':'rec. td'})


  receiver_rz = df.loc[(df.receiver==player) & (df.posteam==team) &
                 (df.week>= start_week) & (df.week<= stop_week) &
                 (df.yardline_100<20)]

  rz_tgts = receiver_rz.groupby(['game_id'])[['play_id']].count().rename(
                            columns = {'play_id':'rz tgts'}
                            )

  gamelog = pd.concat([gamelog,rz_tgts], axis=1).fillna(0)

  gamelog = gamelog[['targets', 'receptions', 'rec. yards',
                     'air yards','rz tgts','rec. td']]

  cm = sns.light_palette(COLORS.get(team), as_cmap=True)

  gamelog_style = gamelog.style.background_gradient(cmap=cm).set_precision(0)

  return gamelog_style

with row1_3, _lock:
    st.subheader('Game Log')
    has_data = len((df.loc[(df.receiver==player) & (df.posteam==team) &
                    (df.week>= start_week) & (df.week<= stop_week)]))
    if has_data > 0:
        st.dataframe(game_logs(player,team), width=5000,height=700)

    else:
        st.error(
            "Oops! This player did not play during the selected time period. "\
            "Change the filter and try again.")
        st.stop()

# ROW 2 ------------------------------------------------------------------------
st.write('')
row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.beta_columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))


def air_yards(player, team):
  '''
  This function returns an ay dist for the desired wr
  '''
  receiver=df.loc[(df.receiver==player) & (df.posteam==team) &
                  (df.week>= start_week) & (df.week<= stop_week)]

  fig1 = Figure()
  ax = fig1.subplots()
  sns.kdeplot(data=df['air_yards'], color='#CCCCCC',
                fill=True, label='NFL Average',ax=ax)
  sns.kdeplot(data=receiver['air_yards'], color=COLORS.get(team),
                fill=True, label=player,ax=ax)
  ax.legend()
  ax.set_xlabel('Air Yards', fontsize=12)
  ax.set_ylabel('Density', fontsize=12)
  ax.grid(zorder=0,alpha=.2)
  ax.set_axisbelow(True)
  ax.set_xlim([-10,55])
  st.pyplot(fig1)

with row2_1, _lock:
    st.subheader('Air Yards Distribution')
    air_yards(player,team)

def ay_bins(player, team):
    '''
    This function returns binned ay counts
    '''

    receiver=df.loc[(df.receiver==player) & (df.posteam==team) &
                    (df.week>= start_week) & (df.week<= stop_week)]
    bins = pd.DataFrame(pd.cut(receiver.air_yards,bins=4).value_counts()).T

    fig2 = Figure()
    ax = fig2.subplots()
    sns.barplot(data = bins,color=COLORS.get(team),ax=ax)
    ax.set_xlabel('Air Yards', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    st.pyplot(fig2)

with row2_2, _lock:
    st.subheader('Air Yards Binned')
    ay_bins(player,team)

def ay_tgt(player, team):
    '''
    This function returns binned ay counts
    '''

    week_filter = df.loc[(df.week>= start_week) & (df.week<= stop_week) &
                         (df.receiver.isin(player_list))]

    ay_tgt = week_filter.groupby(['receiver','posteam']).agg(
                {'air_yards':'sum','play_id':'count'}
                ).reset_index().sort_values(
                    by=['air_yards'],ascending=False
                    ).reset_index(drop=True)
    tgt_filt = stop_week - start_week

    ay_tgt = ay_tgt.loc[ay_tgt.play_id>tgt_filt]
    fig3 = Figure()
    ax = fig3.subplots()

    sns.scatterplot(x=ay_tgt.play_id, y=ay_tgt.air_yards,data=ay_tgt,
    color='#E8E8E8',ax=ax)

    sns.scatterplot(x=ay_tgt[(ay_tgt.receiver==player) &
                             (ay_tgt.posteam==team)].play_id,
                             y=ay_tgt.air_yards,data=ay_tgt,
                             color=COLORS.get(team),s=100,
                             legend=False, ax=ax)

    x = ay_tgt.play_id
    y = ay_tgt.air_yards
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, 'k',alpha=.2,linestyle='-')

    ax.set_xlabel('Targets', fontsize=12)
    ax.set_ylabel('Air Yards', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    st.pyplot(fig3)

with row2_3, _lock:
    st.subheader('Air Yards as a Function of Targets')
    ay_tgt(player,team)

# ROW 3 ------------------------------------------------------------------------
st.write('')
row3_space1, row3_1, row3_space2, row3_2, row3_space3, row3_3, row3_space4 = st.beta_columns(
    (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))

def fpts_chart(player, team):
    '''
    This function returns fpts by week
    '''
    week_filter = fpts_skill.loc[(fpts_skill.week>= start_week) &
                                 (fpts_skill.week<= stop_week)]

    fpts_player = week_filter[(week_filter.player==player) &
                            (week_filter.posteam==team)].sort_values(by='week')

    fig4 = Figure()
    ax = fig4.subplots()

    sns.lineplot(data=fpts_player, x="week", y="total_fpts",
                 color=COLORS.get(team), marker='o',
                 markersize=10, linewidth=2, ax=ax)
    ax.set_xticks(list(fpts_player.week))
    ax.axhline(y=fpts_player['total_fpts'].mean(),linestyle='--',
                 color='black', label=f"{player}'s Avg")
    avg = fpts_player['total_fpts'].mean()
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('PPR Points', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    ax.legend()
    st.pyplot(fig4)

with row3_1, _lock:
    st.subheader('PPR Points by Week')
    fpts_chart(player, team)

def epa_chart(player, team):
    '''
    This function returns epa chart
    '''

    week_filter = df.loc[(df.week>= start_week) & (df.week<= stop_week) &
                         (df.receiver.isin(player_list))]
    epa = week_filter.groupby(['receiver','posteam']).agg(
            {'success':'mean','epa':'mean','play_id':'count'}
            ).reset_index()
    #error handling similar names due to id issues
    epa = epa.loc[~((epa['receiver'] == 'I.Smith') & (epa['posteam'] == 'ATL')) &
                  ~((epa['receiver'] == 'M.Brown') & (epa['posteam'] == 'LAR')) &
                  ~((epa['receiver'] == 'D.Johnson') & (epa['posteam'] == 'HOU'))]

    tgt_filt = stop_week - start_week
    epa = epa.loc[epa.play_id>tgt_filt]
    epa['color'] = '#EFEFEF'
    epa.loc[(epa.receiver==player) & (epa.posteam==team),'color'] = COLORS.get(team)
    epa = epa.sort_values(by='color',ascending=False)
    fig5 = Figure()
    ax = fig5.subplots()

    sns.scatterplot(x=epa.success, y=epa.epa,data=epa,
    color=epa.color,s=(epa.play_id * 2),ax=ax)

    ax.axhline(y=epa.epa.mean(),linestyle='--',color='black',alpha=0.2)
    ax.axvline(x=epa.success.mean(),linestyle='--',color='black',alpha=0.2)
    # ax.get_legend().remove()
    ax.set_xlabel('Success Rate', fontsize=12)
    ax.set_ylabel('EPA/Target', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    st.pyplot(fig5)

with row3_2, _lock:
    st.subheader('EPA/Target')
    epa_chart(player, team)

def catch_rate(player,team):

    week_filter = df.loc[(df.week>= start_week) & (df.week<= stop_week) &
                         (df.receiver.isin(player_list))]

    test = week_filter[week_filter.receiver==player]
    plr = test.groupby('air_yards').agg(
            {'complete_pass':'mean','play_id':'count'}
                ).reset_index()

    nfl = df.groupby('air_yards').agg(
            {'complete_pass':'mean','play_id':'count'}
                ).reset_index()

    fig6 = Figure()
    ax = fig6.subplots()
    sns.regplot(data=plr,x=plr.air_yards,y=plr.complete_pass,lowess=True,
                    scatter_kws={'s':plr.play_id * 10},
                    color=COLORS.get(team), ax=ax)
    sns.regplot(data=nfl,x=nfl.air_yards,y=nfl.complete_pass,lowess=True,
                    scatter = False, color='gray',ax=ax)
    ax.set_xlabel('Depth of Target', fontsize=12)
    ax.set_ylabel('Catch Rate', fontsize=12)
    ax.grid(zorder=0,alpha=.2)
    ax.set_axisbelow(True)
    st.pyplot(fig6)

with row3_3, _lock:
    st.subheader('Catch Rate')
    catch_rate(player, team)


# ROW 5 ------------------------------------------------------------------------
row5_spacer1, row5_1, row5_spacer2 = st.beta_columns((.1, 3.2, .1))

with row5_1:
    st.markdown('___')
    about = st.beta_expander('About/Additional Info')
    with about:
        '''
        Thanks for checking out my app! It was built entirely using [nflfastR]
        (https://www.nflfastr.com/) data. Special thanks to [Ben Baldwin]
        (https://twitter.com/benbbaldwin) and [Sebastian Carl]
        (https://twitter.com/mrcaseb) who do a great job maintaining this public
        data, making the barrier to entry for NFL analytics incredibly low! This
        is the first time I have ever built something like this, so any comments
        or feedback is greatly appreciated. I hope you enjoy!

        ---

        This app is a dashboard that runs an analysis on any desired WR or TE who
        has logged at least 30 total targets in the 2020 season. Player info,
        a game log, and six visualizations of various statistics are displayed
        for the selected player. They are briefly described below:

        **Player Info** - Headshot along with the name, college,
        position, height,  and weight of the selected player.

        **Game Log** - A boxscore for each game the selected player appeared in.

        **Air Yards Distribution** - A density plot of air yards for the
        selected player. NFL Average in gray.
        (NFL Average takes into account every pass)

        **Binned Air Yards** - A Bar graph of target counts with five discrete
        air yards bins for the selected player.

        **Air Yards as a Function of targets** - A scatterplot of air yards
        as a function of targets. Selected player is colored. Trend line is
        shown to indicate whether a player is seeing more (or less) air yards
        than their target count suggests based on the sample.
        (Only players available in this dashboard are charted)

        **PPR Points by Week** - A line chart of PPR fantasy points the selected
        player has scored each week using vanilla fantasy socring. A missing
        week label means the selected player did not play or was on BYE.
        (Does *not* include fantasy points from passing, but *does* include 6
        point bonuses for rush and return touchdowns, which aren't shown on the
        game log)

        **EPA/Target** - A scatterplot of EPA/Target as a function of success
        rate (success = epa>0 on a given play). Size is a function of targets.
        Dotted lines are averages of the charted players.
        (Only players available in this dashboard are charted)

        **Catch Rate** - A scatterplot of catch rate as a function of target
        depth. Size is a function of targets. NFL average in gray. Smoothed
        using a locally weighted linear regression (LOWESS).
        (NFL Average takes into account every pass)

        *Tip - To get a better look at any individual chart, click the
        expander box!*

        *Disclaimer - Some of the air yards data might not be perfectly correct.
        The NFL has logged some incorrect air yards values this season and as a
        result, there may be some occurences where air yards values are within
        a +/-5 margain. Also, since this app is using nflfastR data, new games
        and data won't show up until they are scraped by the nflfastR team. If
        you aren't seeing new data yet, it will be updated soon.*

        ### Max Bolger, 2020
        '''
        st.image("https://www.nflfastr.com/reference/figures/logo.png",
        width= 100, caption='nflfastR')
