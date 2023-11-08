from errno import EBADMSG
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import missingno as msno
import altair as alt
import os, sys
from configs import Configs

class dataViz:
    """
    This class generates the master.csv data file and contains functions for data viz.
    """
    def __init__(self):
        self.vizConfigs = Configs()
        self.data_dict = self.make_data_dict()
    
    def make_data_dict(self):
        """
        Reads in Bloomberg, companies, and industry data. 
        Merges dfs into a master df; saves output.
        Generates data dictionary.
        """
        data_dict = {}
        for df_name in ['bloomberg', 'companies', 'industries']:
            try:
                df = pd.read_excel(self.vizConfigs.path_dict[df_name])
            except:
                df = pd.read_csv(self.vizConfigs.path_dict[df_name])
            data_dict[df_name] = df

        data_dict['bloomberg']['Symbol'] = data_dict['bloomberg']['ID'].str.split(' ').str[0]
        master = data_dict['bloomberg'].merge(data_dict['companies'], on='Symbol') 
        master = master.rename(columns={"ESG_SCORE": "ESG", "ENVIRONMENTAL_SCORE": "Environmental", "SOCIAL_SCORE": "Social", "GOVERNANCE_SCORE": "Governance"})
        master.to_csv(self.vizConfigs.path_dict['master'])
        data_dict['master'] = master
        return data_dict

    def plot_scores(self):
        """
        Plots scores for all ESG types in one plot
        """
        viz_config_dict = {'ESG': 
                       {'color': 'turquoise',
                    },
                   'Governance': 
                       {'color': 'orange',
                    },
                   'Social': 
                       {'color': 'green',
                    },
                    'Environmental': 
                       {'color': 'pink',
                    }
                  }

        # Base chart for the scores with the legend DataFrame
        score_chart = alt.Chart(self.data_dict['master']).mark_point()

        # Chart for ESG scores
        esg = score_chart.encode(
            x='ESG',
            y='Sector',
            color=alt.value(viz_config_dict['ESG'])
        )

        # Chart for Governance scores
        gov = score_chart.encode(
            x='Governance',
            y='Sector',
            color=alt.value(viz_config_dict['Governance'])
        )

        # Chart for Social scores
        soc = score_chart.encode(
            x='Social',
            y='Sector',
            color=alt.value(viz_config_dict['Social'])
        )

        # Combine the three charts
        combined_chart = esg + gov + soc

        # Create a legend by using point marks and a separate chart
        legend_chart = alt.Chart(legend).mark_point(filled=True).encode(
            y=alt.Y('Score Type:N', axis=alt.Axis(orient='right')),
            color=alt.Color('Color:N', scale=None)
        )

        # Combine the score chart with the legend
        final_chart = (combined_chart | legend_chart).resolve_legend(
            color="independent"
        )

        return final_chart

    def plot_non_missing(self):
        """
        Plots non-missing data for all ESG score types
        """
        scores = self.data_dict[master][['ESG','Environmental','Social','Governance']]
        msno.bar(scores, figsize=(7,3.5), fontsize=12)
        plt.title('Non-missing Bloomberg ESG Scores \nS&P 500 Companies', fontsize = 15)
        plt.xticks()
        plt.show;

        msno.matrix(scores, figsize=(7,3.5), fontsize=12)
        plt.title('Non-missing Bloomberg ESG Scores \nS&P 500 Companies', fontsize = 14)
        plt.show

    def plot_scatter(self,x_score,y_score)
        """
        Plots a scatterplot of two different ESG score types
        """
        sns.scatterplot(data=self.data_dict['master'], y=y_score,x=x_score, hue='Sector', marker='+')

    def plot_score_by_sector(self,score_type)
        """
        Takes in score_type (i.e. "ESG") and returns a plot of that score by sector
        """
        sns.set_theme(style="ticks")
        f, ax = plt.subplots(figsize=(7, 5))
        sns.despine(f)
        sns.histplot(
            self.data_dict['master'],
            x=score_type, hue="Sector",
            multiple="stack",
            palette="light:m_r",
            edgecolor=".3",
            linewidth=.5,
            log_scale=True,
        )
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xticks([2, 4, 6, 8, 10]);
