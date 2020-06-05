# import matplotlib.pyplot as plt
import seaborn as sns
import os

# import matplotlib
# matplotlib.use('Agg')
# matplotlib.style.use('ggplot')

# sns.set()

# plotting the scatterplot
def plot_scatter(col1, col2, path, data, hue):
	fig = sns.scatterplot(x=col1, y=col2, hue=hue, data=data, size=2.5)
	fig_name = "scatterplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the line plot
def plot_line(col1, col2, path, data, hue):
	fig = sns.lineplot(x=col1, y=col2, hue=hue, data=data, size=2.5)
	fig_name = "lineplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the bar chart
def plot_bar(col1, col2, path, data, hue):
	fig = sns.barplot(x=col1, y='col2', data=data, hue=hue)
	fig_name = "barplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


#plotting the histogram
def plot_dist(col1, path, data, hue):
	col_x = data[col1]
	fig = sns.distplot(col_x)
	fig_name = "distplot" + col1 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the box-plots
def plot_box(col1, col2, path, data, hue):
	fig = sns.boxplot(x=col1, y=col2, data=data, hue=hue)
	fig_name = "boxplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name



# plotting the pointplot
def plot_point(col1, col2, path, data, hue):
	fig = sns.pointplot(x=col1, y=col2, data=data, hue=hue)
	fig_name = "pointplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name



# plotting the count plot
def plot_count(col1, path, data, hue):
	fig = sns.countplot(x=col1, data=data, hue=hue)
	fig_name = "countplot" + col1 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name



# plotting the violin-plots
def plot_violin(col1, col2, path, data, hue):
	fig = sns.violinplot(x=col1, y=col2, data=data, hue=hue)
	fig_name = "violinplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the stripplot
def plot_strip(col1, col2, path, data, hue):
	fig = sns.stripplot(x=col1, y=col2, data=data, hue=hue)
	fig_name = "stripplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the swarm-plot
def plot_swarm(col1, col2, path, data, hue):
	fig = sns.swarmplot(x=col1, y=col2, data=data, hue=hue)
	fig_name = "swarmplot" + col1 + col2 + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the heatmap
def heatmap(data):
	fig = sns.heatmap(data=data, annot=True, linewidth=0.5)
	fig_name = "heatmap" + ".png"
	fig.figure.savefig(os.path.join(path, fig_name))
	return fig_name


# plotting the clustermap
# plotting the lmplot
# plotting the regplot


# plotting the subplots


# plotting the pairplot
# plotting the jointplot
# plotting the facetgrid