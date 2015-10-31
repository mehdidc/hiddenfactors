from bokeh.io import vplot
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.browserlib import view
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.plotting import figure, show, output_file
from bokeh.charts import HeatMap, Line, Histogram
from bokeh.io import output_file, show, vplot, gridplot, hplot
from bokeh.palettes import RdBu11 as palette
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy as np
import pandas as pd


colormap =cm.get_cmap("jet")
palette = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]

from collections import defaultdict

from report_generic import get_figures
def save_reports(reports):
    output_file("vary_hidden_factors.html")

    datasets = set(r["dataset"] for r in reports)

    reports = filter(lambda r:r["dataset"]=="font", reports)
    reports = filter(lambda r:"convnet" in r, reports)

    R = reports

    all_figures = []
    R = sorted(R, key=lambda r:r["latent_size"])
    figures = []

    rec_data = OrderedDict()
    acc_data = OrderedDict()
    crosscor_data = OrderedDict()
    corr_data = OrderedDict()
    tools = "pan,wheel_zoom,box_zoom,reset,resize,save"

    for r in R:
        latent_size = r["latent_size"]
        #if latent_size == 5:
        #    continue
        p = figure(title="latent size:{0}".format(latent_size))
        figures.append(p)
        hm = HeatMap(pd.DataFrame(np.abs(np.array(r["hidfactcorr"]))),
                     title="correlation matrix",
                     palette=palette,
                     tools=tools,
                     legend=True)
        figures.append(hm)
        figures.extend(get_figures(r))

        vals = np.array(r["hidfactcorr"])
        vals = np.extract(1 -  np.eye(vals.shape[0]), vals)

        hist = Histogram(vals, bins=50, legend=True, tools=tools)
        figures.append(hist)
        corr_data["latent {0}(m:{1:.4f},s:{2:.3f})".format(latent_size, vals.mean(), vals.std())] = vals
        rec_data["latent {0}".format(latent_size)] = r["rec_valid"]
        acc_data["latent {0}".format(latent_size)] = r["acc_valid"]
        crosscor_data["latent {0}".format(latent_size)] = r["crosscor_valid"]

        #rec_data["epoch"] = r["epoch"]
        #acc_data["epoch"] = r["epoch"]
        #crosscor_data["epoch"] = r["epoch"]


    rec = Line(rec_data, title="reconstruction valid", xlabel="epoch", ylabel="reconstruction", legend=True, tools=tools)
    acc = Line(acc_data, title="accuracy valid", xlabel="epoch", ylabel="accuracy", legend=True, tools=tools)
    crosscor = Line(crosscor_data, title="crosscor valid", xlabel="epoch", ylabel="crosscor", legend=True, tools=tools)
    hist = Histogram(corr_data, title="correlation matrix distribution", bins=50, legend=True, tools=tools)
    figures.append(rec)
    figures.append(acc)
    figures.append(crosscor)
    figures.append(hist)
    all_figures.append(vplot(*figures))

    show(vplot(*all_figures))

if __name__ == "__main__":

    from lightexperiments.light import Light
    light = Light()
    light.launch()
    reports = list(light.db.find({"tags": ["decorrelation", "vary_hidden_factors"]}))
    light.close()
    save_reports(list(reports))
