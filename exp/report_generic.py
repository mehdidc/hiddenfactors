from bokeh.io import vplot
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.browserlib import view
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_file, show, vplot
from bokeh.charts import Line

def get_figures(e):

    objects = []
    
    p = figure(title="accuracy")
    p.line(e["epoch"], e["acc_train"], name="train", legend="train", color="blue")
    p.line(e["epoch"], e["acc_valid"], name="valid", legend="valid", color="green")
    objects.append(p)

    p = figure(title="reconstruction")
    p.line(e["epoch"], e["rec_train"], name="train", legend="train", color="blue")
    p.line(e["epoch"], e["rec_valid"], name="valid", legend="valid", color="green")
    objects.append(p)

    p = figure(title="cross correlation")
    p.line(e["epoch"], e["crosscor_train"], name="train", legend="train", color="blue")
    p.line(e["epoch"], e["crosscor_valid"], name="valid", legend="valid", color="green")
    objects.append(p)

    return objects
