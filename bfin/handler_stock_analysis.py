import re
import pandas as pd
import datetime
import pyecharts.options as opts
from pyecharts.charts import Line,Page,Grid
from pyecharts.components import Table
from pyecharts.globals import ThemeType
import numpy as np
from dateutil import tz
PAR = tz.gettz('Europe/Paris')
from . import handler_boursorama as hbourso
import warnings
warnings.filterwarnings('ignore')

# compute gain between 2 values
def gainp(buy,sell):
    return (sell-buy)/buy*100


# get ticker name for a specific bourso Symbol
# ex:
#    1rTMAA 
#           => 1rT for tracker
#           => MAA for tracker name: Mid Amercian Apartment
#           => we look for MAA.PA (.PA: for PAris trading location)
#    1rT = PARIS
#    1rA = AMSTERDAM
def get_ticker_name_from_symbol(symbol):
    if symbol.startswith("1rT"):
        sym=symbol[3:]+".PA"
    elif symbol.startswith("1rA"):
        sym=symbol[3:]+".AS"
    if symbol=="1rAEXS1":
        sym="EXS1.F"
    return sym


def graph_isin_with_bourso(isin,last=-1,start="2022-01-01"):
    length=(int(datetime.datetime.now().strftime("%Y")) - int(start.split("-")[0])) * 365
    df=hbourso.get_history(symbol=hbourso.get_bourso_symbol_from_localdb(isin),length=length)
    return graph_ticker(last=last,start=start,df_values=df,name=hbourso.get_isin_name_from_localdb(isin))


# Graph a value with price channel
def graph_ticker(last=-1,start="2022-01-01",df_values=None,name=""):
    if df_values is not None:
        df=df_values
    else:
        return None
    feature=opts.ToolBoxFeatureOpts(               
                    save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(title="Download PNG",is_show=True), 
                    restore=opts.ToolBoxFeatureRestoreOpts(is_show=False),          
                    data_view=opts.ToolBoxFeatureDataViewOpts(title="Data source",is_show=True),       
                    data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False),      
                    magic_type=opts.ToolBoxFeatureMagicTypeOpts(is_show=False),      
                    brush=opts.ToolBoxFeatureBrushOpts(type_=0)
    )
    c = (Line(init_opts=opts.InitOpts(width="1500px",height="500px",theme=ThemeType.DARK))
        .add_xaxis(df["Date"].apply(lambda x: str(x).split(" ")[0]).to_list())
        .add_yaxis("Close", df["Close"].to_list(), is_smooth=True,linestyle_opts=opts.LineStyleOpts(color="white",width=2.5))
        .add_yaxis("PC_low", df["PC_low"].to_list(), is_smooth=True,linestyle_opts=opts.LineStyleOpts(color="red"))
        .add_yaxis("PC_high", df["PC_high"].to_list(), is_smooth=True, linestyle_opts=opts.LineStyleOpts(color="green"))
        .add_yaxis("B_low", df["B_low"].to_list(), is_smooth=True,linestyle_opts=opts.LineStyleOpts(color="red"))
        .add_yaxis("B_high", df["B_high"].to_list(), is_smooth=True, linestyle_opts=opts.LineStyleOpts(color="green"))
        .set_series_opts(
                label_opts=opts.LabelOpts(is_show=False),
        )
    )
    nfo=df.iloc[last].to_dict()
    title=name
    subtitle="Last: "+str(nfo["Date"]).split(" ")[0]+"\tLow:"+str(nfo["PC_low"])+"\tClose:"+str(nfo["Close"])+"\tHigh:"+str(nfo["PC_high"])+"\t\tPC_High-Close:"+str(+nfo["PC_high-close%"])+"%"
    subtitle+=" Up/Down: -"+str( round(float((nfo["Close"]-nfo["Low"])/nfo["Close"]*100),2) ) + "% + " + str( round(float((nfo["High"]-nfo["Close"])/nfo["Close"]*100),2) ) +"%"

    c.set_global_opts(
        title_opts=opts.TitleOpts(title=title,subtitle=subtitle),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=False),
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False,axislabel_opts=opts.LabelOpts(rotate=0),is_scale=False,axistick_opts=opts.AxisTickOpts(is_align_with_label=True),),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        legend_opts=opts.LegendOpts(pos_left="right"),
        toolbox_opts=opts.ToolboxOpts(
                pos_top=250,pos_left=0,
                is_show=True,
                orient="vertical",
                feature=feature),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_realtime=True,
                type_="inside",
                start_value=30,
                end_value=70,
                xaxis_index=[0, 1],
            )
        ],
    )

    c2 = (Line(init_opts=opts.InitOpts(width="1500px",height="200px",theme=ThemeType.DARK))
        .add_xaxis(df["Date"].apply(lambda x: str(x).split(" ")[0]).to_list())
        .add_yaxis("PC_High-Close", df["PC_high-close%"], is_smooth=True, linestyle_opts=opts.LineStyleOpts(color="yellow"))
        .add_yaxis("PC_High-Low", df["PC_high-low%"], is_smooth=True, linestyle_opts=opts.LineStyleOpts(color="blue_light"))
        .add_yaxis("RSI", df["RSI"].to_list(), is_smooth=True, linestyle_opts=opts.LineStyleOpts(color="orange"))
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    )
    title2="RSI + PC_High_Close"
    title2=""
    c2.set_global_opts(
        title_opts=opts.TitleOpts(title=title2),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        yaxis_opts=opts.AxisOpts(
                type_="value",
                axistick_opts=opts.AxisTickOpts(is_show=False),
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False,axislabel_opts=opts.LabelOpts(rotate=0),is_scale=False,axistick_opts=opts.AxisTickOpts(is_align_with_label=True),),
        axispointer_opts=opts.AxisPointerOpts(
            is_show=True, link=[{"xAxisIndex": "all"}]
        ),
        legend_opts=opts.LegendOpts(pos_left="50%"),

        toolbox_opts=opts.ToolboxOpts(
                pos_top=250,pos_left=0,
                is_show=True,
                orient="vertical",
                feature=feature),
        datazoom_opts=[
            opts.DataZoomOpts(
                is_realtime=True,
                type_="inside",
                start_value=30,
                end_value=70,
                xaxis_index=[0, 1],
            )
        ],
    )

    g = (
        Grid(init_opts=opts.InitOpts(width="1500px", height="500px",theme=ThemeType.DARK))
        .add(chart=c, grid_opts=opts.GridOpts(pos_left=50, pos_right=50, height="35%"))
        .add(
            chart=c2,
            grid_opts=opts.GridOpts(pos_left=50, pos_right=50, pos_top="55%", height="25%"),
        ))
    return (name,g,df)


def get_page_stock(ref,type="isin",last=-1,start="2021-01-01",PC_bearish=0.4,PC_bullish=19,RSI_bearish=60,RSI_bullish=30,period_ana=90):
    if type == "isin":
        (name,c,df)=graph_isin_with_bourso(ref,last=last,start=start)
    
    #df_bm=calculate_best_moves(df,PC_bearish=PC_bearish,PC_bullish=PC_bullish,RSI_bearish=RSI_bearish,RSI_bullish=RSI_bullish,duration_limit=45)
    # Add low limit to PC & RSI
    df_bm=calculate_best_moves_low_limit(df,PC_bearish=PC_bearish,PC_bullish=PC_bullish,RSI_bearish=RSI_bearish,RSI_bullish=RSI_bullish,duration_limit=45)
    df_bm_ana=analyse_best_moves(df=df,dfbm=df_bm,period=period_ana)
    if len(df_bm_ana) > 0:
        df_bm_ana_display=df_bm_ana[["date_buy","date_sell","best_date_sell","days","best_days","PC_buy","PC_sell","best_PC_sell","diffp","best_diffp","RSI_move","RSI","d-diffp"]]
        df_bm_ana_display["date_sell"]=df_bm_ana_display["date_sell"].apply(lambda x:str(x).split(" ")[0])
        df_bm_ana_display["date_buy"]=df_bm_ana_display["date_buy"].apply(lambda x:str(x).split(" ")[0])
        df_bm_ana_display["best_date_sell"]=df_bm_ana_display["best_date_sell"].apply(lambda x:str(x).split(" ")[0])        
    else:
        df_bm_ana_display=pd.DataFrame()

    if len(df_bm)>0:
        df_bm_display = df_bm[df_bm.covered==0]
        df_bm_display["date_sell"]=df_bm_display["date_sell"].apply(lambda x:str(x).split(" ")[0])
        df_bm_display["date_buy"]=df_bm_display["date_buy"].apply(lambda x:str(x).split(" ")[0])
        df_bm_display=df_bm_display.drop(columns=["year","covered"])
    else:
        df_bm_display=pd.DataFrame()
    
    page = Page(layout=Page.SimplePageLayout,page_title=str(ref)+":"+name)
    table_color = '#333333'
    table = (
        Table()
        .add(headers=list(df_bm_display.columns),rows=df_bm_display.values.tolist(),attributes={
		"style": "background:{}; width:1500px; font-size:14px; color:#C0C0C0;padding:3px;text-align: center;".format(table_color)
	    }).set_global_opts(title_opts=opts.ComponentTitleOpts(title="Best moves for "+name,title_style={"style":"color: white"})))
    table_ana = (
        Table()
        .add(headers=list(df_bm_ana_display.columns),rows=df_bm_ana_display.values.tolist(),attributes={
		"style": "background:{}; width:1500px; font-size:14px; color:#C0C0C0;padding:3px;text-align: center;".format(table_color)
	    }).set_global_opts(title_opts=opts.ComponentTitleOpts(title="Analyse Best moves for "+name,title_style={"style":"color: white"})))

    page.add(
        c,
        table,
        table_ana
    )
    return (name,page,df,df_bm,df_bm_ana)

# compute RSI
def enrich_RSI(df):
    ## 14_Day RSI
    df['Up Move'] = np.nan
    df['Down Move'] = np.nan
    df['Average Up'] = np.nan
    df['Average Down'] = np.nan
    # Relative Strength
    df['RS'] = np.nan
    # Relative Strength Index
    df['RSI'] = np.nan
    ## Calculate Up Move & Down Move
    for x in range(1, len(df)):
        df['Up Move'].iloc[x]=0
        df['Down Move'].iloc[x]=0     
        if df['Close'][x] > df['Close'][x-1]:
            df['Up Move'].iloc[x] = df['Close'][x] - df['Close'][x-1]
        if df['Close'][x] < df['Close'][x-1]:
            df['Down Move'].iloc[x]=abs(df['Close'][x] - df['Close'][x-1])
    ## Calculate initial Average Up & Down, RS and RSI
    df['Average Up'].iloc[14]=df['Up Move'][1:15].mean()
    df['Average Down'].iloc[14]=df['Down Move'][1:15].mean()
    df['RS'].iloc[14]=df['Average Up'][14] / df['Average Down'][14]
    df['RSI'].iloc[14]=100 - (100/(1+df['RS'][14]))
    ## Calculate rest of Average Up, Average Down, RS, RSI
    for x in range(15, len(df)):
        df['Average Up'].iloc[x]=(df['Average Up'][x-1]*13+df['Up Move'][x])/14
        df['Average Down'].iloc[x]=(df['Average Down'][x-1]*13+df['Down Move'][x])/14
        df['RS'].iloc[x]=df['Average Up'][x] / df['Average Down'][x]
        df['RSI'].iloc[x]=100 - (100/(1+df['RS'][x]))
    return df


# find best moves based on strategy:
# Bullish = RSI < 30% && Price Channel HC% > 19%
# Bearish = RSI > 70% && Price Channel HC% =~ 0%
# duration_limit = max number of days to keep the stock (with this strategy after 90 days it s rarely usefull to keep it) 90 days = 60 Working days
def calculate_best_moves(df,PC_bullish=19,PC_bearish=0.1,RSI_bullish=30,RSI_bearish=70,duration_limit=60):
    if ("PC_high-close%" in df.columns) & ("RSI" in df.columns):
        ldiff=[]
        for uc in df[(df["PC_high-close%"]>=PC_bullish) & (df["RSI"]<=RSI_bullish)].reset_index()["index"].to_list():
            _d=df[((df.index == uc)) | ((df.index > uc) & (df["PC_high-close%"] <= PC_bearish) & (df["RSI"]>=RSI_bearish)) | ((df.index > uc + duration_limit)) ].head(2)
            _e=_d.reset_index()
            diff=_e.iloc[-1]["Close"]-_e.iloc[0]["Close"]
            diffp=(_e.iloc[-1]["Close"]-_e.iloc[0]["Close"]) / _e.iloc[0]["Close"] * 100
            days=(_e.iloc[-1]["Date"]-_e.iloc[0]["Date"]).days
            ldiff.append({  "date_buy":_e.iloc[0]["Date"],
                            "date_sell":_e.iloc[-1]["Date"] if len(_e) > 1 else None,
                            "close_buy":_e.iloc[0]["Close"],
                            "close_sell":_e.iloc[-1]["Close"] if len(_e) > 1 else None,
                            "PC_buy":_e.iloc[0]["PC_high-close%"],
                            "RSI_buy":_e.iloc[0]["RSI"],
                            "PC_sell":_e.iloc[-1]["PC_high-close%"] if len(_e) > 1 else None,
                            "diff":diff,
                            "diffp":diffp,
                            "days":days,
                            "year":str(_e.iloc[0]["Date"]).split("-")[0]})

        df_bm=pd.DataFrame(ldiff)
        if len(df_bm) > 0:
            df_bm["covered"]=0
            df_bm=df_bm.reset_index().rename(columns={"index":"ind"})
            # lookup unfinished move at the end
            first_unfinish=df_bm[df_bm["date_sell"].isna()]["ind"].to_list()
            if len(first_unfinish) > 0:
                df_bm["covered"]=df_bm.apply(lambda x: 1 if (x["ind"] > first_unfinish[0]) else x["covered"],axis=1)
            (u_index,u_row)=[x for x in df_bm[(df_bm.covered==0)].head(1).iterrows()][0]
            while u_index is not None and u_row["date_sell"] is not None:
                df_bm["covered"]=df_bm.apply(lambda x : 1 if ((x["ind"] > u_row["ind"]) & (x["date_buy"] <= u_row["date_sell"])) else x["covered"],axis=1)
                try:
                    (u_index,u_row)=[x for x in df_bm[(df_bm["covered"]==0)&(df_bm["ind"]>u_row["ind"])].head(1).iterrows()][0]
                except:
                    u_index = None
            df_bm=df_bm.drop(columns=["ind"])
            df_bm["RSI_buy"]=df_bm["RSI_buy"].apply(lambda x:int(x))
            df_bm["diff"]=df_bm["diff"].apply(lambda x:round(x,2))
            df_bm["diffp"]=df_bm["diffp"].apply(lambda x:round(x,2))
            df_bm["Win/Day%"]=df_bm.apply(lambda x: int(x["diffp"]/x["days"]*365) if x["days"]>0 else 0,axis=1)
            df_bm["Price_high"]=df_bm.apply(lambda x: df[(df["Date"]>x["date_buy"])&(df["Date"]<x["date_sell"])]["Close"].max() if x["date_sell"] is not None else df[df["Date"]>x["date_buy"]]["Close"].max(),axis=1) 
            df_bm["Price_low"]=df_bm.apply(lambda x: df[(df["Date"]>x["date_buy"])&(df["Date"]<x["date_sell"])]["Close"].min() if x["date_sell"] is not None else df[df["Date"]>x["date_buy"]]["Close"].min(),axis=1) 
            df_bm["Gain_high"]=df_bm.apply(lambda x: str(round(((x["Price_high"]-float(x["close_buy"]))/float(x["close_buy"]) * 100),2))+"%",axis=1)
            df_bm["Gain_low"]=df_bm.apply(lambda x: str(round(((x["Price_low"]-float(x["close_buy"]))/float(x["close_buy"]) * 100),2))+"%",axis=1)
            df_bm["RSI_max"]=df_bm.apply(lambda x: df[(df["Date"]>x["date_buy"])&(df["Date"]<x["date_sell"])]["RSI"].max() if x["date_sell"] is not None else df[df["Date"]>x["date_buy"]]["RSI"].max(),axis=1) 
            df_bm["PC_HC_low"]=df_bm.apply(lambda x: df[(df["Date"]>x["date_buy"])&(df["Date"]<x["date_sell"])]["PC_high-close%"].min() if x["date_sell"] is not None else df[df["Date"]>x["date_buy"]]["PC_high-close%"].min(),axis=1) 
            df_bm["RSI_max"]=df_bm["RSI_max"].fillna(0)
            df_bm["RSI_max"]=df_bm["RSI_max"].apply(lambda x: int(x))

        return df_bm#[df_bm["covered"]==0]
    else:
        return pd.DataFrame()

# add a low indicator control to calculate_best_moves to avoid engaging on low PC_bullish (we keep 19 as a minimum even if the quantile give a lower value)
def calculate_best_moves_low_limit(df,PC_bullish=19,PC_bearish=0.1,RSI_bullish=30,RSI_bearish=70,duration_limit=60):
    if PC_bullish < 21:
        PC_bullish = 21
    return calculate_best_moves(df,PC_bullish=PC_bullish,PC_bearish=PC_bearish,RSI_bullish=RSI_bullish,RSI_bearish=RSI_bearish,duration_limit=duration_limit)

# find information around selling date to see if we can improve win & reduce loss
def analyse_best_moves(df,dfbm,period=30):
    lana=[]
    if len(dfbm) > 0:
        for st in dfbm[dfbm.covered == 0]["date_buy"].to_list():
            nh=df[df.index == df[(df.Date > st) & (df.Date < st+datetime.timedelta(period))]["Close"].idxmax()][["Date","Close","PC_high-close%","PC_high-low%","RSI"]]
            nh["BM_id"]=dfbm[dfbm["date_buy"]==st].index[0]
            nh["best_diff"]=nh.apply(lambda x: x["Close"] - dfbm.iloc[x["BM_id"]]["close_buy"] ,axis=1)
            nh["best_diffp"]=nh.apply(lambda x: (x["Close"] - dfbm.iloc[x["BM_id"]]["close_buy"]) / dfbm.iloc[x["BM_id"]]["close_buy"] * 100 ,axis=1)
            
            #lana.append(nh)
            nhj=dfbm[dfbm.covered==0].reset_index().merge(nh,left_on="index",right_on="BM_id")
            nhj=nhj.rename(columns={"PC_high-close%":"best_PC_sell"})
            nhj["d-diffp"]=nhj["best_diffp"]-nhj["diffp"]
            nhj["RSI_move"]=nhj.apply(lambda x : df[df.Date == x["date_sell"]]["RSI"].to_list(),axis=1)
            nhj["RSI_move"]=nhj["RSI_move"].apply(lambda x:round(x[0],2) if len(x) > 0 else None)
            nhj["best_days"]=nhj.apply(lambda x: (x["Date"]-x["date_buy"]).days,axis=1)
            lana.append(nhj)
        ret = pd.concat(lana).rename(columns={"Date":"best_date_sell"})
        ret["best_diff"]=ret["best_diff"].apply(lambda x: round(x,2))
        ret["best_diffp"]=ret["best_diffp"].apply(lambda x: round(x,2))
        ret["d-diffp"]=ret["d-diffp"].apply(lambda x: round(x,2))
        ret["diffp"]=ret["diffp"].apply(lambda x: round(x,2))
        ret["RSI"]=ret["RSI"].apply(lambda x: round(x,2))

        return ret #pd.concat(lana).rename(columns={"Date":"best_date_sell"})
    return pd.DataFrame()

def get_Close_RSI_PC_period(df,dfbm_ana,moveid):
    start=dfbm_ana.iloc[moveid]["date_sell"]
    end=dfbm_ana.iloc[moveid]["Date"]
    return df[["Date","Close","RSI","PC_high-close%"]][(df.Date >= start)&(df.Date <= end)].T

def change_bg_color(filename,color="#000000"):
    with open(filename, 'r+') as f:
        text = f.read()
        text = re.sub('<body>', '<body bgcolor="'+color+'">', text)
        f.seek(0)
        f.write(text)
        f.truncate()

def viewit(isin,start="2021-01-01"):
    RSI_bearish=75
    RSI_bullish=30
    PC_bearish=0
    PC_bullish=20

    (n,p,df,dfbm,dfbm_ana)=get_page_stock(ref=isin,type="isin",start=start,PC_bearish=PC_bearish,PC_bullish=PC_bullish,RSI_bearish=RSI_bearish,RSI_bullish=RSI_bullish,period_ana=60)
    pPCHC_Bearish=int(df["PC_high-close%"].quantile(0.03))
    pPCHC_Bullish=int(df["PC_high-close%"].quantile(0.87))
    pRSI_Bullish=            int(df["RSI"].quantile(0.30))
    pRSI_Bearish=            int(df["RSI"].quantile(0.80))
    (n,p,df,dfbm,dfbm_ana)=get_page_stock(ref=isin,type="isin",start=start,PC_bearish=pPCHC_Bearish,PC_bullish=pPCHC_Bullish,RSI_bearish=pRSI_Bearish,RSI_bullish=pRSI_Bullish,period_ana=60)

    print("pRSI_Bearish=",pRSI_Bearish," \t pRSI_Bullish=",pRSI_Bullish," \t pPCHC_Bearish=",pPCHC_Bearish, " \t pPCHC_Bullish=",pPCHC_Bullish)
    if len(dfbm_ana) > 0:
        print("DFBM ANA Gain:",dfbm_ana["best_diffp"].sum(),"DB BM Gain:",dfbm_ana["diffp"].sum())
    else:
        print("No move no Gain :(")
    return p