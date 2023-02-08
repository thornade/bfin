import requests
from lxml import html
import pandas as pd
import json
import datetime
import bfin.handler_stock_analysis as hsa
from dateutil import tz
import warnings
warnings.filterwarnings('ignore')
import pkg_resources
PAR = tz.gettz('Europe/Paris')

def get_bourso_symbol(isin):
    url="https://bourse.boursorama.com/recherche/ajax?query="+isin
    sa=requests.request(method="GET",url=url)
    if "search__list-link" in sa.text:
        return sa.text.split("search__list-link")[1].split("=")[1].split('"')[1].split("/")[-2]
    else:
        return None

def get_info_from_symbol(symbol):
    url="https://www.boursorama.com/cours/"+symbol+"/"
    query_response = requests.request("GET", url,verify=False)
    tree = html.fromstring(query_response.content)
    try:
        eligibilite=tree.xpath("/html/body/main/div/section/header/div/div/div[3]/div[6]/div/ul/li/p[2]")[0].text
        isin=tree.xpath("/html/body/main/div/section/header/div/div/div[1]/div[2]/h2")[0].text.split(" ")[0]
    except:
         return None
    return {"eligibilite":eligibilite,"isin":isin}


### Get info for a specific ISIN from local database
stream = pkg_resources.resource_stream(__name__, "CSV/df_libelles_with_bourso_symbol_all.csv")
libelles=pd.read_csv(stream).drop_duplicates("ISIN")
libelles["isin"]=libelles["ISIN"].apply(lambda x:x[0:12])
def get_bourso_symbol_from_localdb(isin):
    symbol_bourso=libelles[libelles["isin"]==isin]["bourso_symbol"].to_list()
    if len(symbol_bourso) > 0:
        symbol_bourso=symbol_bourso[-1]
    else:
        symbol_bourso=None
    return symbol_bourso
def get_url_bourso(isin):
    ref=get_bourso_symbol_from_localdb(isin)
    if ref is None:
        ref=get_bourso_symbol(isin)
    if ref is not None:
        return "https://bourse.boursorama.com/cours/"+ref+"/"
    else:
        return ""

def get_isin_name_from_localdb(isin):
    name=libelles[libelles["isin"]==isin]["name"].to_list()
    if len(name) > 0:
        name=name[-1]
    else:
        name="!! unknwon !!!"
    return name

# Get opcvm from Boursovie (without fee: 0%)
def get_bourso_opcvm(page=1):
        url_fund_bourso="https://bourse.boursorama.com/bourse/opcvm/recherche/page-"+str(page)+"?fundSearch[classe]=all&fundSearch[critgen]=all&fundSearch[lifeInsurance]=1&fundSearch[saving]=1"
        url_fund_response=requests.request("GET", url_fund_bourso,verify=False)
        tree = html.fromstring(url_fund_response.content)
        for table in tree.xpath("//*/div[1]/div/div[1]/div/table"):
                header = [th.text_content() for th in table.xpath('//th')]
                data = [[td.text_content() for td in tr.xpath('td')]
                        for tr in table.xpath('//tr')]
                data = [row for row in data if len(row)==len(header)]
        fund_links=[]
        for tr in tree.xpath("//*/div[1]/div/div[1]/div/table")[0].xpath("//tr"):
                for td in tr.xpath("td/div/div/a"):
                        fund_links.append({"Name2":td.text_content(),"Link":td.get("href")})
        _df = pd.concat([pd.DataFrame(fund_links),pd.DataFrame(data, columns=header).rename(columns={"Libellé":"Name","Dernier":"Last"})],axis=1)
        _df["Last"]=_df["Last"].apply(lambda x:x.replace('\n',''))
        _df["Var."]=_df["Var."].apply(lambda x:x.replace('\n',''))
        _df["Var. 1 an"]=_df["Var. 1 an"].apply(lambda x:x.replace('\n',''))
        _df["Morningstar"]=_df["Morningstar"].apply(lambda x:int(x) if x != '' else -1)
        _df["Type"]="BVie"
        return _df.drop(columns=["Name2",""])

# Get opcvm from Bourso Partners ( fee > 0%)
def get_bourso_opcvm_partners(page=1):
        url_fund_bourso="https://bourse.boursorama.com/bourse/opcvm/recherche/page-"+str(page)+"?fundSearch[classe]=all&fundSearch[critgen]=all&fundSearch[lifeInsurance]=1&fundSearch[partners]=1"
        url_fund_response=requests.request("GET", url_fund_bourso,verify=False)
        tree = html.fromstring(url_fund_response.content)
        for table in tree.xpath("//*/div[1]/div/div[1]/div/table"):
                header = [th.text_content() for th in table.xpath('//th')]
                data = [[td.text_content() for td in tr.xpath('td')]
                        for tr in table.xpath('//tr')]
                data = [row for row in data if len(row)==len(header)]
        fund_links=[]
        for tr in tree.xpath("//*/div[1]/div/div[1]/div/table")[0].xpath("//tr"):
                for td in tr.xpath("td/div/div/a"):
                        fund_links.append({"Name2":td.text_content(),"Link":td.get("href")})
        _df = pd.concat([pd.DataFrame(fund_links),pd.DataFrame(data, columns=header).rename(columns={"Libellé":"Name","Dernier":"Last"})],axis=1)

        _df["Last"]=_df["Last"].apply(lambda x:x.replace('\n',''))
        _df["Var."]=_df["Var."].apply(lambda x:x.replace('\n',''))
        _df["Var. 1 an"]=_df["Var. 1 an"].apply(lambda x:x.replace('\n',''))
        _df["Morningstar"]=_df["Morningstar"].apply(lambda x:int(x) if x != '' else -1)
        _df["Type"]="Partners"
        return _df.drop(columns=["Name2",""])

# Get opcvm from Bourso others ( fee > 0%)
def get_bourso_opcvm_others(page=1):
        url_fund_bourso="https://bourse.boursorama.com/bourse/opcvm/recherche/autres/page-"+str(page)+"?fundSearch[classe]=all&fundSearch[critgen]=all&fundSearch[lifeInsurance]=1"
        url_fund_response=requests.request("GET", url_fund_bourso,verify=False)
        tree = html.fromstring(url_fund_response.content)
        for table in tree.xpath("//*/div[1]/div/div[1]/div/table"):
                header = [th.text_content() for th in table.xpath('//th')]
                data = [[td.text_content() for td in tr.xpath('td')]
                        for tr in table.xpath('//tr')]
                data = [row for row in data if len(row)==len(header)]
        fund_links=[]
        for tr in tree.xpath("//*/div[1]/div/div[1]/div/table")[0].xpath("//tr"):
                for td in tr.xpath("td/div/div/a"):
                        fund_links.append({"Name2":td.text_content(),"Link":td.get("href")})
        _df = pd.concat([pd.DataFrame(fund_links),pd.DataFrame(data, columns=header).rename(columns={"Libellé":"Name","Dernier":"Last"})],axis=1)
        _df["Last"]=_df["Last"].apply(lambda x:x.replace('\n',''))
        _df["Var."]=_df["Var."].apply(lambda x:x.replace('\n',''))
        _df["Var. 1 an"]=_df["Var. 1 an"].apply(lambda x:x.replace('\n',''))
        _df["Morningstar"]=_df["Morningstar"].apply(lambda x:int(x) if x != '' else -1)
        _df["Type"]="Others"
        return _df.drop(columns=["Name2",""])

# Get tracker from Bourso Partners ( fee > 0%)
def get_bourso_tracker_partners(page=1):
        url_fund_bourso="https://bourse.boursorama.com/bourse/trackers/recherche/page-"+str(page)+"?etfSearch[isEtf]=1&etfSearch[lifeInsurance]=1&etfSearch[partners]=1"

        url_fund_response=requests.request("GET", url_fund_bourso,verify=False)
        tree = html.fromstring(url_fund_response.content)
        for table in tree.xpath("//*/div[1]/div/div[1]/div/table"):
                header = [th.text_content() for th in table.xpath('//th')]
                data = [[td.text_content() for td in tr.xpath('td')]
                        for tr in table.xpath('//tr')]
                data = [row for row in data if len(row)==len(header)]
        fund_links=[]
        for tr in tree.xpath("//*/div[1]/div/div[1]/div/table")[0].xpath("//tr"):
                for td in tr.xpath("td/div/div/a"):
                        fund_links.append({"Name2":td.text_content(),"Link":td.get("href")})
        _df = pd.concat([pd.DataFrame(fund_links),pd.DataFrame(data, columns=header).rename(columns={"Libellé":"Name","Dernier":"Last","Perf. 1 an":"Var. 1 an"})],axis=1)
        _df["Last"]=_df["Last"].apply(lambda x:x.replace('\n',''))
        _df["Var."]=_df["Var."].apply(lambda x:x.replace('\n',''))
        _df["Var. 1 an"]=_df["Var. 1 an"].apply(lambda x:x.replace('\n',''))
        _df["Morningstar"]=_df["Morningstar"].apply(lambda x:int(x) if x != '' else -1)
        _df["Type"]="Partners"
        return _df.drop(columns=["Name2",""])

# Get tracker from Bourso others ( fee > 0%)
def get_bourso_tracker_others(page=1):
        url_fund_bourso="https://bourse.boursorama.com/bourse/trackers/recherche/autres/page-"+str(page)+"?etfSearch[isEtf]=1&etfSearch[lifeInsurance]=1"
        url_fund_response=requests.request("GET", url_fund_bourso,verify=False)
        tree = html.fromstring(url_fund_response.content)
        for table in tree.xpath("//*/div[1]/div/div[1]/div/table"):
                header = [th.text_content() for th in table.xpath('//th')]
                data = [[td.text_content() for td in tr.xpath('td')]
                        for tr in table.xpath('//tr')]
                data = [row for row in data if len(row)==len(header)]
        fund_links=[]
        for tr in tree.xpath("//*/div[1]/div/div[1]/div/table")[0].xpath("//tr"):
                for td in tr.xpath("td/div/div/a"):
                        fund_links.append({"Name2":td.text_content(),"Link":td.get("href")})
        _df = pd.concat([pd.DataFrame(fund_links),pd.DataFrame(data, columns=header).rename(columns={"Libellé":"Name","Dernier":"Last","Perf. 1 an":"Var. 1 an"})],axis=1)
        _df["Last"]=_df["Last"].apply(lambda x:x.replace('\n',''))
        _df["Var."]=_df["Var."].apply(lambda x:x.replace('\n',''))
        _df["Var. 1 an"]=_df["Var. 1 an"].apply(lambda x:x.replace('\n',''))
        _df["Morningstar"]=_df["Morningstar"].apply(lambda x:int(x) if x != '' else -1)
        _df["Type"]="Others"
        return _df.drop(columns=["Name2",""])

# Get fund detail information
# symbol="0P0001KW97" (asset ID)
# length=3650 (days)
def get_fund_history(symbol,length=7300):
    if symbol is None:
        return pd.DataFrame()
    url_base="https://www.boursorama.com/bourse/action/graph/ws/GetTicksEOD?period=0&guid="
    url = url_base + "&symbol="+symbol+"&length="+str(length)
    query_response = requests.request("GET", url,verify=False)
    if "d" in query_response.text:
        df=pd.DataFrame(json.loads(query_response.text)["d"]["QuoteTab"]).rename(columns={"d":"day","o":"opening","h":"higher","l":"lowest","c":"closing","v":"variation"})
        try:
            df=df[df["day"]>0]
            #df["day"]=df["day"].apply(lambda x:str(datetime.datetime(1970, 1, 1)+datetime.timedelta(x)))
            df["day"]=df["day"].apply(lambda x:datetime.datetime(1970, 1, 1,tzinfo=PAR)+datetime.timedelta(x))
        except:
            print(symbol,": warning some df[day] have strange values:",df["day"].to_list())
        return df.rename(columns={"day":"Date"})
    else:
        return pd.DataFrame()

def get_history(symbol,length=365):
    _df=get_fund_history(symbol,length)
    _df=_df.rename(columns={"opening":"Open","lowest":"Low","higher":"High","closing":"Close","variation":"Volume"})
    if len(_df) > 0:
        # compute Price Channel
        _df["PC_low"]=_df["Low"].rolling(window=20).min()
        _df["PC_high"]=_df["High"].rolling(window=20).max()
        _df["PC_high-close%"]=_df.apply(lambda x: round((x["PC_high"]-x["Close"])/x["Close"]*100,2) if x["Close"]!=0 else 0,axis=1)
        _df["PC_high-low%"]=_df.apply(lambda x: round((x["PC_high"]-x["PC_low"])/x["PC_low"]*100,2) if x["PC_low"]!=0 else 0,axis=1)
        # compute Bollinger Bands
        period = 20
        multiplier = 2
        _df['B_high'] = _df['Close'].rolling(period).mean() + _df['Close'].rolling(period).std() * multiplier
        _df['B_low'] =  _df['Close'].rolling(period).mean() - _df['Close'].rolling(period).std() * multiplier
        # compute RSI (need at least 14 days of quotation)
        if len(_df)>14:
            _df=hsa.enrich_RSI(_df)
        # compute Keltner Channel
        return _df
    else:
        return pd.DataFrame()


def _get_xpath(t,x):
    z=t.xpath(x)
    if len(z) > 0:
        return z[0].text
    else:
        return "0"
    
def get_opcvm_info(symbol):
    url="https://www.boursorama.com/bourse/opcvm/cours/"+symbol
    query_response = requests.request("GET", url,verify=False)
    tree = html.fromstring(query_response.content)
    try:
        date=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[1]/ul/li[1]/p[2]").replace("\n","").replace(" ","")
        company=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[1]/ul/li[2]/p[2]/a")
        group=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[1]/ul/li[4]/p[2]/a")
        managers=[x.text.replace("\n","").replace("  ","") for x in tree.xpath("//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[1]/ul/li[3]/ul/li")]
        general_category=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[2]/ul/li[1]/p[2]/a")
        morningstar_category=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[2]/ul/li[2]/p[2]/a")
        amf_category=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[2]/ul/li[3]/p[2]/a")
        legal_type=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[3]/ul/li[1]/p[2]").replace("\n","").replace(" ","")
        investment_type=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[3]/ul/li[2]/p[2]").replace("\n","").replace(" ","")
        gain_affectation=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[3]/ul/li[3]/p[2]").replace("\n","").replace(" ","")
        fund_of_fund=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[3]/ul/li[5]/p[2]").replace("\n","").replace(" ","")
        last_ticket_value=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/div[2]/article/div[1]/div[3]/div[1]/div[4]/div/div[2]/div[3]/ul/li[4]/p[2]").replace("\n","").replace(" ","")
        isin=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section[1]/header/div/div/div[1]/div/h2").split(" - ")[0]
        entry_fee=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[5]/div/div[2]/table/tbody/tr[1]/td[3]").replace("\n","").replace(" ","")
        exit_fee=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[5]/div/div[2]/table/tbody/tr[2]/td[3]").replace("\n","").replace(" ","")
        regular_fee=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[5]/div/div[2]/table/tbody/tr[3]/td[2]").replace("\n","").replace(" ","")
        p1M=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[2]").replace("\n","").replace(" ","")
        p6M=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[3]").replace("\n","").replace(" ","")
        p1Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[4]").replace("\n","").replace(" ","")
        p3Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[5]").replace("\n","").replace(" ","")
        p5Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[6]").replace("\n","").replace(" ","")
        p10Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[9]/div/div[2]/section/div[3]/div[1]/table/tbody/tr[1]/td[7]").replace("\n","").replace(" ","")
    except:
        return tree

    return {
        "creation":date,
        "company":company,
        "group":group,
        "managers":managers,
        "general_category":general_category,
        "morningstar_category":morningstar_category,
        "amf_category":amf_category,
        "legal_type":legal_type,
        "investment_type":investment_type,
        "gain_affectation":gain_affectation,
        "fund_of_fund":fund_of_fund,
        "last_ticket_value":last_ticket_value,
        "isin":isin,
        "entry_fee":entry_fee.replace('%',''),
        "regular_fee":regular_fee.replace('%',''),
        "exit_fee":exit_fee.replace('%',''),
        "p1M":p1M.replace('%',''),
        "p6M":p6M.replace('%',''),
        "p1Y":p1Y.replace('%',''),
        "p3Y":p3Y.replace('%',''),
        "p5Y":p5Y.replace('%',''),
        "p10Y":p10Y.replace('%',''),
        }

def get_tracker_info(symbol):
    try:
        url="https://www.boursorama.com/bourse/trackers/cours/"+symbol
        query_response = requests.request("GET", url,verify=False)
        tree = html.fromstring(query_response.content)
        date=_get_xpath(tree,"/html/body/main/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[1]/ul/li[1]/p[2]").replace("\n","").replace(" ","")
        company=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[1]/ul/li[2]/p[2]").replace("\n","").replace("  ","")
        #managers=[x.replace("\n","").replace("  ","").split(",") for x in tree.xpath("//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[1]/ul/li[3]/p[2]")]
        isin=_get_xpath(tree,"//*/h2[@class=\"c-faceplate__isin\"]").split(" - ")[0]
        morningstar_category=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[1]/ul/li[4]/p[2]").replace("\n","").replace("  ","")
        legal_type=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[2]/ul/li[1]/p[2]").replace("\n","").replace(" ","")
        geo=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[2]/ul/li[3]/p[2] ").replace("\n","").replace("  ","")
        asset_class=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[2]/ul/li[2]/p[2]").replace("\n","").replace("  ","")
        dividende=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[2]/ul/li[4]/p[2]").replace("\n","").replace("  ","")
        entry_fee=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[3]/ul/li[1]/p[2]").replace("\n","").replace("  ","")
        exit_fee=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[3]/ul/li[2]/p[2]").replace("\n","").replace("  ","")
        regular_fee=_get_xpath(tree,"//*[@id=\"main-content\"]/div/section/div[2]/article/div[1]/div[1]/div[2]/div[3]/div/div[2]/div[3]/ul/li[3]/p[2]").replace("\n","").replace("  ","")
        p1M=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[2]").replace("\n","").replace(" ","")
        p3M=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[3]").replace("\n","").replace(" ","")
        p6M=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[4]").replace("\n","").replace(" ","")
        p1Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[5]").replace("\n","").replace(" ","")
        p3Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[6]").replace("\n","").replace(" ","")
        p5Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[7]").replace("\n","").replace(" ","")
        p10Y=_get_xpath(tree,"/html/body/main/div/section[1]/div[2]/article/div[1]/div[7]/div/div[2]/div[1]/table/tbody/tr[1]/td[8]").replace("\n","").replace(" ","")
        return {
                "creation":date,
                "company":company,
                #"managers":managers,
                "morningstar_category":morningstar_category,
                "legal_type":legal_type,
                "geo":geo,
                "isin":isin,
                "asset_class":asset_class,
                "dividende":dividende.replace('%',''),
                "entry_fee":entry_fee.replace('%',''),
                "exit_fee":exit_fee.replace('%',''),
                "regular_fee":regular_fee.replace('%',''),
                "p1M":p1M.replace('%',''),
                "p3M":p3M.replace('%',''),
                "p6M":p6M.replace('%',''),
                "p1Y":p1Y.replace('%',''),
                "p3Y":p3Y.replace('%',''),
                "p5Y":p5Y.replace('%',''),
                "p10Y":p10Y.replace('%','')
                }
    except:
        print("error symbol is:",symbol)

