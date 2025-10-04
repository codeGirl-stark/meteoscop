from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests, pandas as pd, os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = FastAPI()
NASA_API_KEY = "C26JAE1bYDw02arXFa12sAfy8NowS3evx8f03Osi"
CLICK_FILE = "cities.csv"
WEATHER_FILE = "weather.csv"

# Initialize CSVs
if not os.path.exists(CLICK_FILE):
    pd.DataFrame(columns=["city","latitude","longitude"]).to_csv(CLICK_FILE,index=False)
if not os.path.exists(WEATHER_FILE):
    pd.DataFrame(columns=["latitude","longitude","date","temp_max","temp_min","humidity","wind","precip","condition"]).to_csv(WEATHER_FILE,index=False)

def analyze_condition(temp_max,temp_min,humidity,wind,precip):
    messages=[]
    color="green"
    if temp_max>35: messages.append("Très chaud"); color="red"
    if temp_min<10: messages.append("Très froid"); color="blue"
    if wind>15: messages.append("Très venteux"); color="orange"
    if humidity>80: messages.append("Très humide"); color="purple"
    if precip>0: messages.append("Pluie possible"); color="darkblue"
    if not messages: messages.append("Confortable")
    return ", ".join(messages), color

# ML estimation simple pour 30 jours dans le futur
def predict_weather(lat, lon, date_str):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)  # dernière année
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters":"T2M_MAX,T2M_MIN,WS10M,RH2M,PRECTOTCORR",
        "community":"RE",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format":"JSON",
        "api_key": NASA_API_KEY
    }
    r = requests.get(url, params=params)
    if r.status_code!=200: return None

    data = r.json()['properties']['parameter']
    days = list(data['T2M_MAX'].keys())
    X=[]
    Y_temp_max=[]
    Y_temp_min=[]
    Y_wind=[]
    Y_humidity=[]
    Y_precip=[]
    for d in days:
        day_num = int(d[-4:])  # MMDD
        X.append([day_num])
        Y_temp_max.append(data['T2M_MAX'][d])
        Y_temp_min.append(data['T2M_MIN'][d])
        Y_wind.append(data['WS10M'][d])
        Y_humidity.append(data['RH2M'][d])
        Y_precip.append(data['PRECTOTCORR'][d])
    X = np.array(X)
    # ML models
    pred_models={}
    for name, Y in [("temp_max",Y_temp_max),("temp_min",Y_temp_min),("wind",Y_wind),("humidity",Y_humidity),("precip",Y_precip)]:
        model = RandomForestRegressor()
        model.fit(X,Y)
        pred_models[name]=model
    day_num_future = int(datetime.strptime(date_str,"%Y-%m-%d").strftime("%m%d"))
    pred={}
    for name, model in pred_models.items():
        pred[name] = round(model.predict([[day_num_future]])[0],1)
    return pred

@app.get("/interactive_map", response_class=HTMLResponse)
def interactive_map():
    html_content = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Weather Risk Map</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
#map { width: 100%; height: 400px; transition: height 0.3s; }
#map.fullscreen { height: 90vh; }
#controls { margin:5px; display:flex; gap:5px; flex-wrap:wrap; }
</style>
</head>
<body>
<div id="controls">
<input type="text" id="cityInput" placeholder="Enter city">
<input type="date" id="dateInput">
<button onclick="searchCity()">Check Weather</button>
<button onclick="toggleMap()">Toggle Size</button>
</div>
<div id="map"></div>

<script>
var map = L.map('map').setView([6.37, 2.43],6);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19}).addTo(map);

function toggleMap(){
    document.getElementById('map').classList.toggle('fullscreen');
    map.invalidateSize();
}

async function searchCity(){
    let city=document.getElementById('cityInput').value;
    let date=document.getElementById('dateInput').value;
    if(!city || !date){ alert("City and date required"); return; }

    let res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${city}`);
    let data = await res.json();
    if(data.length==0){ alert("City not found"); return; }
    let lat = data[0].lat;
    let lon = data[0].lon;

    let weatherRes = await fetch(`/get_weather?lat=${lat}&lon=${lon}&date=${date}&city=${city}`);
    let info = await weatherRes.text();

    let color="green";
    if(info.includes("Très chaud")) color="red";
    else if(info.includes("Très froid")) color="blue";
    else if(info.includes("Très venteux")) color="orange";
    else if(info.includes("Très humide")) color="purple";
    else if(info.includes("Pluie possible")) color="darkblue";

    let marker = L.circleMarker([lat,lon],{
        radius:8, color:color, fillColor:color, fillOpacity:0.7
    }).addTo(map);
    marker.bindPopup(info).openPopup();

    // JS alert if unfavorable
    if(info.includes("Très") || info.includes("Pluie")) alert("⚠ Weather alert: "+info);

    map.setView([lat,lon],12);
}
</script>
</body>
</html>
    """
    return HTMLResponse(html_content)

@app.get("/get_weather", response_class=HTMLResponse)
def get_weather(lat: float, lon: float, date: str, city: str):
    dt_obj = datetime.strptime(date,"%Y-%m-%d")
    today = datetime.today()
    delta = dt_obj - today
    # future up to 30 days -> ML
    if delta.days>0:
        if delta.days>30: return "<b>Date too far in the future (max 30 days)</b>"
        pred = predict_weather(lat, lon, date)
        if not pred: return "<b>Error predicting future weather</b>"
        temp_max = pred['temp_max']; temp_min=pred['temp_min']
        humidity=pred['humidity']; wind=pred['wind']; precip=pred['precip']
    else:  # past -> NASA POWER
        day_str = dt_obj.strftime("%Y%m%d")
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params={"parameters":"T2M_MAX,T2M_MIN,WS10M,RH2M,PRECTOTCORR",
                "community":"RE","longitude":lon,"latitude":lat,
                "start":day_str,"end":day_str,"format":"JSON","api_key":NASA_API_KEY}
        r = requests.get(url,params=params)
        if r.status_code!=200: return "<b>Error fetching weather data</b>"
        try:
            data = r.json()['properties']['parameter']
            temp_max = list(data['T2M_MAX'].values())[0]
            temp_min = list(data['T2M_MIN'].values())[0]
            humidity = list(data['RH2M'].values())[0]
            wind = list(data['WS10M'].values())[0]
            precip = list(data['PRECTOTCORR'].values())[0]
        except: return "<b>No data available for this date/location</b>"

    condition,color = analyze_condition(temp_max,temp_min,humidity,wind,precip)

    # Save CSV
    df_click = pd.read_csv(CLICK_FILE)
    df_click = pd.concat([df_click, pd.DataFrame([{"city":city,"latitude":lat,"longitude":lon}])],ignore_index=True)
    df_click.to_csv(CLICK_FILE,index=False)

    df_weather = pd.read_csv(WEATHER_FILE)
    df_weather = pd.concat([df_weather, pd.DataFrame([{
        "latitude":lat,"longitude":lon,"date":date,
        "temp_max":temp_max,"temp_min":temp_min,"humidity":humidity,
        "wind":wind,"precip":precip,"condition":condition
    }])],ignore_index=True)
    df_weather.to_csv(WEATHER_FILE,index=False)

    return f"<b>City:</b> {city}<br><b>Date:</b> {date}<br><b>Max Temp:</b> {temp_max}°C<br><b>Min Temp:</b> {temp_min}°C<br><b>Humidity:</b> {humidity}%<br><b>Wind:</b> {wind} m/s<br><b>Precipitation:</b> {precip} mm<br><b>Condition:</b> {condition}"


