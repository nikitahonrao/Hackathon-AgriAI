# src/weather_utils.py
import requests

API_KEY = "46c269f85b908436bda0a0c2210006d8"  # put your key here

def get_weather(city_name):
    """
    Fetch weather data from OpenWeatherMap for a given city.
    Returns dictionary with temperature (°C), humidity (%), rainfall (mm if available).
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_KEY}&units=metric"
    resp = requests.get(url)
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    main = data.get("main", {})
    weather_info = {
        "temp_c": main.get("temp", None),
        "humidity": main.get("humidity", None),
        "rainfall": None  # daily rainfall may require OneCall API
    }

    # Try to add rainfall if exists
    rain = data.get("rain", {})
    if "1h" in rain:
        weather_info["rainfall"] = rain["1h"]
    elif "3h" in rain:
        weather_info["rainfall"] = rain["3h"]

    return weather_info
