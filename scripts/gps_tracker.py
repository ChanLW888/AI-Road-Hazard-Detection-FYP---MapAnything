import pandas as pd
import folium
from folium.plugins import MeasureControl

# ============================================================
# GPS FILE
# ============================================================

GPS_FILE = "/home/lcha0115/bt60_scratch/lcha_data/map-anything/personal_data/camera_lidar_gps_test_101/extracted/gps/position.csv"

df = pd.read_csv(GPS_FILE)

print("Number of GPS points:", len(df))

print(
    "Latitude range:",
    df["latitude_deg"].min(),
    df["latitude_deg"].max()
)

print(
    "Longitude range:",
    df["longitude_deg"].min(),
    df["longitude_deg"].max()
)

# ============================================================
# Remove invalid GPS points
# ============================================================

df = df.dropna(subset=["latitude_deg", "longitude_deg"]).copy()

# Optional: remove obviously invalid coordinates
df = df[
    (df["latitude_deg"].between(-90, 90)) &
    (df["longitude_deg"].between(-180, 180))
].copy()

print("Valid GPS points:", len(df))

# ============================================================
# START / END COORDINATES
# ============================================================

start_lat = df["latitude_deg"].iloc[0]
start_lon = df["longitude_deg"].iloc[0]

end_lat = df["latitude_deg"].iloc[-1]
end_lon = df["longitude_deg"].iloc[-1]

print("\nSTART:")
print("Latitude :", start_lat)
print("Longitude:", start_lon)

print("\nEND:")
print("Latitude :", end_lat)
print("Longitude:", end_lon)

# ============================================================
# CREATE MAP
# ============================================================

center_lat = df["latitude_deg"].mean()
center_lon = df["longitude_deg"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=18,
    tiles="OpenStreetMap",
    control_scale=True
)

# ============================================================
# VEHICLE TRAJECTORY
# ============================================================

trajectory = list(
    zip(
        df["latitude_deg"],
        df["longitude_deg"]
    )
)

folium.PolyLine(
    trajectory,
    color="blue",
    weight=4,
    opacity=0.8,
    tooltip="Vehicle GPS trajectory"
).add_to(m)

# ============================================================
# START MARKER
# ============================================================

folium.Marker(
    [start_lat, start_lon],
    popup=folium.Popup(
        f"""
        <b>START</b><br>
        Latitude: {start_lat:.8f}<br>
        Longitude: {start_lon:.8f}
        """,
        max_width=300
    ),
    tooltip="START",
    icon=folium.Icon(
        color="green",
        icon="play"
    )
).add_to(m)

# ============================================================
# END MARKER
# ============================================================

folium.Marker(
    [end_lat, end_lon],
    popup=folium.Popup(
        f"""
        <b>END</b><br>
        Latitude: {end_lat:.8f}<br>
        Longitude: {end_lon:.8f}
        """,
        max_width=300
    ),
    tooltip="END",
    icon=folium.Icon(
        color="red",
        icon="stop"
    )
).add_to(m)

# ============================================================
# FIRST / LAST GPS POINT DETAILS
# ============================================================

folium.CircleMarker(
    [start_lat, start_lon],
    radius=8,
    color="green",
    fill=True,
    fill_opacity=1
).add_to(m)

folium.CircleMarker(
    [end_lat, end_lon],
    radius=8,
    color="red",
    fill=True,
    fill_opacity=1
).add_to(m)

# ============================================================
# MEASUREMENT TOOL
# ============================================================

m.add_child(MeasureControl())

# ============================================================
# AUTOMATICALLY FIT MAP TO GPS TRACK
# ============================================================

m.fit_bounds([
    [df["latitude_deg"].min(), df["longitude_deg"].min()],
    [df["latitude_deg"].max(), df["longitude_deg"].max()]
])

# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE = "gps_vehicle_map.html"

m.save(OUTPUT_FILE)

print("\nSaved map to:")
print(OUTPUT_FILE)