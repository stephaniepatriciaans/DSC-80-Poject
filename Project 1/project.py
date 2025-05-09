# project.py


import pandas as pd
import numpy as np
from pathlib import Path

###
from collections import deque
from shapely.geometry import Point
###

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pd.options.plotting.backend = 'plotly'

import geopandas as gpd

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def create_detailed_schedule(schedule, stops, trips, bus_lines):
    merged_df = pd.merge(schedule, stops, on='stop_id')
    merged_df = pd.merge(merged_df, trips, on='trip_id')
    
    merged_df = merged_df[merged_df['route_id'].isin(bus_lines)]

    # Order the df so that bus lines are sorted
    merged_df['route_id'] = pd.Categorical(merged_df['route_id'], categories=bus_lines, ordered=True)
    
    # Sort trip_ids by number of stops and merge into the dataframe
    stop_counts = merged_df.groupby(['route_id', 'trip_id'])['stop_id'].count().reset_index(name="num_stops")
    merged_df = merged_df.merge(stop_counts, on=['route_id', 'trip_id'])
    
    # sort the dataframe by num_stops
    merged_df.sort_values(by=['route_id', 'num_stops', 'trip_id', 'stop_sequence'], inplace=True)
    
    # After sorting, delete the num_stops column
    merged_df.drop(columns=['num_stops'], inplace=True)
    
    # Set index to trip_id
    merged_df.set_index('trip_id', inplace=True)
    return merged_df

def visualize_bus_network(bus_df):
    # Load the shapefile for San Diego city boundary
    san_diego_boundary_path = 'data/data_city/data_city.shp'
    san_diego_city_bounds = gpd.read_file(san_diego_boundary_path)
    
    # Ensure the coordinate reference system is correct
    san_diego_city_bounds = san_diego_city_bounds.to_crs("EPSG:4326")
    
    san_diego_city_bounds['lon'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.x)
    san_diego_city_bounds['lat'] = san_diego_city_bounds.geometry.apply(lambda x: x.centroid.y)
    
    fig = go.Figure()
    
    # Add city boundary
    fig.add_trace(go.Choroplethmapbox(
        geojson=san_diego_city_bounds.__geo_interface__,
        locations=san_diego_city_bounds.index,
        z=[1] * len(san_diego_city_bounds),
        colorscale="Greys",
        showscale=False,
        marker_opacity=0.5,
        marker_line_width=1,
    ))

    color_palette = px.colors.qualitative.Plotly
    unique_route_data = bus_df['route_id'].drop_duplicates()
    colored_routes = {k: color_palette[i % len(color_palette)] for i, k in enumerate(unique_route_data)}
    
    for route in unique_route_data:
        route_df = bus_df[bus_df['route_id'] == route]
        fig.add_trace(go.Scattermapbox(
            lat=route_df['stop_lat'],
            lon=route_df['stop_lon'],
            mode="markers+lines",
            marker=dict(size=6, color=colored_routes[route]),
            line=dict(width=2, color=colored_routes[route]),
            name=f"Bus Line {route}",
            text=route_df['stop_name'],
            hoverinfo="text+name"
        ))

    # Update layout
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": san_diego_city_bounds['lat'].mean(), "lon": san_diego_city_bounds['lon'].mean()},
            zoom=10,
        ),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    return fig

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def find_neighbors(station_name, detailed_schedule):
    station_data = detailed_schedule[detailed_schedule['stop_name'] == station_name]
    if station_data.empty:
        return []
    
    neighbors = set()
    for trip_id in station_data.index.unique():
        trip_data = detailed_schedule.loc[trip_id]
        
        current_sequence = trip_data[trip_data['stop_name'] == station_name]['stop_sequence'].values[0]
        
        next_stop = trip_data[trip_data['stop_sequence'] == current_sequence + 1]
        
        if not next_stop.empty:
            neighbors.add(next_stop['stop_name'].values[0])
    
    return list(neighbors)


def bfs(start_station, end_station, detailed_schedule):
    all_stops = detailed_schedule['stop_name'].unique()
    if start_station not in all_stops:
        return f"Start station {start_station} not found."
    
    if end_station not in all_stops:
        return f"End station '{end_station}' not found."
    
    queue = deque()
    queue.append([start_station])
    visited = set()
    visited.add(start_station)
    
    while queue:
        path = queue.popleft()
        current_station = path[-1]
        
        if current_station == end_station:
            path_details = []
            for stop_name in path:
                stop_details = detailed_schedule[detailed_schedule['stop_name'] == stop_name].iloc[0]
                path_details.append({
                    'stop_name': stop_name,
                    'stop_lat': stop_details['stop_lat'],
                    'stop_lon': stop_details['stop_lon'],
                })
            
            result_df = pd.DataFrame(path_details)
            result_df['stop_num'] = range(1, len(result_df) + 1)
            return result_df[['stop_name', 'stop_lat', 'stop_lon', 'stop_num']]
        
        neighbors = find_neighbors(current_station, detailed_schedule)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return "No path found"

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def simulate_bus_arrivals(tau, seed=12):
    
    np.random.seed(seed) # Random seed -- do not change
    
    # Time keeping
    start = 360
    end = 1440
    n_buses = int((end - start) / tau)
    
    # Random bus times within interval
    bus_times_random = np.random.uniform(start, end, size=n_buses)
    bus_times = np.sort(bus_times_random)
    
    intervals = np.diff(bus_times, prepend=start)
    
    arrival_times = []
    for t in bus_times:
        hours = int(t // 60)
        mins = int(t % 60)
        secs = int((t * 60) % 60)  
        arrival_times.append(f"{hours:02d}:{mins:02d}:{secs:02d}")
    
    return pd.DataFrame({
        'Arrival Time': arrival_times,
        'Interval': intervals
    })


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def simulate_wait_times(arrival_times_df, n_passengers):
    # Midnight --> change arrival times to minutes 
    bus_times = pd.to_datetime(arrival_times_df['Arrival Time'], format='%H:%M:%S')
    bus_minutes = bus_times.dt.hour * 60 + bus_times.dt.minute + bus_times.dt.second / 60
    bus_minutes = bus_minutes.values
    
    if len(bus_minutes) == 0:
        return pd.DataFrame(columns=['Passenger Arrival Time', 
                                     'Bus Arrival Time', 
                                     'Bus Index', 
                                     'Wait Time'])
    
    # Passenger arrival times between 6:00 AM (360) & last bus arrival
    start = 360  # 6 AM --> 6 * 60 = 360 min
    end = bus_minutes[-1]  # Last bus arrival 
    
    passenger_minutes = np.random.uniform(start, end, size=n_passengers)
    passenger_minutes.sort()
    
    # Next bus passenger
    idx = []
    for passenger_time in passenger_minutes:
        for i, bus_time in enumerate(bus_minutes):
            if bus_time >= passenger_time:
                idx.append(i)
                break
    idx = np.array(idx)
    
    # Filter passengers who arrive after the last bus
    valid = idx < len(bus_minutes)
    passenger_minutes = passenger_minutes[valid]
    idx = idx[valid]
    
    wait_times = bus_minutes[idx] - passenger_minutes
    
    # Convert passenger & bus times to HH:MM:SS 
    passenger_times = []
    for t in passenger_minutes:
        hours = int(t // 60)
        mins = int(t % 60)
        secs = int((t - int(t)) * 60)
        passenger_times.append(f"{hours:02d}:{mins:02d}:{secs:02d}")
    
    bus_arrival_times = arrival_times_df.iloc[idx]['Arrival Time'].values
    
    return pd.DataFrame({
        'Passenger Arrival Time': passenger_times,
        'Bus Arrival Time': bus_arrival_times,
        'Bus Index': idx,
        'Wait Time': wait_times
    })


def visualize_wait_times(wait_times_df, timestamp):
    passenger_datetimes = pd.to_datetime(wait_times_df['Passenger Arrival Time'], format='%H:%M:%S')
    passenger_minutes = (passenger_datetimes.dt.hour * 60) + (passenger_datetimes.dt.minute) + (passenger_datetimes.dt.second / 60)
    
    bus_datetimes = pd.to_datetime(wait_times_df['Bus Arrival Time'], format='%H:%M:%S')
    bus_minutes = (bus_datetimes.dt.hour * 60) + (bus_datetimes.dt.minute) + (bus_datetimes.dt.second / 60)

    start = (timestamp.hour * 60) + timestamp.minute
    end = start + 60

    passenger_within_1hr = (passenger_minutes >= start) & (passenger_minutes <= end)
    bus_within_1hr = (bus_minutes >= start) & (bus_minutes <= end)
    
    relative_passenger_minutes = passenger_minutes[passenger_within_1hr] - start
    relative_bus_minutes = pd.Series(bus_minutes[bus_within_1hr]).drop_duplicates().values - start

    wait_times_within = wait_times_df.loc[passenger_within_1hr, 'Wait Time']
    passenger_wait_times = wait_times_within.values  

    fig = go.Figure()

    for passenger_minute, wait_duration in zip(relative_passenger_minutes, passenger_wait_times):
        fig.add_trace(go.Scatter(
            x = [passenger_minute, passenger_minute],
            y = [0, wait_duration],
            mode = 'lines',
            line = dict(color='red', dash='dash'),
            showlegend = False
        ))

    # Markers for passengers
    fig.add_trace(go.Scatter(
        x = relative_passenger_minutes,
        y = passenger_wait_times,
        mode = 'markers',
        marker = dict(color='red', size=6),
        name='Passengers'
    ))

    # Markers for buses
    fig.add_trace(go.Scatter(
        x = relative_bus_minutes,
        y = [0] * len(relative_bus_minutes),
        mode = 'markers',
        marker = dict(color='blue', size=10),
        name = 'Buses'
    ))
    
    if passenger_wait_times.size > 0:
        max_wait = passenger_wait_times.max() * 1.1
    else:
        max_wait = 1
        
    fig.update_layout(
        title = 'Passenger Wait Times in a 60‑Minute Block',
        xaxis = dict(
            title = 'Time (minutes) within the block',
            range = [0, 60],
            dtick = 10
        ),
        yaxis = dict(
            title = 'Wait Time (minutes)',
            range = [0, max_wait+10]
        ),
        showlegend = True,
        template = 'plotly_white'
    )

    return fig