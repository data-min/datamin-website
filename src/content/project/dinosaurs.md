---
title: "Dinosaur Data Analytics"
description: "Analyzed the museum's dinosaur fossil records to discover insights and and advise the museum on the quality of the data."
pubDate: "Jun 3 2024"
heroImage: "https://i.imgur.com/RhN7SQa.png"
badge: "Challenge"
tags: ["Data Analytics"]
---

## Main

You will be able to find the data and jupyter notebook file from my github.
https://github.com/data-min/dinosaur-data

## 📖 Background

You're applying for a summer internship at a national museum for natural history. The museum recently created a database containing all dinosaur records of past field campaigns. Your job is to dive into the fossil records to find some interesting insights, and advise the museum on the quality of the data.

## 💾 The data

#### You have access to a real dataset containing dinosaur records from the Paleobiology Database ([source](https://paleobiodb.org/#/)):

| Column name  | Description                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------- |
| occurence_no | The original occurrence number from the Paleobiology Database.                                           |
| name         | The accepted name of the dinosaur (usually the genus name, or the name of the footprint/egg fossil).     |
| diet         | The main diet (omnivorous, carnivorous, herbivorous).                                                    |
| type         | The dinosaur type (small theropod, large theropod, sauropod, ornithopod, ceratopsian, armored dinosaur). |
| length_m     | The maximum length, from head to tail, in meters.                                                        |
| max_ma       | The age in which the first fossil records of the dinosaur where found, in million years.                 |
| min_ma       | The age in which the last fossil records of the dinosaur where found, in million years.                  |
| region       | The current region where the fossil record was found.                                                    |
| lng          | The longitude where the fossil record was found.                                                         |
| lat          | The latitude where the fossil record was found.                                                          |
| class        | The taxonomical class of the dinosaur (Saurischia or Ornithischia).                                      |
| family       | The taxonomical family of the dinosaur (if known).                                                       |

The data was enriched with data from Wikipedia.

## 💪 Challenge

Help your colleagues at the museum to gain insights on the fossil record data. Include:

1. How many different dinosaur names are present in the data?
2. Which was the largest dinosaur? What about missing data in the dataset?
3. What dinosaur type has the most occurrences in this dataset? Create a visualization (table, bar chart, or equivalent) to display the number of dinosaurs per type. Use the AI assistant to tweak your visualization (colors, labels, title...).
4. Did dinosaurs get bigger over time? Show the relation between the dinosaur length and their age to illustrate this.
5. Use the AI assitant to create an interactive map showing each record.
6. Any other insights you found during your analysis?

## Install Libraries

```python
!pip3 install pandas
!pip3 install numpy
!pip3 install folium
```

## Import Libraries & Data

```python
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt

dinosaurs = pd.read_csv('/Users/mac/Documents/GitHub/dinosaur-data/dinosaurs.csv')
```

## 1. How many different dinosaur names are present in the data?

```python
unique_dinosaur_names_count = dinosaurs["name"].nunique()
print(unique_dinosaur_names_count)
```

```
1042
```

## 2. Which was the largest dinosaur? What about missing data in the dataset?

```python
largest_dinosaur_row = dinosaurs.loc[dinosaurs["length_m"].idxmax()]

largest_dinosaur_name = largest_dinosaur_row["name"]

print(largest_dinosaur_name)
```

```
Supersaurus
```

## 3. What dinosaur type has the most occurrences in this dataset? Create a visualization.

```python
category_counts = dinosaurs['type'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(category_counts.index, category_counts.values, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Category Distribution')
plt.xticks(rotation=45)

plt.show()
```

![alt text](https://i.imgur.com/pDMjca4.png)

```python
dinosaurs['average_ma'] = (dinosaurs['max_ma'] + dinosaurs['min_ma']) / 2

dinosaurs.head
```

#4. Did dinosaurs get bigger over time? Show the relation between the dinosaur length and their age to illustrate this.

```python
def categorize_age(age):
    return (age // 20) * 20

dinosaurs['age_range'] = dinosaurs['average_ma'].apply(categorize_age)

average_length_by_age_range_type = dinosaurs.groupby(['age_range', 'type'])['length_m'].mean().unstack()

plt.figure(figsize=(12, 8))
average_length_by_age_range_type.plot(kind='line', marker='o', ax=plt.gca())
plt.xlabel("Age Range (Ma)")
plt.ylabel("Average Length (m)")
plt.title("Average Length of Dinosaurs by Age Range and Type")

plt.ylim(0, 25)
plt.yticks(range(0, 26, 5))
plt.xticks(range(60, 259, 20), labels=[f'{i}-{i+20}' for i in range(60, 259, 20)])

plt.legend(title="Dinosaur Type")
plt.grid(True)
plt.show()
```

![alt text](https://i.imgur.com/epzE3PA.png)
As you can find from overall dinosaurs' length average, 4 out of 5 types of dinosaur types shows the deline of the length. It implies that dinosaurs get smaller over time. It shows the shows the negative relation between the dinosaur length and their age. <br>

# 5. Use the AI assitant to create an interactive map showing each record.

```python
from folium.plugins import MarkerCluster

m = folium.Map(location=[20, 0], zoom_start=2)

marker_cluster = MarkerCluster().add_to(m)

for idx, row in dinosaurs.iterrows():
    folium.Marker(
        location=[row['lat'], row['lng']],
        popup=f"Name: {row['name']}<br>Type: {row['type']}<br>Diet: {row['diet']}<br>Length: {row['length_m']} m<br>Age: {row['average_ma']} Ma",
    ).add_to(marker_cluster)

m
```

![alt text](https://i.imgur.com/EdPAw6L.png)

## 6. Any other insights you found during your analysis?

- Data Cleaning: You might want to consider exploring the missing data in the dataset. For instance, how many entries have missing values for "length_m"? Are there patterns in the missing data?

```python
print(dinosaurs.isnull().sum())
```

```python
occurrence_no       0
name                0
diet             1355
type             1355
length_m         1383
max_ma              0
min_ma              0
region             42
lng                 0
lat                 0
class               0
family           1457
average_ma          0
age_range           0
dtype: int64
```

- Diet Analysis: While the focus was on dinosaur size and age, the data also includes diet information. There's a correlation between diet and size, or if certain dinosaur types were predominantly herbivores or carnivores.

```python
average_length_by_diet = dinosaurs.groupby('diet')['length_m'].mean()
print(average_length_by_diet)


plt.figure(figsize=(8, 6))
average_length_by_diet.plot(kind='bar', color=['green', 'red', 'orange'])
plt.xlabel("Diet")
plt.ylabel("Average Length (m)")
plt.title("Average Dinosaur Length by Diet")
plt.xticks(rotation=0)
plt.show()
```

![alt text](https://i.imgur.com/RFSTFzR.png)

```python
diet_composition_by_type = dinosaurs.groupby('type')['diet'].value_counts().unstack(fill_value=0)
diet_composition_by_type_percentages = diet_composition_by_type.div(diet_composition_by_type.sum(axis=1), axis=0) * 100
print(diet_composition_by_type_percentages)


diet_composition_by_type_percentages.plot(kind='bar', stacked=True)
plt.xlabel("Dinosaur Type")
plt.ylabel("Percentage (%)")
plt.title("Diet Composition by Dinosaur Type")
plt.legend(title="Diet")
plt.xticks(rotation=45)
plt.show()
```

![alt text](https://i.imgur.com/I9NCqEN.png)

- Interactive Map Customization: The map is a great way to visualize fossil locations. I customized it further by using different marker colors to represent dinosaur types or diet categories.

```python
def get_marker_color(dinosaur_type):
  color_map = {
      "small theropod": "blue",
      "large theropod": "red",
      "sauropod": "green",
      "ornithopod": "purple",
      "ceratopsian": "orange",
      "armored dinosaur": "brown"
  }
  return color_map.get(dinosaur_type, "gray")
m = folium.Map(location=[20, 0], zoom_start=2)

for idx, row in dinosaurs.iterrows():
  folium.Marker(
      location=[row['lat'], row['lng']],
      popup=f"Name: {row['name']}<br>Type: {row['type']}<br>Diet: {row['diet']}<br>Length: {row['length_m']} m<br>Age: {row['average_ma']} Ma",
      icon=folium.Icon(color=get_marker_color(row["type"]))

m
```

![alt text](https://i.imgur.com/GZKc2qV.png)
