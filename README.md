# A Locational Demand Model for Bike-Sharing

## Project Description
This repository contains relevant source code to reproduce all experiments in the paper [A Locational Demand Model for Bike-sharing](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3311371). If you use any of the material here, please include a reference to the paper and this webpage.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [locations_sim_EM_MM.ipynb](#locations_sim_em_mmipynb)
  - [EM_MM_comparison.ipynb](#em_mm_comparisonipynb)
  - [locations_sim_allin.ipynb](#locations_sim_allinipynb)
  - [locations_sim_discovery.ipynb](#locations_sim_discoveryipynb)
  - [bikesharing_seattle_data.ipynb](#bikesharing_seattle_dataipynb)
  - [block_partition_test.ipynb](#block_partition_testipynb)

## Installation

To install the necessary packages for this project, you can use pip to install the requirements file:
```
pip install -r requirements.txt
```
Once you install all the required packages, you can open the Jupyter notebooks and run the code.

## Usage

### locations_sim_EM_MM.ipynb

The `locations_sim_EM_MM.ipynb` notebook implements the EM algorithm and the MM algorithm in estimating the location weights given that the underlying locations are in a large candidate location set. 

### EM_MM_comparison.ipynb
The `EM_MM_comparison.ipynb` notebook compares the performance between the EM and MM algorithms. The notebook reproduces Table 1 in the paper.

### locations_sim_allin.ipynb

The `locations_sim_allin.ipynb` notebook demonstrates the performance of the all-in algorithm proposed in the paper. 

### locations_sim_discovery.ipynb

The `locations_sim_discovery.ipynb` notebook demonstrates the performance of the location-discovery algorithm (for both single and batch modes) proposed in the paper. 

### bikesharing_seattle_data.ipynb

The `bikesharing_seattle_data.ipynb` notebook carries out an analysis based on real data. The original data set records all bookings in a bike-sharing company in the Seattle region during July and August 2019. The data are preprocessed to reserve bookings only around Seattle’s downtown area where bike rental is popular and the population is dense. We implement a mixed-effects model, the all-in algorithm, and the location-discovery algorithm on this data set.

### block_partition_test.ipynb
The `block_partition_test.ipynb` notebook measure the accuracy in both the training and testing sets based on the Seattle data. It compares the predicted bookings versus the actual bookings under 100 different service region partitions.

## Seattle Dockless Bike-Sharing Data Description

The data set records relevant booking information in a bike-sharing company in the Seattle region during July and August 2019. It tracks the movement of each bike in the system, including the bike id, the time of the recorded data, the longitude and latitude of the bike, and the state of the bike. The dataset can be used to analyze bike usage patterns, identify areas of high demand for bike rentals, and track the availability of bikes in real-time. Specifically, the dataset has the following columns:

1. `index`: Unique identifier for each data record
2. `bike_id`: Unique identifier for each bike in the system
3. `time`: Time at which the data entry was recorded
4. `lng`: Longitude coordinate of the bike's location
5. `lat`: Latitude coordinate of the bike's location
6. `state`: Indicates the state of the bike, which can be one of the following:
   - `available`: The bike is available for rent
   - `rent start`: The bike has been rented and the rental period has started
   - `rent finish`: The bike has been returned and the rental period has ended

   
