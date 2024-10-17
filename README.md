# Bike_Sharing_System_SDIRP

Key Components:

1. Stations:
   Each station has a soft maximum capacity. Exceeding this capacity incurs a penalty but is still allowed.

2. Vehicle:
   The vehicle used for redistribution has a hard capacity constraint and cannot exceed it when carrying bikes between stations.

3. Travel Time:
   The travel time between stations is based on the mean travel time of bikers. When no prior data exists, it is assumed that the vehicle travels at a speed of 20 km/h.
   The system models travel times as normally distributed, with a minimum travel time of 60 seconds.

4. BikerEnv (Environment Class):
   This class represents the environment and the dynamics of the bike-sharing system.
   It simulates the events happening in real-time, such as bikes being picked up or dropped off, customers arriving to rent or return bikes, and the vehicle transporting bikes between stations.

5. BikerTrainer (Trainer Class):
   This is the decision-making component responsible for determining the next action to take based on the current state of the environment.
   The current strategy involves moving bikes from the station with the most bikes to the station with the fewest, trying to balance the bike distribution.

6. Events:
   Each event is time-stamped and represents actions such as:
   A vehicle arriving to pick up or drop off bikes at a station.
   A customer renting or returning a bike.

7. Reward System:
   A penalty of 1 is incurred for not being able to serve a customer (i.e., when a customer wants a bike, but the station has none).
   A penalty of 10 is incurred when a vehicle drops more bikes than the station can accommodate.

Problem Breakdown:

To improve the current solution, the objective is to:

    1. Optimize the redistribution of bikes to minimize unmet demand (i.e., when stations run out of bikes) and avoid overfilling stations.
    2. Incorporate uncertainty in both customer behavior (bike rentals/returns) and travel times, which are modeled as stochastic (normally distributed).
