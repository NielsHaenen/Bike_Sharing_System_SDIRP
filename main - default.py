# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 11:32:14 2021
Note, we need the TestData in the same folder to be able to run the code!
"""
import numpy as np
import csv
import math
import copy
from datetime import datetime


# - global information demand matrix
# - global information 

OUTPUT_FLAG = False
BUSY_PARAMETER = 4

STATIONCAPACITY_PARAMETER = 10
TOTALBIKES_PARAMETER = 0.7

VEHICLECAPACITY_PARAMETER = 10
SEED_PARAMETER = 10102021

NITERATIONS_PARAMETER = 5

EPISODELENGTH_PARAMETER = 1  #in days

#This function translates coordinates (latitudes, longitudes) to a distance in kilometers 
def calcDistance(lat1_, lon1_, lat2_, lon2_):
        
    R = 6373.0
    
    lat1 = math.radians(lat1_)
    lon1 = math.radians(lon1_)
    lat2 = math.radians(lat2_)
    lon2 = math.radians(lon2_)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
    

# This class defines a bike sharing station. 
# Its ID is not used, its merely to identify it in the wild
# capacity is the number of bikes that can be stored
# latitude and longitute are coordinates
# name is its name
# currentCap is the current number of vehicles in the racks at the station (during execution)
class Station:
    def __init__(self, ID, capacity, latitude, longitude, name):
        self.ID        = ID
        self.capacity  = capacity
        self.latitute  = latitude
        self.longitude = longitude
        self.name      = name
        self.currentCap = int(TOTALBIKES_PARAMETER * capacity);
                
# This class stores all information of the vehicle in the environment 
# Because decisions are currently a pair where bikes are picked and bikes are dropped we store these from, to locations and their arrival times        
class Vehicle:
    def __init__(self, location, load, arrivalTime):
        self.location         = location        # current location (or where it is assigned a new rebalancing decision)
        self.capacity         = VEHICLECAPACITY_PARAMETER        # capacity of vehicle
        self.load             = load            # current load of vehicle
        
        self.arrivalTimeFrom  = arrivalTime     #-1 if unassigned next
        self.arrivalTimeTo    = arrivalTime     #-1 if unassigned next
        
        self.locationFrom     = -1
        self.locationTo       = -1
    
#This describes an decision.
class Decision:
    def __init__(self, stationFrom, stationTo, demand):
        self.stationFrom = stationFrom;
        self.stationTo = stationTo;
        self.demand    = demand;

#This describes a State
class State:
    def __init__(self, hour, time, arrivalTime, capacities, vehicleCapAvail):
        self.hour = hour
        self.time = time
        self.arrivalTime = arrivalTime
        self.capacities = capacities
        self.vehicleCapAvail = vehicleCapAvail

class Event:
    def __init__(self, quantity, ID, time, isDemand):      
        self.quantity  = quantity
        self.ID        = ID
        self.time      = time
        self.isDemand  = isDemand


class BikerEnv:
      
    def __init__(self, name):
        self.game_over = False;    # Boolean that inidcates if an episode has ended or not
        self.name = name           # Name of the environment
        # self.busyness = BUSY_PARAMETER  # busyness parameter (it scales mean interarrival time between customers, so lower is more demand)
        
        self.current_step = 0 #a counter for the decision epochs/points
        
        self.time = 0         #current time in seconds
        self.objective = 0    #currenta cumulative objective in the episode
        self.rejectedBikes = 0 #total number of demand realization that could not be served
        self.acceptedBikes = 0 #total number of accepted bikes
        
        # list of events that we know will happen (customers bringing back their bike e.g.)
        # an event is merily a time stamp and some extra information what will happen at that time. 
        # NOTE; we cannot use information about future bike drops; we do not KNOW this, so this is basicaly cheating
        
        self.events = []       
     
        #list of bike-sharing stations
        self.stations     = self.readStationList()
        
        #list of bike-sharing stations in their initial state (to restart episode easily)
        self.initStations = copy.deepcopy(self.stations)
        
        #the vehicle object, denoting where the vehicle is
        self.vehicle  = Vehicle(0, 0, 0)
        
        #seed set to be able to replicate results. (this way to set a seet is a bit depreciated - if this does not work we can seek other opportunities)
        np.random.seed(SEED_PARAMETER)

        #arrival rate for orders in an hour period
        self.prob = [0] * 24
        
        ## below are probability distributions for arrivals and travel times
        
        #probability conditional on hour and origin, probaility conditional on hour for origin, mean travel times and std travel times for requests (assumed normal IS IT!?!)
        self.probHourConditional, self.probHour, self.meanTravelTime, self.stdTravelTime = self.ProbDemandPerHour()
        
        #travel time of the vehicle: equal to mean if there is information, otherwise set to 20km/h avg speed.
        self.travelTime = self.setTravelTime()

        #self explanatory time units
        self.dayLength = 3600 * 24
        
        self.episodeLength = EPISODELENGTH_PARAMETER * self.dayLength
        
        #important variable. denotes the hour of the day; probability and demand depends on this. First improvement might be to consider demand differing for each day
        self.hour = 0
        
        #in any case, put a new pick in the eventqueue using interarrival time 
        interArrival, stationFrom = self.generateNewPick()
         
        self.events.append(Event(-1, stationFrom, self.time + interArrival, True))
               
    #These are a couple of functions that set the members correctly. not directly interesting.    
    def setTravelTime(self):
        
        travelTime = np.zeros((24, len(self.stations), len(self.stations)))
        
        for t in range(0, 24):
            for i in range(0, len(self.stations)):
                for j in range(0, len(self.stations)):
                    if (self.meanTravelTime[t][i][j] > 0):
                        travelTime[t][i][j] = self.meanTravelTime[t][i][j] 
                    else:
                        dist = calcDistance(self.stations[i].latitute,
                                            self.stations[i].longitude,
                                            self.stations[j].latitute,
                                            self.stations[j].longitude)
                        
                        travelTime[t][i][j] = dist / 20 * 3600   
        return travelTime
    
    def readStationList(self):
        with open ('TestData.csv') as csv_file:
            
            csv_reader = csv.DictReader(csv_file)
            stationVector = [0] * 10000
            
            stations = []
            
            for row in csv_reader:
                if stationVector[int(row["start station id"])] == 0:
                    stations.append(Station(int(row["start station id"]),
                                            STATIONCAPACITY_PARAMETER,
                                            float(row["start station latitude"]),
                                            float(row["start station longitude"]),
                                            row["start station name"]))    
                
                if stationVector[int(row["end station id"])] == 0:
                    stations.append(Station(int(row["end station id"]),
                                            STATIONCAPACITY_PARAMETER,
                                            float(row["end station latitude"]),
                                            float(row["end station longitude"]),
                                            row["end station name"]))  
                    
                stationVector[int(row["start station id"])] = 1
                stationVector[int(row["end station id"])] = 1          
        return stations
   
    def ProbDemandPerHour(self):
       
        hashTable = [0] * 10000
        
        for station in self.stations:
          ID = station.ID
          index = [ x.ID for x in self.stations ].index(ID)
          hashTable[ID] = index
             
        with open ('TestData.csv') as csv_file:

            csv_reader = csv.DictReader(csv_file)
            demandProb = np.zeros((24, len(self.stations), len(self.stations)))
            
            travelTimeSum         = np.zeros((24, len(self.stations), len(self.stations)))
            travelTimeSumSquared  = np.zeros((24, len(self.stations), len(self.stations)))
            
            for row in csv_reader:
                
                start    = int(row["start station id"])
                startIdx = hashTable[start]

                end    = int(row["end station id"])
                endIdx = hashTable[end]

                time =  row["starttime"].split()
                hour =  int(time[1][:2])
                
                demandProb[hour][startIdx][endIdx] += 1
                travelTimeSum[hour][startIdx][endIdx] += int(row["tripduration"])
                travelTimeSumSquared[hour][startIdx][endIdx] += ((int(row["tripduration"]) * int(row["tripduration"])))
                       
        travelTimeMean = np.zeros((24, len(self.stations), len(self.stations)))
        travelTimeStd  = np.zeros((24, len(self.stations), len(self.stations)))
    
        demandProbOrigin = np.zeros((24, len(self.stations)))
        
        for t in range(0, len(demandProb)):
            for i in range(0, len(demandProb[t])):
                
                totalObs = float(sum(demandProb[t][i]))
                
                if totalObs == 0:
                    totalObs = 1
                    
                demandProbOrigin[t][i] = totalObs;
                
                for j in range(0, len(demandProb[t][i])):
                    
                    nObs = max(1, demandProb[t][i][j])
                        
                    travelTimeMean[t][i][j] = int( travelTimeSum[t][i][j] / nObs)
                    
                    squaredExpectation   = (travelTimeSum[t][i][j] / nObs) * (travelTimeSum[t][i][j] / nObs)
                    ExpectationOfSquares = travelTimeSumSquared[t][i][j]/nObs
                
                    travelTimeStd[t][i][j] = math.sqrt(ExpectationOfSquares - squaredExpectation)
                    
                    demandProb[t][i][j] = demandProb[t][i][j] / totalObs 
                
                ## to ensure it is exactly equaL to 1 (due to rounding floats this might internally go wrong)
                
                if sum(demandProb[t][i]) <= 0.0001:
                    demandProb[t][i][j] = 1
            
                totalObsHour = sum(demandProbOrigin[t])
                             
            for i in range(0, len(demandProb[t])):
                demandProbOrigin[t][i] /= totalObsHour;
               
            ## to ensure it is exactly equaL to 1 (due to rounding floats this might internally go wrong)
            
            demandProbOrigin[t][0] += 1 - sum(demandProbOrigin[t])
            
            self.prob[t] = totalObsHour / 3600 
              
        return demandProb, demandProbOrigin, travelTimeMean, travelTimeStd
    
    def takeDecision(self, decision):
        
        self.hour = (int)((self.time % self.dayLength) / 3600)
        
        if decision.stationFrom == -1:
            return 0;
        
        ## arrivaltime at from depot
        timeToFrom = self.time  + self.travelTime[self.hour][self.vehicle.location][decision.stationFrom]
        timeFromTo = timeToFrom + self.travelTime[self.hour][decision.stationFrom][decision.stationTo]
        
        # the einding location of the previous rebalancing is the new location of assignment
        self.vehicle.location = self.vehicle.locationTo 

        self.vehicle.arrivalTimeFrom = int(timeToFrom)
        self.vehicle.arrivalTimeTo   = int(timeFromTo)
        
        self.vehicle.locationFrom = decision.stationFrom
        self.vehicle.locationTo   = decision.stationTo
    
        self.events.append(Event(-1 * decision.demand, decision.stationFrom, self.vehicle.arrivalTimeFrom, False))
        
        self.events.append(Event(decision.demand, decision.stationTo, self.vehicle.arrivalTimeTo, False))
        
        if OUTPUT_FLAG:
            print ("----- arrival time at from: " + str(timeToFrom))
            print ("----- arrival time at to: " + str(timeFromTo))
            
            print()
            print()
        return 
     
    # This function processes the next event in the list of events self.events
    def nextObservation(self):
        
        if OUTPUT_FLAG:
            print ("-------------------------------------> Next observation")
            
        event = self.events[0]
        self.time = event.time
        #set the hour index
        self.hour = (int)((self.time % self.dayLength) / 3600)
 
        #just for output purposes.
        indicator = ("B" if event.isDemand else "V")
        
        if OUTPUT_FLAG:
            print("-------- Event: " + str(indicator) + 
                  ": " + str(event.quantity) + 
                  " bike at station " 
                  + str(event.ID) 
                  + " at time " 
                  + str(event.time))
            print()
                    
        #this is the cost we will incur by processing the event
        cost = 0
 
        # An event that is not a demand, is a vehicle arrival at a station. if the quantity is negative, it means a pick.
        if (not event.isDemand) and event.quantity < 0:
            
            self.vehicle.location = event.ID 
            
            #check how many bikes we can pick; between making this event and executing there could have been new customers
            amountBikesPicked = min(self.stations[event.ID].currentCap, -1 * event.quantity)
            
            self.stations[event.ID].currentCap -= amountBikesPicked
            
            #we store the number of picked bikes in the vehicle object.
            self.vehicle.load = amountBikesPicked
    
        # this is a vehicle arrival that drops bikes. 
        if (not event.isDemand) and event.quantity >= 0:
            
            #if we drop to much bikes then we incur a penalty of 10 - 
            if (self.vehicle.load + self.stations[event.ID].currentCap > self.stations[event.ID].capacity):
                cost = 10
            
            self.vehicle.location = event.ID
            
            #we add the number of bikes on the vehicle and set that member to zero.
            self.stations[event.ID].currentCap += self.vehicle.load
            self.vehicle.load = 0
                        
        #Customer picks a new bike
        if event.isDemand and event.quantity == -1:
            
            #no capacity, penalty incurred
            if self.stations[event.ID].currentCap == 0:
                
                self.rejectedBikes += 1
                cost = 1
            
            
            #capacity, so decrease currentCap and put drop event in the queue 
            else:
            
                self.stations[event.ID].currentCap += event.quantity
                self.acceptedBikes += 1
               
                #put a drop in the event queue
                travelTime, stationTo = self.generateNewDrop(event.ID)
                              
                self.events.append(Event(1, stationTo, self.time + travelTime, True))
                
           
           #in any case, put a new pick in the eventqueue using interarrival time 
            interArrival, stationFrom = self.generateNewPick()
            
            self.events.append(Event(-1, stationFrom, self.time + interArrival, True))
            
        #customer drops a bike
        if event.isDemand and event.quantity == 1:
            
            #capacity of station exceeded, not allowed: -1
            if self.stations[event.ID].currentCap == self.stations[event.ID].capacity:
              cost = 1
            
            #drop the bike anyhow
            self.stations[event.ID].currentCap += event.quantity
     
        # delete the processed event.
        del self.events[0]
        
        #sort the events based on their time, from small to large
        self.events.sort(key = lambda x: x.time)       
        self.objective += cost
        
        if OUTPUT_FLAG:
            print("Events:")
        
            for ev in self.events:
                print(str(ev.time),  end=' ')
            
            print()
        return cost
        
    # This function takes a single step towards the future by processing a single event from the list of events in the environment
    def step(self, decision):
        
        self.takeDecision(decision)
        self.current_step += 1
    
        cost = self.nextObservation();
        
        self.objective += cost
        
        if self.time > self.episodeLength:
            self.game_over = True
        return cost
    
    # this function resets the environment to their initial object. After calling, an episode can start  
    def reset(self):
        self.game_over = False;
 
        self.current_step = 0
        self.time         = 0
        self.hour         = 0
        self.objective    = 0
        
        #list of events that we know will happen (customers bringing back their bike e.g.)
        self.events = []
        self.vehicle  = Vehicle(0, 0, 0)
        
        self.rejectedBikes = 0
        self.acceptedBikes = 0

        interArrival, stationFrom = self.generateNewPick()
         
        self.events.append(Event(-1, stationFrom, self.time + interArrival, True))
        self.stations = copy.deepcopy(self.initStations)
        return
  
    # This function translates the current environment/reality to an aggregated representation of the reality 
    # This is all the information we base our decision on.
    # Note: to train completely independent of the model you should implement a hash table/ hash function to translate the returned State object to an index,
    # and to let the trainer only work with the index of the state. 
    def getState(self):
        capacities = []
        for station in self.stations:
            if station.currentCap < 0:
                exit(1)
                #if this happens something is wrong.
            capacities.append(station.currentCap)  
        return State(self.hour, self.time, self.vehicle.arrivalTimeTo, capacities, self.vehicle.capacity - self.vehicle.load)
  
    #This function generates a new bike drop. This is only called after a customer has sucessfully gotten a bike.
    def generateNewDrop(self, station):
        to = np.random.choice(range(0, len(self.stations)),
                              1, 
                              replace = False,
                              p = self.probHourConditional[self.hour][station])
                          
        mu = self.meanTravelTime[self.hour][station][to]
        std = self.stdTravelTime[self.hour][station][to]
        
        #impose some minimum travel time to ensure things don't end up negative.
        #indeed, we could have put 0 there, but by no apparent reason it is 60       
        traveltime = max(60, np.random.normal(mu, std, 1))
        
        if OUTPUT_FLAG:
            print("New drop at station" + str(to) + " at time " + str(self.time) + "+" + str(int(traveltime)) + " -  prob: " + str(self.probHour[self.hour][station]))
        return int(traveltime), int(to)
        
    #This function generates a new customer arrival in the environment.
    def generateNewPick(self):
        #the time after which the customer arrives.
        # interArrival = np.random.exponential(self.prob[self.hour] * self.busyness)

        ## generate origin:
        station = np.random.choice(range(0, len(self.stations)),
                                   1,
                                   replace = False,
                                   p = self.probHour[self.hour])
        
        if OUTPUT_FLAG:
            print("New pick at station" + str(station) + " at time " + str(self.time) + "+" + str(int(interArrival)) + " -  prob: " + str(self.probHour[self.hour][station]))
        return int(interArrival), int(station)
    

class BikerTrainer:   
    #this should initilize everything related to storing previous values of decisions/policies etc.
    def __init__ (self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2
        
    # This functions should update the reward associated with the state decision combination
    # Note: this implementation is currently myopic, so this is not required. 
    # Simple Q-learning is implemented here by remembering which decision state pair had a good performance in the feature
    # optimal solutions require exact state and decision representations (impossible for this assignmnet)
    def update(self, state, decision, reward):
      return;

    #This is where the decision happens; now a simple myopic heuristic rule is implemented. This does not consider future demand,
    #nor does it consider any intelligent vehicle routing/assignment decisions or inventory replenishment decisions.
    #For the moment, the vehicle averages the current capacity of the current stations with smallest and heighest capacity.
    def getDecision(self, state):        
        # if bike is on the way, do nothing
        if state.arrivalTime > state.time:
            return Decision(-1, -1, -1)
        
        # the vehicle is available.
        # find station with minimum and maximum number of capacity.
        minStation = state.capacities.index(min(state.capacities))
        maxStation = state.capacities.index(max(state.capacities))
        
        if abs(min(state.capacities) - max(state.capacities)) <= 2:
            return Decision(-1, -1, -1)
        
        #determine how many bikes to take from max to min station
        avg = int(0.5 * (state.capacities[minStation] + state.capacities[maxStation]))
        
        fromMax = max(0,min(state.vehicleCapAvail, min(state.capacities[maxStation] - avg, avg - state.capacities[minStation])));

        #return the decision from minstation, to maxstation, frommax        
       
        return Decision(maxStation, minStation, fromMax)

    
env     = BikerEnv("Test")
trainer = BikerTrainer(1,2)

numberEpisodes = NITERATIONS_PARAMETER
obj = 0
for i in range(0, numberEpisodes):
    
    #print()
    #print()
    #for stat in env.stations:
    #    print(stat.currentCap, end = ' ')
    #print()
    start_time = datetime.now().replace(microsecond=0)   
    while (not env.game_over):
        
        if OUTPUT_FLAG:
            print()
            print()
            print()
            print("NEW ITERATION at time: " + str(env.time))
        
        state   = env.getState()
        decision  = trainer.getDecision(state)        
        reward  = env.step(decision)
      
        trainer.update(state, decision, reward)
        
        if OUTPUT_FLAG:
            print("END ITERATION at time: " + str(env.time))
        
    #for stat in env.stations:
    #    print(stat.currentCap, end = ' ')
        
    end_time = datetime.now()#.replace(microsecond=0)   
    print("Objective of episode = " + str(env.objective)
          + "(" + str(env.acceptedBikes) + "/" + str(env.acceptedBikes + env.rejectedBikes) 
          + " = " 
          + str(int( float(env.acceptedBikes)/float(env.acceptedBikes + env.rejectedBikes) * 100.00)) + "%)"
          + 'Duration: {}'.format(end_time - start_time))
    
    obj += env.objective
    
    env.reset()

print("Total perfomance = " + str(obj/NITERATIONS_PARAMETER))

#stations = env.readStationList()


    

    